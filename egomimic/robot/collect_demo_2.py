#!/usr/bin/env python3
"""
This script collects demonstrations from the robot using a VR controller.

Refactored version of collect_demo.py that fixes:
1. Engagement check causing zero values for non-engaged arm
2. Data being appended multiple times per tick
3. Async state inconsistency from multiple get_* calls

Key changes:
- Single atomic snapshot of robot state per tick
- Both arms' data populated regardless of engagement
- Data appended exactly once per tick
- Cleaner separation of data collection from robot control
"""

import os
import sys
import time
import numpy as np
import copy
import cv2
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# Add path to oculus_reader if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "oculus_reader"))
from oculus_reader import OculusReader

# Import local modules
from robot_utils import RateLoop

# Add path to robot_interface
sys.path.append(os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))
from robot_interface import ARXInterface, YAMInterface
from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver

# Add i2rt to path for YAM robot utilities
sys.path.insert(0, os.path.expanduser("~/i2rt"))
from i2rt.robots.utils import GripperType
# PyRoKi-based kinematics (replaces i2rt Kinematics)
from pyroki_kinematics import PyRoKiKinematics


# ------------------------- Configuration -------------------------

# Control parameters
DEFAULT_FREQUENCY = 30.0  # Hz
POSITION_SCALE = 1.0  # Scale factor for position deltas
ROTATION_SCALE = 1.0  # Scale factor for rotation deltas

# Velocity limits (per tick) - 0 = disabled
DEFAULT_MAX_DELTA_POS = 0.0  # meters/tick
DEFAULT_MAX_DELTA_ROT_DEG = 0.0  # degrees/tick

# Headset orientation correction (when headset cameras face user instead of away)
# This flips X and Z axes to correct for 180° rotation around Y (vertical) axis
HEADSET_FLIPPED = True  # Set True if headset is worn backwards (cameras facing user)

# Dead-zone thresholds (to filter out jitter)
POS_DEAD_ZONE = 0.002  # meters
ROT_DEAD_ZONE_RAD = np.deg2rad(0.8)  # radians

R_YPR_OFFSET = np.array([
  [ 0.66509066, -0.16738938,  0.72776041],
  [ 0.22521625,  0.97413813,  0.0182356 ],
  [-0.71199161,  0.15177514,  0.68558898],
], dtype=np.float64)

L_YPR_OFFSET = np.array([
  [0.6785254459380761, 0.036920397978411894, 0.7336484876476287],
  [-0.05616599291181174, 0.9984199955834385, 0.0017010759532792748],
  [-0.7324265153957544, -0.04236031907675143, 0.6795270435479006],
], dtype=np.float64)


NEUTRAL_ROT_OFFSET_R = np.eye(3)
NEUTRAL_ROT_OFFSET_L = np.eye(3)
YPR_VEL = [1.5, 1.5, 1.5]  # rad/s
YPR_RANGE = [2, 2, 2]

# Trigger thresholds for engagement detection
TRIGGER_ON_THRESHOLD = 0.8
TRIGGER_OFF_THRESHOLD = 0.2

# Gripper thresholds for ARX robot
GRIPPER_OPEN_VALUE = 0.08
GRIPPER_CLOSE_VALUE = -0.018
GRIPPER_WIDTH = GRIPPER_OPEN_VALUE - GRIPPER_CLOSE_VALUE
GRIPPER_VEL = 1  # m/s gripper width is normally around 0.08m

# Gripper thresholds for YAM robot (normalized 0-1)
YAM_GRIPPER_OPEN_VALUE = 1.0   # fully open
YAM_GRIPPER_CLOSE_VALUE = 0.0  # fully closed
YAM_GRIPPER_WIDTH = YAM_GRIPPER_OPEN_VALUE - YAM_GRIPPER_CLOSE_VALUE

# Demo recording
DEMO_DIR = "./demos"
MAX_DEMO_LENGTH = 10000  # Maximum number of steps per demo


# ------------------------- Helper Functions -------------------------


def load_velocity_limits_from_config():
    """Load velocity limits from configs_yam.yaml."""
    import yaml
    config_paths = [
        os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/config/configs_yam.yaml"),
        "/home/robot/robot_ws/egomimic/robot/eva/eva_ws/src/config/configs_yam.yaml",
    ]
    for cfg_path in config_paths:
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                    if "vr_teleop" in cfg:
                        vr_cfg = cfg["vr_teleop"]
                        max_pos = vr_cfg["max_delta_pos"] if "max_delta_pos" in vr_cfg else DEFAULT_MAX_DELTA_POS
                        max_rot = vr_cfg["max_delta_rot_deg"] if "max_delta_rot_deg" in vr_cfg else DEFAULT_MAX_DELTA_ROT_DEG
                        return max_pos, np.deg2rad(max_rot)
            except Exception:
                pass
    return DEFAULT_MAX_DELTA_POS, np.deg2rad(DEFAULT_MAX_DELTA_ROT_DEG)


def clamp_delta_pos(dpos: np.ndarray, max_delta: float) -> np.ndarray:
    """Clamp position delta magnitude. Returns original if max_delta <= 0."""
    if max_delta <= 0:
        return dpos
    norm = np.linalg.norm(dpos)
    return dpos if (norm <= max_delta or norm < 1e-9) else dpos * (max_delta / norm)


def clamp_delta_rot(rotvec: np.ndarray, max_rad: float) -> np.ndarray:
    """Clamp rotation vector angle. Returns original if max_rad <= 0."""
    if max_rad <= 0:
        return rotvec
    angle = np.linalg.norm(rotvec)
    return rotvec if (angle <= max_rad or angle < 1e-9) else (rotvec / angle) * max_rad


def se3_to_xyzxyzw(se3):
    """Convert SE(3) transformation matrix (4x4) to position and quaternion."""
    rot = se3[:3, :3]
    xyzw = R.from_matrix(rot).as_quat()
    xyz = se3[:3, 3]
    return xyz, xyzw


def xyzxyzw_to_se3(xyz, xyzw):
    """
    Convert position (xyz) and quaternion (xyzw) to SE(3) 4x4 transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = R.from_quat(xyzw).as_matrix()
    T[:3, 3] = xyz
    return T


def flip_roll_only(R_i, up=np.array([0.0, 0.0, 1.0]), add_pi=True):
    # body axes from R_i (columns)
    x = R_i[:, 0]
    y = R_i[:, 1]
    if abs(x @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])

    y0 = up - (up @ x) * x
    y0 /= np.linalg.norm(y0)
    z0 = np.cross(x, y0)

    c = y @ y0
    s = y @ z0
    y_flipped = c * y0 - s * z0
    z_flipped = np.cross(x, y_flipped)

    R_out = np.column_stack([x, y_flipped, z_flipped])

    if add_pi:
        # 180° about body X (roll): leaves x col, flips y/z cols
        R_out = R_out @ np.diag([1.0, -1.0, -1.0])
        # equivalently: R_out[:, 1:] *= -1

    return R_out


def safe_rot3_from_T(T, ortho_tol=1e-3, det_tol=1e-3):
    Rm = np.asarray(T, dtype=float)[:3, :3]
    if Rm.shape != (3, 3) or not np.all(np.isfinite(Rm)):
        return np.eye(3)
    det = np.linalg.det(Rm)
    if det <= 0 or abs(det - 1.0) > det_tol:
        return np.eye(3)
    if np.linalg.norm(Rm.T @ Rm - np.eye(3), ord="fro") > ortho_tol:
        return np.eye(3)
    return Rm


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion in XYZW format."""
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def clip_ypr(ypr, clipped_bound) -> np.ndarray:
    ypr_range = np.array(clipped_bound)
    clipped_ypr = np.clip(np.array(ypr), -ypr_range, ypr_range)
    return clipped_ypr


def limit_delta_quat_by_rate(
    delta_quat_xyzw: np.ndarray, max_rate_rad_s: float, dt: float
) -> np.ndarray:
    # Limit the angular magnitude of the delta quaternion to max_rate * dt
    R_delta = R.from_quat(delta_quat_xyzw)  # xyzw
    rotvec = R_delta.as_rotvec()  # axis * angle
    angle = np.linalg.norm(rotvec)
    max_angle = max_rate_rad_s * dt
    if angle > max_angle and angle > 1e-12:
        rotvec = rotvec * (max_angle / angle)
    return R.from_rotvec(rotvec).as_quat()  # xyzw


def quat_xyzw_to_wxyz(qxyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from XYZW to WXYZ format."""
    return np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(qwxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from WXYZ to XYZW format."""
    return np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]], dtype=np.float64)


def pose_from_T(T: np.ndarray):
    """Extract position and quaternion (WXYZ) from transformation matrix."""
    pos = T[:3, 3].astype(np.float64)
    rot_mat = safe_rot3_from_T(T[:3, :3])
    q_xyzw = R.from_matrix(rot_mat).as_quat()
    q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
    return pos, q_wxyz


def get_analog(buttons: dict, keys, default=0.0) -> float:
    """Extract analog value from button dictionary."""
    for k in keys:
        if k not in buttons:
            continue
        v = buttons[k]
        if isinstance(v, (list, tuple)) and len(v) > 0:
            try:
                return float(v[0])
            except Exception:
                continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, bool):
            return 1.0 if v else 0.0
    return float(default)


def controller_to_internal(pos_xyz: np.ndarray, q_wxyz: np.ndarray):
    """
    Convert controller coordinates to internal robot frame.

    Applies fixed coordinate transformations as defined in vr_controller.py.
    pos : xyz, quat: xyzw
    """
    A = np.array(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
    )
    B = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    M = B @ A

    R_c = R.from_quat(quat_wxyz_to_xyzw(q_wxyz)).as_matrix()
    pos_i = M @ pos_xyz
    R_i = M @ R_c @ M.T
    q_i = R.from_matrix(R_i).as_quat()
    return pos_i, q_i


def quat_rel_wxyz(q_cur_wxyz: np.ndarray, q_prev_wxyz: np.ndarray) -> np.ndarray:
    """Compute relative quaternion between current and previous orientations."""
    R_cur = R.from_quat(quat_wxyz_to_xyzw(q_cur_wxyz))
    R_prev = R.from_quat(quat_wxyz_to_xyzw(q_prev_wxyz))
    R_rel = R_cur * R_prev.inv()
    return quat_xyzw_to_wxyz(R_rel.as_quat())


def apply_delta_pose(
    current_pos: np.ndarray,
    current_quat_xyzw: np.ndarray,
    delta_pos: np.ndarray,
    delta_quat_xyzw: np.ndarray,
) -> tuple:
    """
    Apply delta pose to current pose.

    Args:
        current_pos: Current position [x, y, z]
        current_quat_xyzw: Current orientation quaternion [x, y, z, w]
        delta_pos: Delta position [dx, dy, dz]
        delta_quat_xyzw: Delta orientation quaternion [w, dx, dy, dz]

    Returns:
        Tuple of (new_pos, new_quat_xyzw)
    """
    # Apply position delta
    new_pos = current_pos + delta_pos

    # Apply rotation delta
    R_current = R.from_quat(current_quat_xyzw)
    R_delta = R.from_quat(delta_quat_xyzw)
    R_new = R_delta * R_current
    new_quat_xyzw = R_new.as_quat()

    return new_pos, new_quat_xyzw


def compute_delta_pose(
    side: str, target_vr_data: dict, cur_pos: dict, cur_quat: np.ndarray
) -> tuple:
    """
    Compute delta pose for one side.

    Args:
        side: 'left' or 'right'
        vr_data: VR controller data dictionary
        prev_pos: Previous position (or None)
        prev_quat: Previous quaternion (or None)

    Returns:
        Tuple of (delta_pos, delta_quat)
    """
    target_side_data = target_vr_data[side]

    target_pos = target_side_data["pos"]
    target_quat = target_side_data["quat"]

    delta_pos = target_pos - cur_pos
    delta_rot = R.from_quat(target_quat) * R.from_quat(cur_quat).inv()

    return delta_pos * POSITION_SCALE, delta_rot.as_quat()


# ------------------------- VR Interface Class -------------------------


class VRInterface:
    """Tracks VR controller state and provides access to VR data."""

    def __init__(self):
        """Initialize VR interface."""
        print("Initializing Oculus Reader...")
        self.device = OculusReader()

        # State tracking for delta computation
        self.r_prev_pos = None
        self.r_prev_quat = None
        self.l_prev_pos = None
        self.l_prev_quat = None

        # Trigger engagement state (hysteresis)
        self.r_engaged = False
        self.l_engaged = False

        self.r_up_edge = False
        self.r_down_edge = False
        self.l_up_edge = False
        self.l_down_edge = False

        # Gripper state
        self.r_gripper_closed = False
        self.l_gripper_closed = False
        self.r_gripper_value = GRIPPER_OPEN_VALUE
        self.l_gripper_value = GRIPPER_OPEN_VALUE

        print("VR Interface initialized!")

    def update_engagement(self, trigger_value: float, arm: str):
        """
        Update engagement with hysteresis and edge detection.
        Returns (rising_edge, falling_edge, engaged_state).
        """
        if arm == "right":
            engaged = self.r_engaged
        else:
            engaged = self.l_engaged

        rising = False
        falling = False

        # clear edge flags each call (one-shot semantics)
        if arm == "right":
            self.r_up_edge = False
            self.r_down_edge = False
        else:
            self.l_up_edge = False
            self.l_down_edge = False

        if not engaged and trigger_value >= TRIGGER_ON_THRESHOLD:
            engaged = True
            rising = True
        elif engaged and trigger_value <= TRIGGER_OFF_THRESHOLD:
            engaged = False
            falling = True

        # write back state + edges
        if arm == "right":
            self.r_engaged = engaged
            self.r_up_edge = rising
            self.r_down_edge = falling
        else:
            self.l_engaged = engaged
            self.l_up_edge = rising
            self.l_down_edge = falling

    def read_vr_controller(self, se3=False):
        """Read VR controller state and return parsed data."""
        sample = self.device.get_transformations_and_buttons()
        if not sample:
            return None

        transforms, buttons = sample
        if not transforms:
            return None

        # Extract button/trigger values
        trig_l = get_analog(buttons, ["leftTrig", "LT", "trigger_l"], 0.0)
        trig_r = get_analog(buttons, ["rightTrig", "RT", "trigger_r"], 0.0)
        idx_l = get_analog(buttons, ["leftGrip", "LG", "grip_l"], trig_l)
        idx_r = get_analog(buttons, ["rightGrip", "RG", "grip_r"], trig_r)

        # Get buttons - use direct access with fallback for booleans
        btn_a = bool(buttons["A"]) if "A" in buttons else False
        btn_b = bool(buttons["B"]) if "B" in buttons else False
        btn_x = bool(buttons["X"]) if "X" in buttons else False
        btn_y = bool(buttons["Y"]) if "Y" in buttons else False

        if "l" not in transforms or "r" not in transforms:
            return None
        Tl = transforms["l"]
        Tr = transforms["r"]
        if Tl is None or Tr is None:
            return None

        # Convert to internal coordinates
        l_pos_raw, l_quat_raw = pose_from_T(np.asarray(Tl))
        r_pos_raw, r_quat_raw = pose_from_T(np.asarray(Tr))
        l_pos_cur, l_quat_cur = controller_to_internal(l_pos_raw, l_quat_raw)
        r_pos_cur, r_quat_cur = controller_to_internal(r_pos_raw, r_quat_raw)
        l_quat_cur = normalize_quat_xyzw(l_quat_cur)
        r_quat_cur = normalize_quat_xyzw(r_quat_cur)
        
        R_l_cur = R.from_quat(l_quat_cur).as_matrix()
        R_r_cur = R.from_quat(r_quat_cur).as_matrix()

        R_l_new = L_YPR_OFFSET @ R_l_cur
        R_r_new = R_YPR_OFFSET @ R_r_cur

        l_quat_cur = normalize_quat_xyzw(R.from_matrix(R_l_new).as_quat())
        r_quat_cur = normalize_quat_xyzw(R.from_matrix(R_r_new).as_quat())

        # Create SE(3) matrices with YPR offset applied
        Tl = xyzxyzw_to_se3(l_pos_cur, l_quat_cur)
        Tr = xyzxyzw_to_se3(r_pos_cur, r_quat_cur)

        if se3:
            # Return SE(3) transformation matrices
            return {
                "left": {
                    "T": Tl,
                    "trigger": trig_l,
                    "index": idx_l,
                },
                "right": {
                    "T": Tr,
                    "trigger": trig_r,
                    "index": idx_r,
                },
                "buttons": {"A": btn_a, "B": btn_b, "X": btn_x, "Y": btn_y},
            }
        else:
            # Return position and quaternion format
            return {
                "left": {
                    "pos": l_pos_cur,
                    "quat": l_quat_cur,
                    "trigger": trig_l,
                    "index": idx_l,
                },
                "right": {
                    "pos": r_pos_cur,
                    "quat": r_quat_cur,
                    "trigger": trig_r,
                    "index": idx_r,
                },
                "buttons": {"A": btn_a, "B": btn_b, "X": btn_x, "Y": btn_y},
            }


# ------------------------- Robot Snapshot Helper -------------------------


def take_robot_snapshot(robot_interface, arms_list):
    """
    Capture complete robot state atomically in a single call.
    
    This ensures all state values (joints, poses, images) are captured
    at the same instant, avoiding inconsistencies from async updates.
    
    Args:
        robot_interface: The robot interface (ARXInterface or YAMInterface)
        arms_list: List of arm names to capture state for
        
    Returns:
        dict with keys:
            - "joints": {arm: 7-element ndarray} for each arm
            - "pose_se3": {arm: 4x4 ndarray} for each arm
            - "images": {cam_name: image_array or None} for each camera
            - "depth": {cam_name: depth_array or None} for each camera
    """
    snapshot = {
        "joints": {},
        "pose_se3": {},
        "images": {},
        "depth": {},
    }
    
    # Capture joint positions and poses for all arms
    for arm in arms_list:
        joints = robot_interface.get_joints(arm)
        if joints is not None:
            snapshot["joints"][arm] = np.asarray(joints, dtype=np.float64)
        else:
            snapshot["joints"][arm] = np.zeros(7, dtype=np.float64)
        
        pose_se3 = robot_interface.get_pose(arm, se3=True)
        if pose_se3 is not None:
            snapshot["pose_se3"][arm] = np.asarray(pose_se3, dtype=np.float64)
        else:
            snapshot["pose_se3"][arm] = np.eye(4, dtype=np.float64)
    
    # Capture images/depth from all cameras
    for cam_name, recorder in robot_interface.recorders.items():
        img = recorder.get_image()
        snapshot["images"][cam_name] = img
        depth = recorder.get_depth() if hasattr(recorder, "get_depth") else None
        snapshot["depth"][cam_name] = depth
    
    return snapshot


def se3_to_xyz_ypr(se3: np.ndarray):
    """Convert SE(3) matrix to position (xyz) and euler angles (ypr)."""
    xyz = se3[:3, 3]
    rot_mat = se3[:3, :3]
    ypr = R.from_matrix(rot_mat).as_euler("ZYX", degrees=False)
    return xyz, ypr


# ------------------------- Demo Recording Helpers -------------------------


def depth_dataset_name_from_camera(cam_name: str) -> str:
    """Map RGB camera key to depth dataset key, matching pair_teleop naming."""
    if cam_name.endswith("_img_1"):
        return cam_name.replace("_img_", "_depth_", 1)
    if cam_name.endswith("_img"):
        return cam_name[:-4] + "_depth"
    return f"{cam_name}_depth"


def reset_data(demo_data: dict):
    demo_data["cmd_joint_actions"] = []
    demo_data["robot_joint_actions"] = []
    demo_data["cmd_eepose_actions"] = []
    demo_data["robot_eepose_obs"] = []
    demo_data["obs"] = []
    demo_data["depth_obs"] = []


def _process_camera_stream(
    cam_name: str,
    demo_data: dict,
    num_obs_steps: int,
    depth_obs_steps,
    save_resolution: dict = None,
):
    """Convert one camera stream (RGB + depth) into contiguous arrays."""
    target_w = save_resolution["width"] if save_resolution is not None else None
    target_h = save_resolution["height"] if save_resolution is not None else None
    default_w = target_w if target_w is not None else 640
    default_h = target_h if target_h is not None else 480

    images = np.zeros((num_obs_steps, default_h, default_w, 3), dtype=np.uint8)
    depth = np.zeros((num_obs_steps, default_h, default_w, 1), dtype=np.float32)
    saw_depth_frame = False

    for i in range(num_obs_steps):
        img = demo_data["obs"][i].get(cam_name, None)
        if img is not None:
            img_rgb = img[..., ::-1]  # BGR -> RGB
            if save_resolution is not None:
                img_rgb = cv2.resize(
                    img_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC
                )
            images[i] = img_rgb.astype(np.uint8, copy=False)

        depth_dict = depth_obs_steps[i] if i < len(depth_obs_steps) else {}
        depth_frame = depth_dict.get(cam_name, None) if isinstance(depth_dict, dict) else None
        if depth_frame is not None:
            saw_depth_frame = True
            depth_frame = np.asarray(depth_frame)
            if depth_frame.ndim == 3 and depth_frame.shape[-1] == 1:
                depth_frame = depth_frame[..., 0]
            if depth_frame.ndim != 2:
                raise ValueError(
                    f"Depth for camera {cam_name} has invalid shape {depth_frame.shape}"
                )
            if depth_frame.dtype == np.uint16:
                depth_frame = depth_frame.astype(np.float32) / 1000.0  # mm -> m
            elif depth_frame.dtype != np.float32:
                depth_frame = depth_frame.astype(np.float32)
            if save_resolution is not None:
                depth_frame = cv2.resize(
                    depth_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                )
            depth[i, ..., 0] = depth_frame.astype(np.float32, copy=False)

    return cam_name, images, saw_depth_frame, depth


def save_demo(demo_data: dict, demo_dir, episode_id: int, cam_names, robot_type: str = "arx", yam_gripper_type: str = "linear_4310", save_resolution: dict = None):
    """Save demo to HDF5 file.
    
    Args:
        save_resolution: Optional dict with 'width' and 'height' keys to resize images before saving.
                        Uses bicubic interpolation. If None, images are saved at original resolution.
    """
    data_dict = dict()
    filename = demo_dir / f"demo_{episode_id}.hdf5"

    num_obs_steps = len(demo_data["obs"])
    depth_obs_steps = demo_data.get("depth_obs", [])
    # Parallelize expensive image/depth preprocessing across cameras.
    max_workers = max(1, min(len(cam_names), (os.cpu_count() or 1)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _process_camera_stream,
                cam_name,
                demo_data,
                num_obs_steps,
                depth_obs_steps,
                save_resolution,
            )
            for cam_name in cam_names
        ]
        for fut in as_completed(futures):
            cam_name, images, saw_depth_frame, depth = fut.result()
            data_dict[f"/observations/images/{cam_name}"] = images
            if saw_depth_frame:
                depth_name = depth_dataset_name_from_camera(cam_name)
                data_dict[f"/observations/depth/{depth_name}"] = depth
    print(
        f"Saving demo with {len(demo_data['cmd_eepose_actions'])} steps to {filename}"
    )
    data_dict["/text"] = str(demo_data.get("text", ""))
    data_dict["/observations/joints"] = np.asarray(
        demo_data["robot_joint_actions"], dtype=np.float32
    )
    data_dict["/observations/joint_positions"] = np.asarray(
        demo_data["robot_joint_actions"], dtype=np.float32
    )
    
    # Process cmd_eepose_actions to create different rotation representations
    # Original format: xyz(3) + ypr(3) + gripper(1) per arm = 14 total
    cmd_eepose_ypr = np.asarray(demo_data["cmd_eepose_actions"], dtype=np.float32)
    
    # Batch convert YPR to rotation matrices for both arms
    left_rot_mats = R.from_euler("ZYX", cmd_eepose_ypr[:, 3:6]).as_matrix()
    right_rot_mats = R.from_euler("ZYX", cmd_eepose_ypr[:, 10:13]).as_matrix()
    
    # 6D rotation: first two columns of rotation matrix, flattened
    left_rot_6d = left_rot_mats[:, :, :2].reshape(-1, 6)
    right_rot_6d = right_rot_mats[:, :, :2].reshape(-1, 6)
    
    # Build eepose_6drot: xyz(3) + rot6d(6) + gripper(1) per arm = 20 total
    cmd_eepose_6drot = np.column_stack([
        cmd_eepose_ypr[:, 0:3], left_rot_6d, cmd_eepose_ypr[:, 6:7],
        cmd_eepose_ypr[:, 7:10], right_rot_6d, cmd_eepose_ypr[:, 13:14],
    ]).astype(np.float32, copy=False)
    
    # Full rotation matrices: (N, 2, 3, 3)
    cmd_rot_3x3 = np.stack([left_rot_mats, right_rot_mats], axis=1).astype(
        np.float32, copy=False
    )
    
    data_dict["/actions/eepose_ypr"] = cmd_eepose_ypr.astype(np.float32, copy=False)
    data_dict["/actions/eepose_6drot"] = cmd_eepose_6drot
    data_dict["/actions/rot_3x3"] = cmd_rot_3x3
    data_dict["/actions/joints"] = np.asarray(demo_data["cmd_joint_actions"], dtype=np.float32)
    data_dict["/action"] = np.asarray(demo_data["cmd_joint_actions"], dtype=np.float32)

    # Prefer online-captured robot EE pose to avoid expensive save-time FK.
    precomputed_robot_eepose = demo_data.get("robot_eepose_obs")
    if (
        isinstance(precomputed_robot_eepose, list)
        and len(precomputed_robot_eepose) == len(demo_data["robot_joint_actions"])
    ):
        data_dict["/observations/eepose"] = np.asarray(
            precomputed_robot_eepose, dtype=np.float32
        )
    else:
        # Backward-compatible fallback for old recordings that lack robot_eepose_obs.
        if robot_type == "arx":
            kinematics_solver = EvaMinkKinematicsSolver(
                model_path="/home/robot/robot_ws/egomimic/resources/model_x5.xml"
            )
        elif robot_type == "yam":
            import i2rt
            urdf_path = os.path.join(
                os.path.dirname(i2rt.__file__), "robot_models", "yam", "yam.urdf"
            )
            kinematics_solver = PyRoKiKinematics(urdf_path=urdf_path, site_name="grasp_site")
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")
        
        robot_ee_pose = []
        for i in range(len(demo_data["robot_joint_actions"])):
            robot_joint_action = demo_data["robot_joint_actions"][i]
            left_joints = robot_joint_action[:7]
            right_joints = robot_joint_action[7:]
            # check if left is not 0 array
            if not np.allclose(left_joints, 0):
                if robot_type == "arx":
                    left_ee_xyz, left_ee_rot = kinematics_solver.fk(left_joints)
                    left_ee_ypr = left_ee_rot.as_euler("ZYX", degrees=False)
                else:  # yam
                    T = kinematics_solver.fk(left_joints[:6])
                    left_ee_xyz = T[:3, 3]
                    left_ee_ypr = R.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=False)
            else:
                left_ee_xyz = np.zeros(3)
                left_ee_ypr = np.zeros(3)
            if not np.allclose(right_joints, 0):
                if robot_type == "arx":
                    right_ee_xyz, right_ee_rot = kinematics_solver.fk(right_joints)
                    right_ee_ypr = right_ee_rot.as_euler("ZYX", degrees=False)
                else:  # yam
                    T = kinematics_solver.fk(right_joints[:6])
                    right_ee_xyz = T[:3, 3]
                    right_ee_ypr = R.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=False)
            else:
                right_ee_xyz = np.zeros(3)
                right_ee_ypr = np.zeros(3)
            left_ee_pose = np.concatenate(
                [left_ee_xyz, left_ee_ypr, [robot_joint_action[6]]]
            )
            right_ee_pose = np.concatenate(
                [right_ee_xyz, right_ee_ypr, [robot_joint_action[13]]]
            )
            robot_ee_pose.append(np.concatenate([left_ee_pose, right_ee_pose]))
        data_dict["/observations/eepose"] = np.asarray(robot_ee_pose, dtype=np.float32)
    t0 = time.time()
    max_timesteps = len(demo_data["cmd_eepose_actions"])
    with h5py.File(str(filename), "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        obs = root.create_group("observations")
        image = obs.create_group("images")

        for name, array in data_dict.items():
            if isinstance(array, np.ndarray) and array.ndim >= 1 and array.shape[0] != max_timesteps:
                raise ValueError(
                    f"Dataset length mismatch for {name}: "
                    f"expected {max_timesteps}, got {array.shape[0]}"
                )

        t_chunk = max(1, min(max_timesteps, 32))
        for cam_name in cam_names:
            # Get image dimensions from actual data
            img_data = data_dict[f"/observations/images/{cam_name}"]
            if img_data is not None and len(img_data) > 0:
                img_height, img_width = img_data.shape[1], img_data.shape[2]
            else:
                # Fallback to default dimensions if no data
                img_height, img_width = 480, 640
            _ = image.create_dataset(
                cam_name,
                (max_timesteps, img_height, img_width, 3),
                dtype="uint8",
                chunks=(t_chunk, img_height, img_width, 3),
            )
        depth_dataset_names = sorted(
            key.split("/")[-1]
            for key in data_dict
            if key.startswith("/observations/depth/")
        )
        if depth_dataset_names:
            depth_group = obs.create_group("depth")
            for depth_name in depth_dataset_names:
                depth_data = data_dict[f"/observations/depth/{depth_name}"]
                depth_height, depth_width = depth_data.shape[1], depth_data.shape[2]
                _ = depth_group.create_dataset(
                    depth_name,
                    (max_timesteps, depth_height, depth_width, 1),
                    dtype="float32",
                    chunks=(t_chunk, depth_height, depth_width, 1),
                )

        _ = obs.create_dataset("joints", (max_timesteps, 14), dtype="float32")
        _ = obs.create_dataset("eepose", (max_timesteps, 14), dtype="float32")
        _ = obs.create_dataset("joint_positions", (max_timesteps, 14), dtype="float32")
        _ = root.create_group("actions")
        _ = root["actions"].create_dataset("eepose_ypr", (max_timesteps, 14), dtype="float32")
        _ = root["actions"].create_dataset("eepose_6drot", (max_timesteps, 20), dtype="float32")
        _ = root["actions"].create_dataset("rot_3x3", (max_timesteps, 2, 3, 3), dtype="float32")
        _ = root["actions"].create_dataset("joints", (max_timesteps, 14), dtype="float32")
        _ = root.create_dataset("action", (max_timesteps, 14), dtype="float32")
        _ = root.create_dataset("text", shape=(), dtype=h5py.string_dtype(encoding="utf-8"))

        for name, array in data_dict.items():
            root[name][...] = array

    print(f"Saving: {(time.time() - t0):.1f} secs → {filename}")
    print(f"Demo length: {max_timesteps} steps")
    return True


# ------------------------- Main Entry Point -------------------------


def collect_demo(
    arms_to_collect: str = "right",
    frequency: float = DEFAULT_FREQUENCY,
    demo_dir: str = DEMO_DIR,
    recording: bool = True,
    auto_episode_start: int = None,
    robot_type: str = "arx",
    yam_gripper_type: str = "linear_4310",
    yam_interfaces: dict = None,
    dry_run: bool = False,
    position_scale: float = POSITION_SCALE,
    rotation_scale: float = ROTATION_SCALE,
    max_delta_pos: float = None,
    max_delta_rot_deg: float = None,
):
    """
    Collect demonstrations using VR controller.

    Args:
        arms: Which arm(s) to control ("left", "right", or "both")
        frequency: Control loop frequency in Hz
        demo_dir: Directory to save demos
        robot_type: Robot type to use ("arx" for ARX X5, "yam" for I2RT YAM)
        yam_gripper_type: Gripper type for YAM robot (only used if robot_type="yam")
        yam_interfaces: CAN interface mapping for YAM robot {"left": "can0", "right": "can1"}
        dry_run: If True, don't actuate robot - just log VR commands
        position_scale: Scale factor for VR position deltas
        rotation_scale: Scale factor for VR rotation deltas
        max_delta_pos: Max position change per tick in meters (None = load from config, 0 = disabled)
        max_delta_rot_deg: Max rotation change per tick in degrees (None = load from config, 0 = disabled)
    """
    # Load velocity limits from config if not specified
    if max_delta_pos is None or max_delta_rot_deg is None:
        cfg_pos, cfg_rot_rad = load_velocity_limits_from_config()
        max_delta_pos = cfg_pos if max_delta_pos is None else max_delta_pos
        max_delta_rot_rad = cfg_rot_rad if max_delta_rot_deg is None else np.deg2rad(max_delta_rot_deg)
    else:
        max_delta_rot_rad = np.deg2rad(max_delta_rot_deg)
    
    if max_delta_pos > 0 or max_delta_rot_rad > 0:
        print(f"[VelocityLimit] pos={max_delta_pos:.4f}m/tick, rot={np.rad2deg(max_delta_rot_rad):.2f}deg/tick")
    
    # Setup demo directory
    demo_dir = Path(demo_dir)
    demo_dir.mkdir(exist_ok=True, parents=True)

    # Initialize VR interface
    vr = VRInterface()
    prev_vr_data = None
    
    # Track previous commanded SE3 for velocity limiting
    prev_cmd_T = {}

    # Initialize robot interfaces (one per arm)
    if arms_to_collect == "both":
        arms_list = ["right", "left"]
    elif arms_to_collect == "right":
        arms_list = ["right"]
    elif arms_to_collect == "left":
        arms_list = ["left"]
    else:
        raise ValueError("Invalid arm values inputted.")
    
    # Select robot interface based on type
    gripper_type_enum = None
    if robot_type == "arx":
        if dry_run:
            print("ARX robot does not support dry run mode - use YAM robot for dry run testing")
            print("Proceeding anyway (will fail if ARX hardware not connected)")
        print("Using ARX robot interface")
        robot_interface = ARXInterface(arms=arms_list)
        # ARX uses raw gripper values
        gripper_open = GRIPPER_OPEN_VALUE
        gripper_close = GRIPPER_CLOSE_VALUE
        gripper_width = GRIPPER_WIDTH
    elif robot_type == "yam":
        print("Using YAM robot interface")
        gripper_type_enum = GripperType.from_string_name(yam_gripper_type)
        robot_interface = YAMInterface(
            arms=arms_list,
            gripper_type=gripper_type_enum,
            interfaces=yam_interfaces,
            zero_gravity_mode=False,  # Hold position on startup for safety
            dry_run=dry_run,
        )
        # Print detailed config for YAM
        robot_interface.print_config()
        # YAM uses normalized gripper values (0-1)
        gripper_open = YAM_GRIPPER_OPEN_VALUE
        gripper_close = YAM_GRIPPER_CLOSE_VALUE
        gripper_width = YAM_GRIPPER_WIDTH
    else:
        raise ValueError(f"Unknown robot type: {robot_type}. Use 'arx' or 'yam'.")

    # Demo recording state
    demo_data = dict()
    camera_names = list(robot_interface.recorders.keys())
    
    # Per-arm command state (populated during teleop)
    cmd_pos = dict()
    cmd_quat = dict()
    cmd_joints = dict()
    gripper_pos = dict()
    
    # Frame tracking for delta computation
    vr_frame_zero_se3 = dict()
    robot_frame_zero_se3 = dict()
    
    # Recording state
    collecting_data = False
    last_cmd_joint_action = None
    last_robot_joint_action = None
    last_cmd_eepose_action = None
    # Safety gating around homing/start transitions to avoid CAN instability.
    teleop_pause_until = 0.0
    require_trigger_release = False
    last_home_time = 0.0
    min_home_reissue_sec = 4.0 if robot_type == "yam" else 1.0
    post_home_pause_sec = 1.0 if robot_type == "yam" else 0.2
    # YAM can drop CAN links on repeated home commands; keep home explicit on Y only.
    home_on_b_start = robot_type != "yam"
    max_recovery_attempts = 3 if robot_type == "yam" else 0
    loop_lag_recover_strikes = 3 if robot_type == "yam" else 0
    loop_lag_threshold_s = max(0.25, 4.0 / frequency)
    loop_lag_strikes = 0
    last_tick_time = None
    runtime_state = "INIT"
    recovery_count = 0
    fault_count = 0
    last_status_ts = 0.0

    def seed_action_buffers_from_snapshot(snapshot):
        """Seed action/proprio buffers from snapshot."""
        cmd_joint_action = np.zeros(14, dtype=np.float64)
        robot_joint_action = np.zeros(14, dtype=np.float64)
        cmd_eepose_action = np.zeros(14, dtype=np.float64)
        
        for arm in arms_list:
            arm_offset = 7 if arm == "right" else 0
            joints = snapshot["joints"][arm]
            if joints is None or joints.shape[0] < 7:
                continue
            robot_joint_action[arm_offset:arm_offset + 7] = joints[:7]
            cmd_joint_action[arm_offset:arm_offset + 7] = joints[:7]
            
            rb_se3 = snapshot["pose_se3"][arm]
            if rb_se3 is None:
                continue
            pos, quat_xyzw = se3_to_xyzxyzw(rb_se3)
            ypr = R.from_quat(quat_xyzw).as_euler("ZYX", degrees=False)
            cmd_eepose_action[arm_offset:arm_offset + 3] = pos
            cmd_eepose_action[arm_offset + 3:arm_offset + 6] = ypr
            cmd_eepose_action[arm_offset + 6] = (joints[6] - gripper_close) / gripper_width
            
        return cmd_joint_action, robot_joint_action, cmd_eepose_action

    def triggers_fully_released(vr_sample: dict) -> bool:
        """True when both controller grips/triggers are released."""
        return (
            vr_sample["left"]["index"] <= TRIGGER_OFF_THRESHOLD
            and vr_sample["right"]["index"] <= TRIGGER_OFF_THRESHOLD
            and vr_sample["left"]["trigger"] <= TRIGGER_OFF_THRESHOLD
            and vr_sample["right"]["trigger"] <= TRIGGER_OFF_THRESHOLD
        )

    def safe_home(reason: str):
        """Home robot and arm teleop pause/release gates."""
        nonlocal teleop_pause_until, require_trigger_release, last_home_time, runtime_state
        runtime_state = "HOMING"
        print(f"[STATE] HOMING ({reason})")
        now = time.time()
        if (now - last_home_time) < min_home_reissue_sec:
            print(
                f"[{reason}] Skipping duplicate set_home "
                f"({now - last_home_time:.2f}s since last home)"
            )
        else:
            print(f"[{reason}] Moving robot to home...")
            try:
                robot_interface.set_home()
                last_home_time = time.time()
            except Exception as e:
                print(f"[{reason}] [WARN] set_home failed: {e}")
        teleop_pause_until = max(teleop_pause_until, time.time() + post_home_pause_sec)
        require_trigger_release = True
        prev_cmd_T.clear()
        vr_frame_zero_se3.clear()
        robot_frame_zero_se3.clear()
        runtime_state = "PAUSED"

    def is_motor_comm_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        signatures = (
            "loss communication",
            "motor chain is not running",
            "motor error detected",
            "fail to communicate with the motor",
            "socketcan",
            "dmchaincaninterface",
            "yam_real",
        )
        return any(sig in msg for sig in signatures)

    def wait_for_cameras_ready():
        print("Waiting for incoming images ----------------")
        with RateLoop(frequency=frequency, verbose=False) as loop:
            for _ in loop:
                snapshot = take_robot_snapshot(robot_interface, arms_list)
                all_cam_images_in = all(
                    snapshot["images"][cam_name] is not None for cam_name in camera_names
                )
                all_cam_depth_in = all(
                    (not hasattr(robot_interface.recorders[cam_name], "get_depth"))
                    or (snapshot["depth"][cam_name] is not None)
                    for cam_name in camera_names
                )
                if all_cam_images_in and all_cam_depth_in:
                    break
        print("All cameras are ready (rgb + depth) --------------")

    def recover_yam_interface(reason: str, err: Exception) -> bool:
        nonlocal robot_interface, camera_names
        nonlocal collecting_data, last_cmd_joint_action, last_robot_joint_action, last_cmd_eepose_action
        nonlocal teleop_pause_until, require_trigger_release, last_home_time
        nonlocal runtime_state, recovery_count, fault_count

        if robot_type != "yam":
            return False

        fault_count += 1
        runtime_state = "RECOVERING"
        print(f"[STATE] RECOVERING (trigger={reason})")
        print(f"[RECOVERY] Triggered by {reason}: {err}")

        if collecting_data:
            num_steps = len(demo_data.get("obs", []))
            print(f"[RECOVERY] Aborting current recording with {num_steps} steps")
        collecting_data = False
        reset_data(demo_data)
        last_cmd_joint_action = None
        last_robot_joint_action = None
        last_cmd_eepose_action = None
        prev_cmd_T.clear()
        vr_frame_zero_se3.clear()
        robot_frame_zero_se3.clear()

        # Ensure we do not send teleop commands immediately after reconnect.
        require_trigger_release = True
        teleop_pause_until = max(teleop_pause_until, time.time() + 2.0)

        for attempt in range(1, max_recovery_attempts + 1):
            print(f"[RECOVERY] Reinitializing YAM interface ({attempt}/{max_recovery_attempts})...")
            try:
                try:
                    robot_interface.close()
                except Exception as close_err:
                    print(f"[RECOVERY] Warning during close: {close_err}")
                time.sleep(1.0)

                robot_interface = YAMInterface(
                    arms=arms_list,
                    gripper_type=gripper_type_enum,
                    interfaces=yam_interfaces,
                    zero_gravity_mode=False,
                    dry_run=dry_run,
                )
                robot_interface.print_config()
                camera_names = list(robot_interface.recorders.keys())
                wait_for_cameras_ready()
                last_home_time = 0.0
                recovery_count += 1
                runtime_state = "IDLE"
                print(f"[STATE] IDLE (recoveries={recovery_count})")
                print("[RECOVERY] YAM interface recovered. Ready for next demo.")
                return True
            except Exception as reinit_err:
                print(f"[RECOVERY] Attempt {attempt} failed: {reinit_err}")
                time.sleep(2.0)

        print("[RECOVERY] Failed to recover YAM interface automatically.")
        runtime_state = "FAULT"
        print("[STATE] FAULT")
        return False

    wait_for_cameras_ready()
    runtime_state = "IDLE"
    print("[STATE] IDLE")
    
    auto_episode_id = auto_episode_start
    
    while True:
        if auto_episode_id is None:
            episode_id = input("Input the episode id: ")
        else:
            episode_id = auto_episode_id
            print(f"Set episode id to {episode_id} teleop enabled")
        
        with RateLoop(frequency=frequency, verbose=False) as loop:
            for i in loop:
                tick_now = time.time()
                if last_tick_time is not None:
                    dt_tick = tick_now - last_tick_time
                    if loop_lag_recover_strikes > 0:
                        if dt_tick > loop_lag_threshold_s:
                            loop_lag_strikes += 1
                        else:
                            loop_lag_strikes = max(0, loop_lag_strikes - 1)
                else:
                    dt_tick = 0.0
                last_tick_time = tick_now

                # ============================================================
                # STEP 1: Take atomic snapshot of robot state
                # ============================================================
                try:
                    snapshot = take_robot_snapshot(robot_interface, arms_list)
                except Exception as e:
                    if is_motor_comm_error(e):
                        if recover_yam_interface("snapshot read", e):
                            prev_vr_data = None
                            loop_lag_strikes = 0
                            continue
                    raise
                
                # ============================================================
                # STEP 2: Read VR controller
                # ============================================================
                vr_data = vr.read_vr_controller(se3=True)
                if vr_data is None:
                    vr_data = prev_vr_data
                    if vr_data is None:
                        continue

                # ============================================================
                # STEP 3: Handle button events
                # ============================================================
                buttons = vr_data["buttons"]
                
                # B button: Start/stop recording
                if buttons["B"]:
                    if prev_vr_data is not None and prev_vr_data["buttons"]["B"] == False:
                        if collecting_data:
                            # Stop recording and save
                            collecting_data = False
                            last_cmd_joint_action = None
                            last_robot_joint_action = None
                            last_cmd_eepose_action = None
                            save_resolution = getattr(robot_interface, 'save_resolution', None)
                            try:
                                runtime_state = "SAVING"
                                print("[STATE] SAVING")
                                save_demo(demo_data, demo_dir, episode_id, camera_names, 
                                         robot_type=robot_type, yam_gripper_type=yam_gripper_type, 
                                         save_resolution=save_resolution)
                            except Exception as e:
                                import traceback
                                print(f"[ERROR] save_demo failed: {e}")
                                traceback.print_exc()
                            runtime_state = "IDLE"
                            print("[STATE] IDLE")
                            if auto_episode_id is not None:
                                auto_episode_id += 1
                            break
                        else:
                            # Start recording
                            if home_on_b_start:
                                safe_home("B")
                            else:
                                print("[B] Start recording without homing (Y to home explicitly)")
                                teleop_pause_until = max(
                                    teleop_pause_until, time.time() + post_home_pause_sec
                                )
                                require_trigger_release = True
                                prev_cmd_T.clear()
                                vr_frame_zero_se3.clear()
                                robot_frame_zero_se3.clear()
                            print("Start Collecting Data ------------------------------")
                            collecting_data = True
                            runtime_state = "RECORDING"
                            print("[STATE] RECORDING")
                            reset_data(demo_data)
                            last_cmd_joint_action, last_robot_joint_action, last_cmd_eepose_action = \
                                seed_action_buffers_from_snapshot(snapshot)

                # X button: Delete current data
                if buttons["X"] and prev_vr_data is not None and prev_vr_data["buttons"]["X"] == False:
                    print("Deleting Data -----------------------------------")
                    reset_data(demo_data)
                    if collecting_data:
                        last_cmd_joint_action, last_robot_joint_action, last_cmd_eepose_action = \
                            seed_action_buffers_from_snapshot(snapshot)

                # A button: Kill/exit
                if buttons["A"]:
                    break

                # Y button: Reset to home (rising-edge only)
                if buttons["Y"] and prev_vr_data is not None and prev_vr_data["buttons"]["Y"] == False:
                    if collecting_data:
                        num_steps = len(demo_data.get("obs", []))
                        print(f"[Y] DISCARDING {num_steps} recorded steps and resetting to home")
                    else:
                        print("[Y] Resetting to home")
                    collecting_data = False
                    runtime_state = "PAUSED"
                    print("[STATE] PAUSED")
                    reset_data(demo_data)
                    safe_home("Y")
                    last_cmd_joint_action = None
                    last_robot_joint_action = None
                    last_cmd_eepose_action = None

                # ============================================================
                # STEP 4: Update engagement states
                # ============================================================
                vr.update_engagement(vr_data["right"]["index"], "right")
                vr.update_engagement(vr_data["left"]["index"], "left")

                if require_trigger_release and triggers_fully_released(vr_data):
                    require_trigger_release = False
                    print("[SAFETY] Triggers released; teleop re-armed.")
                teleop_paused = (time.time() < teleop_pause_until) or require_trigger_release

                if (
                    not collecting_data
                    and runtime_state not in ("RECOVERING", "SAVING", "FAULT")
                ):
                    runtime_state = "PAUSED" if teleop_paused else "IDLE"

                if (
                    loop_lag_recover_strikes > 0
                    and loop_lag_strikes >= loop_lag_recover_strikes
                    and runtime_state not in ("RECOVERING", "SAVING")
                ):
                    lag_err = RuntimeError(
                        f"loop stall watchdog dt={dt_tick:.3f}s strikes={loop_lag_strikes}"
                    )
                    if recover_yam_interface("loop stall watchdog", lag_err):
                        prev_vr_data = None
                        loop_lag_strikes = 0
                        continue

                if (tick_now - last_status_ts) >= 1.0:
                    steps = len(demo_data.get("obs", []))
                    print(
                        f"[STATUS] state={runtime_state} episode={episode_id} "
                        f"collecting={collecting_data} paused={teleop_paused} "
                        f"steps={steps} lag_strikes={loop_lag_strikes} "
                        f"recoveries={recovery_count} faults={fault_count}"
                    )
                    last_status_ts = tick_now
                
                # Clear velocity limit tracking when grip released
                if vr.r_down_edge and "right" in prev_cmd_T:
                    del prev_cmd_T["right"]
                if vr.l_down_edge and "left" in prev_cmd_T:
                    del prev_cmd_T["left"]

                # ============================================================
                # STEP 5: Seed action buffers if needed
                # ============================================================
                if collecting_data and last_cmd_joint_action is None:
                    last_cmd_joint_action, last_robot_joint_action, last_cmd_eepose_action = \
                        seed_action_buffers_from_snapshot(snapshot)

                # Initialize action arrays from last values or zeros
                if last_cmd_joint_action is None:
                    cmd_joint_action = np.zeros(14, dtype=np.float64)
                    robot_joint_action = np.zeros(14, dtype=np.float64)
                    cmd_eepose_action = np.zeros(14, dtype=np.float64)
                else:
                    cmd_joint_action = last_cmd_joint_action.copy()
                    robot_joint_action = last_robot_joint_action.copy()
                    cmd_eepose_action = last_cmd_eepose_action.copy()
                robot_eepose_action = np.zeros(14, dtype=np.float64)

                # ============================================================
                # STEP 6: Process each arm - compute commands and control
                # ============================================================
                recovery_triggered = False
                for arm in arms_list:
                    arm_offset = 7 if arm == "right" else 0
                    is_engaged = (
                        ((arm == "left" and vr.l_engaged) or (arm == "right" and vr.r_engaged))
                        and (not teleop_paused)
                    )
                    is_up_edge = (arm == "right" and vr.r_up_edge) or (arm == "left" and vr.l_up_edge)
                    
                    # Get robot state from snapshot (consistent values)
                    rb_se3 = snapshot["pose_se3"][arm]
                    arm_joints = snapshot["joints"][arm]
                    
                    # ALWAYS populate robot_joint_action from snapshot
                    robot_joint_action[arm_offset:arm_offset + 7] = arm_joints
                    # ALWAYS populate actual robot ee pose observation from snapshot
                    rb_xyz, rb_ypr = se3_to_xyz_ypr(rb_se3)
                    robot_eepose_action[arm_offset:arm_offset + 3] = rb_xyz
                    robot_eepose_action[arm_offset + 3:arm_offset + 6] = rb_ypr
                    robot_eepose_action[arm_offset + 6] = arm_joints[6]
                    
                    # DEBUG: print engagement state once per second
                    if i % frequency == 0 and arm == arms_list[0]:
                        print(f"[DBG] tick={i} engaged={is_engaged} arm={arm} joints={arm_joints[:3]}")
                    
                    if is_engaged:
                        # On engagement rising edge, store reference frames
                        if is_up_edge:
                            vr_frame_zero_se3[arm] = np.asarray(vr_data[arm]["T"], dtype=np.float64)
                            robot_frame_zero_se3[arm] = np.asarray(rb_se3, dtype=np.float64)

                        # Compute target pose if we have reference frames
                        if (prev_vr_data is not None and 
                            arm in vr_frame_zero_se3 and 
                            "T" in vr_data[arm]):
                            
                            # Compute relative transformation
                            vr_zero_inv = np.linalg.inv(vr_frame_zero_se3[arm])
                            vr_current_T = np.asarray(vr_data[arm]["T"], dtype=np.float64)
                            delta_T = vr_zero_inv @ vr_current_T
                            
                            # Apply headset flip correction if needed
                            if HEADSET_FLIPPED:
                                delta_T[0, 3] = -delta_T[0, 3]  # flip X
                                delta_T[2, 3] = -delta_T[2, 3]  # flip Z
                                R_180_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
                                delta_T[:3, :3] = R_180_y @ delta_T[:3, :3] @ R_180_y.T
                            
                            # Apply position and rotation scaling
                            delta_T[:3, 3] *= position_scale
                            
                            if rotation_scale != 1.0:
                                delta_rot = R.from_matrix(delta_T[:3, :3])
                                rotvec = delta_rot.as_rotvec()
                                scaled_rotvec = rotvec * rotation_scale
                                delta_T[:3, :3] = R.from_rotvec(scaled_rotvec).as_matrix()
                            
                            cmd_T_raw = robot_frame_zero_se3[arm] @ delta_T
                        else:
                            cmd_T_raw = rb_se3

                        # Apply velocity limiting
                        if arm in prev_cmd_T and (max_delta_pos > 0 or max_delta_rot_rad > 0):
                            prev_T = prev_cmd_T[arm]
                            clamped_pos = clamp_delta_pos(cmd_T_raw[:3, 3] - prev_T[:3, 3], max_delta_pos)
                            R_prev = R.from_matrix(prev_T[:3, :3])
                            R_delta = R.from_matrix(cmd_T_raw[:3, :3]) * R_prev.inv()
                            clamped_rotvec = clamp_delta_rot(R_delta.as_rotvec(), max_delta_rot_rad)
                            cmd_T = np.eye(4, dtype=np.float64)
                            cmd_T[:3, 3] = prev_T[:3, 3] + clamped_pos
                            cmd_T[:3, :3] = (R.from_rotvec(clamped_rotvec) * R_prev).as_matrix()
                        else:
                            cmd_T = cmd_T_raw
                        prev_cmd_T[arm] = cmd_T.copy()

                        # Compute gripper position
                        gripper_pos[arm] = gripper_open - vr_data[arm]["trigger"] * gripper_width

                        # Extract commanded pose
                        cmd_pos[arm], cmd_quat[arm] = se3_to_xyzxyzw(cmd_T)
                        cmd_ypr = R.from_quat(cmd_quat[arm]).as_euler("ZYX", degrees=False)
                        eepose_cmd = np.concatenate([cmd_pos[arm], cmd_ypr])
                        
                        # Solve IK and command robot
                        try:
                            solved_joints = robot_interface.solve_ik(eepose_cmd[:6], arm)
                        except Exception as e:
                            if is_motor_comm_error(e):
                                if recover_yam_interface(f"IK ({arm})", e):
                                    prev_vr_data = None
                                    loop_lag_strikes = 0
                                    recovery_triggered = True
                                    break
                            print(f"[WARN] IK failed for arm {arm}: {e}")
                            import traceback; traceback.print_exc()
                            continue
                            
                        if solved_joints is not None:
                            cmd_joints[arm] = np.concatenate([solved_joints, [gripper_pos[arm]]])
                            try:
                                robot_interface.set_joints(cmd_joints[arm], arm)
                            except Exception as e:
                                if is_motor_comm_error(e):
                                    if recover_yam_interface(f"set_joints ({arm})", e):
                                        prev_vr_data = None
                                        loop_lag_strikes = 0
                                        recovery_triggered = True
                                        break
                                print(f"[WARN] set_joints failed for arm {arm}: {e}")
                                continue
                            # DEBUG: print first few IK successes
                            if i < 5 * frequency:
                                print(f"[DBG IK] arm={arm} solved={solved_joints[:3]} cmd_pos={cmd_pos[arm]}")
                        else:
                            print(f"[DBG IK] arm={arm} solve_ik returned None")

                        # Populate commanded action arrays
                        cmd_eepose_action[arm_offset:arm_offset + 3] = cmd_pos[arm]
                        cmd_eepose_action[arm_offset + 3:arm_offset + 6] = cmd_ypr
                        cmd_eepose_action[arm_offset + 6] = (gripper_pos[arm] - gripper_close) / gripper_width
                        
                        if arm in cmd_joints:
                            cmd_joint_action[arm_offset:arm_offset + 7] = cmd_joints[arm]
                    
                    else:
                        # NOT ENGAGED: Use actual robot state as "commanded" (hold position)
                        # This ensures cmd_* arrays have valid data for both arms
                        xyz, ypr = se3_to_xyz_ypr(rb_se3)
                        gripper_val = arm_joints[6]  # Actual gripper position
                        
                        cmd_eepose_action[arm_offset:arm_offset + 3] = xyz
                        cmd_eepose_action[arm_offset + 3:arm_offset + 6] = ypr
                        cmd_eepose_action[arm_offset + 6] = (gripper_val - gripper_close) / gripper_width
                        cmd_joint_action[arm_offset:arm_offset + 7] = arm_joints

                if recovery_triggered:
                    continue

                # ============================================================
                # STEP 7: Append data ONCE per tick (after processing all arms)
                # ============================================================
                if collecting_data:
                    # Build observation from snapshot
                    obs_copy = {}
                    for cam_name, img in snapshot["images"].items():
                        obs_copy[cam_name] = img.copy() if img is not None else None
                    depth_obs_copy = {}
                    for cam_name, depth in snapshot["depth"].items():
                        depth_obs_copy[cam_name] = depth.copy() if depth is not None else None
                    
                    demo_data["obs"].append(obs_copy)
                    demo_data["depth_obs"].append(depth_obs_copy)
                    demo_data["cmd_joint_actions"].append(cmd_joint_action.copy())
                    demo_data["robot_joint_actions"].append(robot_joint_action.copy())
                    demo_data["cmd_eepose_actions"].append(cmd_eepose_action.copy())
                    demo_data["robot_eepose_obs"].append(robot_eepose_action.copy())
                    
                    # Update last values for next iteration
                    last_cmd_joint_action = cmd_joint_action.copy()
                    last_robot_joint_action = robot_joint_action.copy()
                    last_cmd_eepose_action = cmd_eepose_action.copy()

                # Store previous VR data
                if vr_data is not None:
                    prev_vr_data = vr_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect robot demonstrations using VR controller"
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="right",
        choices=["left", "right", "both"],
        help="Which arm(s) to control",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=DEFAULT_FREQUENCY,
        help="Control loop frequency in Hz",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default=DEMO_DIR,
        help="Directory to save demos",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run VR controller orientation calibration before teleop",
    )
    parser.add_argument(
        "--auto-episode-start",
        type=int,
        default=None,
        help="If set, start at this episode id and auto-increment on each recording",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="arx",
        choices=["arx", "yam"],
        help="Robot type to use: 'arx' for ARX X5, 'yam' for I2RT YAM",
    )
    parser.add_argument(
        "--yam-gripper-type",
        type=str,
        default="linear_4310",
        choices=["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"],
        help="Gripper type for YAM robot (only used if --robot-type=yam)",
    )
    parser.add_argument(
        "--yam-left-can",
        type=str,
        default="can0",
        help="CAN interface for YAM left arm (only used if --robot-type=yam)",
    )
    parser.add_argument(
        "--yam-right-can",
        type=str,
        default="can1",
        help="CAN interface for YAM right arm (only used if --robot-type=yam)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: simulate VR teleop without actuating the robot",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=POSITION_SCALE,
        help=f"Scale factor for VR position deltas (default: {POSITION_SCALE})",
    )
    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=ROTATION_SCALE,
        help=f"Scale factor for VR rotation deltas (default: {ROTATION_SCALE})",
    )
    parser.add_argument(
        "--headset-flipped",
        action="store_true",
        default=HEADSET_FLIPPED,
        help="Enable if headset is worn backwards (cameras facing user). Flips X/Z axes.",
    )
    parser.add_argument(
        "--no-headset-flipped",
        action="store_true",
        help="Disable headset flip correction (normal headset orientation)",
    )

    args = parser.parse_args()
    
    # Handle headset flip flag
    if args.no_headset_flipped:
        HEADSET_FLIPPED = False
    else:
        HEADSET_FLIPPED = args.headset_flipped
    
    # Print configuration summary
    print("\n" + "="*60)
    print("TELEOP CONFIGURATION")
    print("="*60)
    print(f"Robot type:       {args.robot_type}")
    print(f"Arms:             {args.arms}")
    print(f"Frequency:        {args.frequency} Hz")
    print(f"Demo directory:   {args.demo_dir}")
    print(f"Position scale:   {args.position_scale}")
    print(f"Rotation scale:   {args.rotation_scale}")
    print(f"Headset flipped:  {HEADSET_FLIPPED}")
    print(f"Dry run:          {args.dry_run}")
    if args.robot_type == "yam":
        print(f"YAM gripper type: {args.yam_gripper_type}")
        print(f"YAM left CAN:     {args.yam_left_can}")
        print(f"YAM right CAN:    {args.yam_right_can}")
    print("="*60 + "\n")

    if args.calibrate:
        # Import here to avoid dependency if user never calibrates
        from egomimic.robot.calibrate_utils import (
            calibrate_right_controller,
            calibrate_left_controller,
        )

        print("Running VR controller calibration...")
        # Override globals based on which arms are used
        if args.arms in ("right", "both"):
            print("\nCalibrating RIGHT controller...")
            R_off_right = calibrate_right_controller()
            # overwrite module-level constant
            R_YPR_OFFSET = R_off_right

        if args.arms in ("left", "both"):
            print("\nCalibrating LEFT controller...")
            R_off_left = calibrate_left_controller()
            # overwrite module-level constant
            L_YPR_OFFSET = R_off_left

        print("Calibration finished. Using updated offsets for this run.\n")

    # Build YAM interface mapping if using YAM robot
    yam_interfaces = None
    if args.robot_type == "yam":
        yam_interfaces = {
            "left": args.yam_left_can,
            "right": args.yam_right_can,
        }
        print(f"YAM CAN interfaces: {yam_interfaces}")
        print(f"YAM gripper type: {args.yam_gripper_type}")

    collect_demo(
        arms_to_collect=args.arms,
        frequency=args.frequency,
        demo_dir=args.demo_dir,
        auto_episode_start=args.auto_episode_start,
        robot_type=args.robot_type,
        yam_gripper_type=args.yam_gripper_type,
        yam_interfaces=yam_interfaces,
        dry_run=args.dry_run,
        position_scale=args.position_scale,
        rotation_scale=args.rotation_scale,
    )
