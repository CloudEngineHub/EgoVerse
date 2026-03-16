#!/usr/bin/env python3
"""
DWM Streaming Client for YAM Robot.

Standalone client that runs on Jetson Orin to:
- Collect observations from YAM robot and cameras
- Stream observations to desktop inference server  
- Receive actions and execute on robot

Minimal dependencies - no egomimic required.

Usage:
    python dwm_yam_client.py \
        --server tcp://192.168.1.100:5555 \
        --arms both \
        --query-frequency 16 \
        --frequency 30
"""

import os
import sys
import time
import argparse
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R, Slerp

# Add i2rt and robot interface paths
sys.path.insert(0, os.path.expanduser("~/i2rt"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))

import zmq
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from robot_interface import YAMInterface
from i2rt.robots.utils import GripperType


# =============================================================================
# Action Space Utilities
# =============================================================================

def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation back to rotation matrix.
    
    The 6D representation is the first two columns of the rotation matrix, flattened
    in row-major (C) order. Given a (3, 2) slice of columns, reshape(-1, 6) produces:
        [r00, r01, r10, r11, r20, r21]
    where r_ij is element (row i, col j) of the original rotation matrix.
    
    We recover the third column via cross product and orthonormalize.
    
    Args:
        rot6d: (..., 6) array of 6D rotation representation
        
    Returns:
        rot_matrix: (..., 3, 3) rotation matrix
    """
    # Handle batched input
    original_shape = rot6d.shape[:-1]
    rot6d = rot6d.reshape(-1, 6)
    
    # The 6D representation was created by taking the first two columns of the
    # rotation matrix (shape 3x2) and flattening in row-major order:
    #   [[r00, r01], [r10, r11], [r20, r21]] -> [r00, r01, r10, r11, r20, r21]
    # So we reshape back to (N, 3, 2) to correctly extract the columns.
    rot6d_reshaped = rot6d.reshape(-1, 3, 2)
    col1 = rot6d_reshaped[:, :, 0]  # (N, 3) - first column [r00, r10, r20]
    col2 = rot6d_reshaped[:, :, 1]  # (N, 3) - second column [r01, r11, r21]
    
    # Normalize first column
    col1 = col1 / (np.linalg.norm(col1, axis=1, keepdims=True) + 1e-8)
    
    # Make second column orthogonal to first and normalize
    col2 = col2 - np.sum(col1 * col2, axis=1, keepdims=True) * col1
    col2 = col2 / (np.linalg.norm(col2, axis=1, keepdims=True) + 1e-8)
    
    # Third column is cross product
    col3 = np.cross(col1, col2)
    
    # Assemble rotation matrix
    rot_matrix = np.stack([col1, col2, col3], axis=-1)  # (N, 3, 3)
    
    # Reshape back to original batch shape
    rot_matrix = rot_matrix.reshape(*original_shape, 3, 3)
    
    return rot_matrix


def convert_6drot_to_ypr(actions_6drot: np.ndarray) -> np.ndarray:
    """Convert actions from 6D rotation format to YPR format.
    
    6D rotation format (20D total):
        [left_xyz(3), left_rot6d(6), left_grip(1), right_xyz(3), right_rot6d(6), right_grip(1)]
        
    YPR format (14D total):
        [left_xyz(3), left_ypr(3), left_grip(1), right_xyz(3), right_ypr(3), right_grip(1)]
    
    Args:
        actions_6drot: (T, 20) array in 6D rotation format
        
    Returns:
        actions_ypr: (T, 14) array in YPR format
    """
    T = actions_6drot.shape[0]
    
    # Extract components for left arm
    left_xyz = actions_6drot[:, 0:3]
    left_rot6d = actions_6drot[:, 3:9]
    left_grip = actions_6drot[:, 9:10]
    
    # Extract components for right arm
    right_xyz = actions_6drot[:, 10:13]
    right_rot6d = actions_6drot[:, 13:19]
    right_grip = actions_6drot[:, 19:20]
    
    # Convert rot6d to rotation matrices
    left_rot_mat = rot6d_to_matrix(left_rot6d)   # (T, 3, 3)
    right_rot_mat = rot6d_to_matrix(right_rot6d)  # (T, 3, 3)
    
    # Convert rotation matrices to YPR (ZYX Euler angles)
    left_ypr = R.from_matrix(left_rot_mat).as_euler('ZYX')   # (T, 3)
    right_ypr = R.from_matrix(right_rot_mat).as_euler('ZYX')  # (T, 3)
    
    # Assemble YPR format: [left_xyz, left_ypr, left_grip, right_xyz, right_ypr, right_grip]
    actions_ypr = np.hstack([
        left_xyz, left_ypr, left_grip,
        right_xyz, right_ypr, right_grip,
    ])
    
    return actions_ypr


# =============================================================================
# DWM Streaming Client
# =============================================================================

class DWMYAMClient:
    """Streaming client for YAM robot with DWM inference server."""
    
    def __init__(
        self,
        server_addr: str,
        arms: str,
        execution_horizon: int,
        timeout_ms: int,
        gripper_type: str,
        can_interfaces: dict,
        dry_run: bool,
        velocity_limit: float = None,
        wait_for_server: bool = False,
        zero_action_threshold: float = None,
        max_ik_iters: int = 500,
        ik_error_threshold: float = None,
        skip_small_delta: float = None,
        monitor_latency: bool = False,
        read_all_arms: bool = False,
        disable_gripper_calibration: bool = False,
        camera_fps_15: bool = False,
        action_lpf_alpha_xyz: float = None,
        action_lpf_alpha_ypr: float = None,
        action_lpf_alpha_grip: float = None,
        action_slew_max_xyz_step: float = None,
        action_slew_max_ypr_step: float = None,
        action_slew_max_grip_step: float = None,
        action_deadband_xyz: float = None,
        action_deadband_ypr: float = None,
        action_deadband_grip: float = None,
        capture_pause_delay: float = 0.0,
        motion_smooth_mode: str = "lpf",
        interp_xyz_keyframe_step: int = 1,
        interp_orientation_keyframe_step: int = 1,
        interp_gripper_keyframe_step: int = 1,
    ):
        """Initialize YAM client.
        
        Args:
            server_addr: ZMQ server address (e.g., "tcp://192.168.1.100:5555")
            arms: Which arm(s) to control ("left", "right", or "both")
            execution_horizon: Number of actions to execute from each action chunk before
                              requesting new actions. New actions are requested after executing
                              exactly this many actions.
            timeout_ms: Request timeout in milliseconds
            gripper_type: Gripper type string
            can_interfaces: CAN interface mapping {"left": "can0", "right": "can1"}
            dry_run: If True, connect to real robot for proprioception but do NOT execute actions.
                     Robot will be homed and observations will be real, but set_pose() is skipped.
            velocity_limit: Maximum joint velocity in rad/s. If None, no velocity limit is applied.
            wait_for_server: If True, wait indefinitely for the server to become available.
            zero_action_threshold: If set, replace zero-action arms whose action sequence
                                   first-to-last distance is below this threshold with the current pose.
                                   Workaround for models trained with zeros for non-engaged arms.
            max_ik_iters: Maximum IK solver iterations (lower = faster failure, default 500).
            ik_error_threshold: If set, skip arm commands when IK error exceeds this value (meters).
            skip_small_delta: If set, skip arm execution when target xyz is within this distance
                             of current pose (meters). Avoids wasting IK on "hold position" actions.
            monitor_latency: If True, track and report camera latency statistics.
            read_all_arms: If True, initialize both arms for proprioception even when controlling one.
            disable_gripper_calibration: If True, disable linear gripper auto-calibration at startup.
            camera_fps_15: If True, force all cameras to 15 FPS regardless of config.
            action_lpf_alpha_xyz: Optional EMA alpha for xyz smoothing (0, 1].
            action_lpf_alpha_ypr: Optional EMA alpha for yaw/pitch/roll smoothing (0, 1].
            action_lpf_alpha_grip: Optional EMA alpha for gripper smoothing (0, 1].
            action_slew_max_xyz_step: Optional max xyz change per control step (meters).
            action_slew_max_ypr_step: Optional max ypr change per control step (radians).
            action_slew_max_grip_step: Optional max gripper change per control step.
            action_deadband_xyz: Optional xyz deadband relative to previous command (meters).
            action_deadband_ypr: Optional ypr deadband relative to previous command (radians).
            action_deadband_grip: Optional gripper deadband relative to previous command.
            capture_pause_delay: Seconds to pause before capturing images when requesting new
                                actions. Reduces motion blur by allowing the robot to settle.
                                Default 0 (no pause).
            motion_smooth_mode: Action smoothing mode. "lpf" keeps current per-step filtering
                               behavior. "interp" smooths each received horizon with endpoint-
                               preserving interpolation before execution.
            interp_xyz_keyframe_step: Interpolation keyframe step for xyz. Integer >= 1.
                                      Use 1 for no interpolation smoothing (keeps every point).
            interp_orientation_keyframe_step: Interpolation keyframe step for orientation.
                                              Integer >= 1. Use 1 for no interpolation smoothing.
            interp_gripper_keyframe_step: Interpolation keyframe step for gripper.
                                          Integer >= 1. Use 1 for no interpolation smoothing.
        """
        self.server_addr = server_addr
        self.arms = arms
        self.execution_horizon = execution_horizon
        self.timeout_ms = timeout_ms
        self.dry_run = dry_run
        self.velocity_limit = velocity_limit
        self.wait_for_server = wait_for_server
        self.zero_action_threshold = zero_action_threshold
        self.skip_small_delta = skip_small_delta
        self.monitor_latency = monitor_latency
        self.read_all_arms = read_all_arms
        self.disable_gripper_calibration = disable_gripper_calibration
        self.camera_fps_15 = camera_fps_15
        self.action_lpf_alpha_xyz = action_lpf_alpha_xyz
        self.action_lpf_alpha_ypr = action_lpf_alpha_ypr
        self.action_lpf_alpha_grip = action_lpf_alpha_grip
        self.action_slew_max_xyz_step = action_slew_max_xyz_step
        self.action_slew_max_ypr_step = action_slew_max_ypr_step
        self.action_slew_max_grip_step = action_slew_max_grip_step
        self.action_deadband_xyz = action_deadband_xyz
        self.action_deadband_ypr = action_deadband_ypr
        self.action_deadband_grip = action_deadband_grip
        self.motion_smooth_mode = motion_smooth_mode
        self.interp_xyz_keyframe_step = interp_xyz_keyframe_step
        self.interp_orientation_keyframe_step = interp_orientation_keyframe_step
        self.interp_gripper_keyframe_step = interp_gripper_keyframe_step
        if capture_pause_delay < 0:
            raise ValueError(f"capture_pause_delay must be >= 0, got {capture_pause_delay}")
        self.capture_pause_delay = capture_pause_delay
        self._zero_arms = {}
        self._skipped_arms = {}  # Track which arms are being skipped due to small delta
        self._ik_fail_count = {"left": 0, "right": 0}  # Consecutive IK failures per arm
        self._ik_fail_skip_threshold = 3  # Skip arm after N consecutive IK failures
        self._inactive_arm_home_logged = False
        self._validate_motion_smoothing_args()
        self._validate_action_filter_args()
        self._use_action_filtering = any(
            v is not None for v in (
                self.action_lpf_alpha_xyz,
                self.action_lpf_alpha_ypr,
                self.action_lpf_alpha_grip,
                self.action_slew_max_xyz_step,
                self.action_slew_max_ypr_step,
                self.action_slew_max_grip_step,
                self.action_deadband_xyz,
                self.action_deadband_ypr,
                self.action_deadband_grip,
            )
        )
        self._last_filtered_arm_action = {}  # arm -> most recently filtered 7D command
        if self.motion_smooth_mode == "interp":
            print(
                f"[DWMClient] Motion smoothing mode: interp "
                f"(stride xyz/ypr/grip="
                f"{self.interp_xyz_keyframe_step}/"
                f"{self.interp_orientation_keyframe_step}/"
                f"{self.interp_gripper_keyframe_step}); "
                "per-step LPF disabled, slew/deadband remain active."
            )
        else:
            print("[DWMClient] Motion smoothing mode: lpf (per-step LPF + slew + deadband path).")
        
        # Latency tracking
        self._latency_history = {}  # camera_name -> list of frame_age_ms
        self._latency_window = 30  # Keep last N samples for averaging
        
        # Determine arm list
        if arms == "both":
            self.arms_list = ["left", "right"]
        else:
            self.arms_list = [arms]
        
        # Setup ZMQ connection FIRST (before robot to avoid motor timeout during connection)
        print(f"Connecting to inference server: {server_addr}")
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(server_addr)
        
        # Test connection (with retry if wait_for_server is enabled)
        self._connect_with_retry()
        
        # Initialize robot interface AFTER ZMQ is ready
        # Note: We always connect to real robot (dry_run=False in YAMInterface)
        # The client's dry_run flag only controls whether we execute actions
        print(f"Initializing YAM robot interface...")
        gripper_enum = GripperType.from_string_name(gripper_type)
        self.robot = YAMInterface(
            arms=self.arms_list,
            gripper_type=gripper_enum,
            interfaces=can_interfaces,
            zero_gravity_mode=False,
            dry_run=False,  # Always use real robot for proprioception
            max_ik_iters=max_ik_iters,
            ik_error_threshold=ik_error_threshold,
            read_all_arms=read_all_arms,
            disable_gripper_calibration=disable_gripper_calibration,
            camera_fps_override=15 if camera_fps_15 else None,
        )
        self._home_joint_proprio = np.asarray(self.robot.HOME_POSITION, dtype=np.float64).copy()
        self._home_ee_proprio = self._compute_home_ee_proprio()
        
        if dry_run:
            print("[DWMYAMClient] DRY RUN MODE: Real robot connected for proprioception, but actions will NOT be executed")

        time.sleep(2)
        
        # Action buffer and index tracking
        self.actions = None
        self.action_index = 0  # Index of next action to execute in current chunk

    def _compute_home_ee_proprio(self) -> np.ndarray:
        """Compute [xyz, ypr, gripper] for the robot home joint pose."""
        fallback = np.concatenate([
            np.zeros(6, dtype=np.float64),
            [self._home_joint_proprio[6]],
        ])

        try:
            # All YAM arms share the same kinematic model, so any initialized arm works.
            ref_arm = self.arms_list[0]
            home_T = self.robot.kinematics[ref_arm].fk(self._home_joint_proprio[:6])
            home_xyz = home_T[:3, 3]
            home_ypr = R.from_matrix(home_T[:3, :3]).as_euler("ZYX", degrees=False)
            return np.concatenate([home_xyz, home_ypr, [self._home_joint_proprio[6]]])
        except Exception as e:
            print(f"[DWMClient] Warning: failed to compute home ee pose ({e}); using zero xyz/ypr fallback.")
            return fallback

    def _set_inactive_arm_proprio_to_home(self, obs: dict) -> dict:
        """In single-arm mode, set the inactive arm proprioception to home pose."""
        if self.arms == "both":
            return obs

        inactive_arm = "right" if self.arms == "left" else "left"
        arm_offset = 0 if inactive_arm == "left" else 7

        updated_obs = obs.copy()
        changed = False

        joint_positions = updated_obs.get("joint_positions")
        if isinstance(joint_positions, np.ndarray) and joint_positions.shape[0] >= arm_offset + 7:
            joint_positions = joint_positions.copy()
            joint_positions[arm_offset:arm_offset + 7] = self._home_joint_proprio
            updated_obs["joint_positions"] = joint_positions
            changed = True

        ee_poses = updated_obs.get("ee_poses")
        if isinstance(ee_poses, np.ndarray) and ee_poses.shape[0] >= arm_offset + 7:
            ee_poses = ee_poses.copy()
            ee_poses[arm_offset:arm_offset + 7] = self._home_ee_proprio
            updated_obs["ee_poses"] = ee_poses
            changed = True

        if changed and not self._inactive_arm_home_logged:
            print(
                f"[DWMClient] Single-arm mode: setting inactive {inactive_arm} arm proprioception "
                f"to home pose in observations."
            )
            self._inactive_arm_home_logged = True

        return updated_obs if changed else obs
        
    def _connect_with_retry(self):
        """Connect to server, optionally waiting indefinitely if wait_for_server is enabled."""
        retry_interval = 2.0  # seconds between retries
        attempt = 0
        
        while True:
            attempt += 1
            try:
                self._ping()
                return  # Success
            except zmq.error.Again:
                # Timeout - server not responding
                if not self.wait_for_server:
                    raise RuntimeError(
                        f"Server not responding at {self.server_addr}. "
                        "Use --wait-for-server to wait indefinitely."
                    )
                print(f"Server not available (attempt {attempt}), retrying in {retry_interval}s...")
                # Reset socket for retry (ZMQ REQ socket can get stuck after timeout)
                self.sock.close()
                self.sock = self.ctx.socket(zmq.REQ)
                self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                self.sock.setsockopt(zmq.LINGER, 0)
                self.sock.connect(self.server_addr)
                time.sleep(retry_interval)
            except Exception as e:
                if not self.wait_for_server:
                    raise
                print(f"Connection error (attempt {attempt}): {e}, retrying in {retry_interval}s...")
                # Reset socket for retry
                self.sock.close()
                self.sock = self.ctx.socket(zmq.REQ)
                self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                self.sock.setsockopt(zmq.LINGER, 0)
                self.sock.connect(self.server_addr)
                time.sleep(retry_interval)
    
    def _ping(self):
        """Test server connection."""
        self.sock.send(msgpack.packb({'cmd': 'ping'}, use_bin_type=True))
        reply = msgpack.unpackb(self.sock.recv(), raw=False)
        if reply.get('status') != 'ok':
            raise RuntimeError(f"Server ping failed: {reply}")
        print("Server connection verified.")
    
    def _request_actions(self, obs: dict, step: int) -> np.ndarray:
        """Request inference from server.
        
        Args:
            obs: Observation dictionary with images and joint_positions
            step: Current timestep
            
        Returns:
            actions: Raw actions from server (shape depends on action_space)
                     - ypr: (T, 14) [left_xyz_ypr_grip, right_xyz_ypr_grip]
                     - 6drot: (T, 20) [left_xyz_rot6d_grip, right_xyz_rot6d_grip]
        """
        # Prepare observation data
        obs_data = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                obs_data[key] = val
            elif val is not None:
                obs_data[key] = np.asarray(val)
        
        msg = {'cmd': 'infer', 'obs': obs_data, 'step': step}
        
        t0 = time.time()
        self.sock.send(msgpack.packb(msg, use_bin_type=True))
        reply = msgpack.unpackb(self.sock.recv(), raw=False)
        rtt = time.time() - t0
        
        if 'error' in reply:
            raise RuntimeError(f"Server error: {reply['error']}")
        
        actions = reply['actions']
        infer_t = reply.get('infer_time', 0)
        print(f"Step {step}: roundtrip {rtt*1000:.1f}ms (inference {infer_t*1000:.1f}ms)")
        
        return actions
    
    def _process_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        """Process raw actions from server to YPR format for robot execution.
        
        Auto-detects action space from dimensionality:
            - 14D → YPR format [left_xyz_ypr_grip, right_xyz_ypr_grip]
            - 20D → 6drot format [left_xyz_rot6d_grip, right_xyz_rot6d_grip]
        
        Actions are assumed to already be in robot base frame (no coordinate 
        transformation needed).
        
        Slices to execution_horizon actions.
        
        Args:
            raw_actions: Raw actions from server (T, 14) or (T, 20)
            
        Returns:
            actions_ypr: (H, 14) array in YPR format for robot execution,
                         where H = min(T, execution_horizon)
        """

        action_dim = raw_actions.shape[1]
        if action_dim == 20:
            actions_ypr = convert_6drot_to_ypr(raw_actions)
        elif action_dim == 14:
            actions_ypr = raw_actions
        else:
            raise ValueError(f"Unsupported action dimension {action_dim}; expected 14 (YPR) or 20 (6D rot)")

        # Slice to execution_horizon
        original_len = actions_ypr.shape[0]
        actions_ypr = actions_ypr[:self.execution_horizon]
        if original_len != actions_ypr.shape[0]:
            print(f"[DWMClient] Sliced actions from {original_len} to {actions_ypr.shape[0]} (execution_horizon={self.execution_horizon})")

        if self.motion_smooth_mode == "interp" and actions_ypr.shape[0] > 1:
            actions_ypr = self._interpolate_action_horizon(actions_ypr)
        
        return actions_ypr

    def _validate_motion_smoothing_args(self):
        """Validate motion smoothing mode configuration."""
        def _validate_step(name: str, val: int) -> int:
            if not isinstance(val, (int, np.integer)):
                raise ValueError(f"{name} must be an integer in [1, +inf), got {val}")
            if val < 1:
                raise ValueError(f"{name} must be in [1, +inf), got {val}")
            return int(val)

        valid_modes = {"lpf", "interp"}
        if self.motion_smooth_mode not in valid_modes:
            raise ValueError(
                f"motion_smooth_mode must be one of {sorted(valid_modes)}, got {self.motion_smooth_mode}"
            )
        self.interp_xyz_keyframe_step = _validate_step(
            "interp_xyz_keyframe_step", self.interp_xyz_keyframe_step
        )
        self.interp_orientation_keyframe_step = _validate_step(
            "interp_orientation_keyframe_step", self.interp_orientation_keyframe_step
        )
        self.interp_gripper_keyframe_step = _validate_step(
            "interp_gripper_keyframe_step", self.interp_gripper_keyframe_step
        )

    def _build_interp_keyframe_indices(self, horizon_len: int, stride: int) -> np.ndarray:
        """Build reduced keyframe indices, always preserving first and last."""
        if horizon_len <= 2:
            return np.arange(horizon_len, dtype=np.int64)
        idx = np.arange(0, horizon_len, stride, dtype=np.int64)
        if idx[-1] != horizon_len - 1:
            idx = np.append(idx, horizon_len - 1)
        return np.unique(idx)

    def _interpolate_arm_horizon(
        self,
        arm_actions: np.ndarray,
        keyframe_idx_xyz: np.ndarray,
        keyframe_idx_ypr: np.ndarray,
        keyframe_idx_grip: np.ndarray,
    ) -> np.ndarray:
        """Interpolate one arm's [xyz, ypr, grip] horizon with endpoint preservation."""
        horizon_len = arm_actions.shape[0]
        if horizon_len <= 1:
            return arm_actions.copy()

        t = np.arange(horizon_len, dtype=np.float64)
        out = np.empty_like(arm_actions, dtype=np.float64)

        # XYZ: shape-preserving PCHIP through reduced keyframes.
        tk_xyz = t[keyframe_idx_xyz]
        for dim in range(3):
            pchip_xyz = PchipInterpolator(tk_xyz, arm_actions[keyframe_idx_xyz, dim], extrapolate=False)
            out[:, dim] = pchip_xyz(t)

        # Gripper: shape-preserving PCHIP through reduced keyframes.
        tk_grip = t[keyframe_idx_grip]
        pchip_grip = PchipInterpolator(tk_grip, arm_actions[keyframe_idx_grip, 6], extrapolate=False)
        out[:, 6] = pchip_grip(t)

        # Orientation: quaternion SLERP with hemisphere alignment.
        tk_ypr = t[keyframe_idx_ypr]
        key_rot = R.from_euler("ZYX", arm_actions[keyframe_idx_ypr, 3:6], degrees=False)
        key_quat = key_rot.as_quat().copy()
        for i in range(1, key_quat.shape[0]):
            if np.dot(key_quat[i - 1], key_quat[i]) < 0.0:
                key_quat[i] *= -1.0
        slerp = Slerp(tk_ypr, R.from_quat(key_quat))
        out[:, 3:6] = slerp(t).as_euler("ZYX", degrees=False)
        out[:, 3:6] = self._wrap_angles(out[:, 3:6])

        # Ensure exact endpoint preservation.
        out[0, :] = arm_actions[0, :]
        out[-1, :] = arm_actions[-1, :]
        return out.astype(arm_actions.dtype, copy=False)

    def _log_interp_endpoint_check(
        self,
        original: np.ndarray,
        interpolated: np.ndarray,
        keyframe_idx_xyz: np.ndarray,
        keyframe_idx_ypr: np.ndarray,
        keyframe_idx_grip: np.ndarray,
    ):
        """Log endpoint differences to verify endpoint-preserving interpolation."""
        if original.shape[0] == 0:
            return
        for arm, arm_offset in (("left", 0), ("right", 7)):
            start_xyz_err = np.linalg.norm(interpolated[0, arm_offset:arm_offset + 3] - original[0, arm_offset:arm_offset + 3])
            end_xyz_err = np.linalg.norm(interpolated[-1, arm_offset:arm_offset + 3] - original[-1, arm_offset:arm_offset + 3])
            start_ypr_err = np.linalg.norm(
                self._wrapped_angle_delta(
                    interpolated[0, arm_offset + 3:arm_offset + 6],
                    original[0, arm_offset + 3:arm_offset + 6],
                )
            )
            end_ypr_err = np.linalg.norm(
                self._wrapped_angle_delta(
                    interpolated[-1, arm_offset + 3:arm_offset + 6],
                    original[-1, arm_offset + 3:arm_offset + 6],
                )
            )
            start_grip_err = abs(float(interpolated[0, arm_offset + 6] - original[0, arm_offset + 6]))
            end_grip_err = abs(float(interpolated[-1, arm_offset + 6] - original[-1, arm_offset + 6]))
            print(
                f"[DWMClient] interp endpoint check ({arm}): "
                f"start xyz/ypr/grip={start_xyz_err:.2e}/{start_ypr_err:.2e}/{start_grip_err:.2e}, "
                f"end xyz/ypr/grip={end_xyz_err:.2e}/{end_ypr_err:.2e}/{end_grip_err:.2e}"
            )
        print(
            f"[DWMClient] Interpolated horizon: T={original.shape[0]}, "
            f"keyframes xyz/ypr/grip={len(keyframe_idx_xyz)}/{len(keyframe_idx_ypr)}/{len(keyframe_idx_grip)}, "
            f"steps xyz/ypr/grip={self.interp_xyz_keyframe_step}/{self.interp_orientation_keyframe_step}/{self.interp_gripper_keyframe_step}"
        )

    def _interpolate_action_horizon(self, actions_ypr: np.ndarray) -> np.ndarray:
        """Smooth full action horizon while preserving start/end per arm."""
        if actions_ypr.shape[0] <= 1:
            return actions_ypr
        keyframe_idx_xyz = self._build_interp_keyframe_indices(
            actions_ypr.shape[0], self.interp_xyz_keyframe_step
        )
        keyframe_idx_ypr = self._build_interp_keyframe_indices(
            actions_ypr.shape[0], self.interp_orientation_keyframe_step
        )
        keyframe_idx_grip = self._build_interp_keyframe_indices(
            actions_ypr.shape[0], self.interp_gripper_keyframe_step
        )
        interpolated = actions_ypr.copy()
        interpolated[:, 0:7] = self._interpolate_arm_horizon(
            actions_ypr[:, 0:7], keyframe_idx_xyz, keyframe_idx_ypr, keyframe_idx_grip
        )
        interpolated[:, 7:14] = self._interpolate_arm_horizon(
            actions_ypr[:, 7:14], keyframe_idx_xyz, keyframe_idx_ypr, keyframe_idx_grip
        )
        self._log_interp_endpoint_check(
            actions_ypr, interpolated, keyframe_idx_xyz, keyframe_idx_ypr, keyframe_idx_grip
        )
        return interpolated

    def _validate_action_filter_args(self):
        """Validate optional action filtering parameters."""
        alpha_args = {
            "action_lpf_alpha_xyz": self.action_lpf_alpha_xyz,
            "action_lpf_alpha_ypr": self.action_lpf_alpha_ypr,
            "action_lpf_alpha_grip": self.action_lpf_alpha_grip,
        }
        nonnegative_args = {
            "action_slew_max_xyz_step": self.action_slew_max_xyz_step,
            "action_slew_max_ypr_step": self.action_slew_max_ypr_step,
            "action_slew_max_grip_step": self.action_slew_max_grip_step,
            "action_deadband_xyz": self.action_deadband_xyz,
            "action_deadband_ypr": self.action_deadband_ypr,
            "action_deadband_grip": self.action_deadband_grip,
        }

        for name, val in alpha_args.items():
            if val is None:
                continue
            if not (0.0 < val <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {val}")

        for name, val in nonnegative_args.items():
            if val is None:
                continue
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0, got {val}")

    @staticmethod
    def _wrapped_angle_delta(target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Compute shortest signed angular delta."""
        return np.arctan2(np.sin(target - current), np.cos(target - current))

    @staticmethod
    def _wrap_angles(angles: np.ndarray) -> np.ndarray:
        """Wrap angles to [-pi, pi]."""
        return np.arctan2(np.sin(angles), np.cos(angles))

    def _apply_action_filters(self, arm_action: np.ndarray, arm: str, apply_lpf: bool = True) -> np.ndarray:
        """Apply optional low-pass, slew-rate, and deadband filtering."""
        filtered = arm_action.copy()
        if not self._use_action_filtering:
            return filtered

        prev = self._last_filtered_arm_action.get(arm)
        if prev is None:
            self._last_filtered_arm_action[arm] = filtered.copy()
            return filtered

        # 1) Low-pass filter (EMA)
        if apply_lpf:
            if self.action_lpf_alpha_xyz is not None:
                filtered[0:3] = prev[0:3] + self.action_lpf_alpha_xyz * (filtered[0:3] - prev[0:3])

            if self.action_lpf_alpha_ypr is not None:
                ypr_delta = self._wrapped_angle_delta(filtered[3:6], prev[3:6])
                filtered[3:6] = prev[3:6] + self.action_lpf_alpha_ypr * ypr_delta

            if self.action_lpf_alpha_grip is not None:
                filtered[6] = prev[6] + self.action_lpf_alpha_grip * (filtered[6] - prev[6])

        # 2) Slew-rate limit
        if self.action_slew_max_xyz_step is not None:
            xyz_delta = filtered[0:3] - prev[0:3]
            xyz_delta_norm = np.linalg.norm(xyz_delta)
            if xyz_delta_norm > self.action_slew_max_xyz_step and xyz_delta_norm > 1e-12:
                filtered[0:3] = prev[0:3] + xyz_delta * (self.action_slew_max_xyz_step / xyz_delta_norm)

        if self.action_slew_max_ypr_step is not None:
            ypr_delta = self._wrapped_angle_delta(filtered[3:6], prev[3:6])
            ypr_delta_norm = np.linalg.norm(ypr_delta)
            if ypr_delta_norm > self.action_slew_max_ypr_step and ypr_delta_norm > 1e-12:
                ypr_delta = ypr_delta * (self.action_slew_max_ypr_step / ypr_delta_norm)
            filtered[3:6] = prev[3:6] + ypr_delta

        if self.action_slew_max_grip_step is not None:
            grip_delta = filtered[6] - prev[6]
            grip_delta = np.clip(grip_delta, -self.action_slew_max_grip_step, self.action_slew_max_grip_step)
            filtered[6] = prev[6] + grip_delta

        # 3) Deadband
        if self.action_deadband_xyz is not None:
            if np.linalg.norm(filtered[0:3] - prev[0:3]) < self.action_deadband_xyz:
                filtered[0:3] = prev[0:3]

        if self.action_deadband_ypr is not None:
            ypr_delta = self._wrapped_angle_delta(filtered[3:6], prev[3:6])
            if np.linalg.norm(ypr_delta) < self.action_deadband_ypr:
                filtered[3:6] = prev[3:6]

        if self.action_deadband_grip is not None:
            if abs(filtered[6] - prev[6]) < self.action_deadband_grip:
                filtered[6] = prev[6]

        filtered[3:6] = self._wrap_angles(filtered[3:6])
        self._last_filtered_arm_action[arm] = filtered.copy()
        return filtered
    
    def _fix_zero_actions(self, actions: np.ndarray, ee_poses: np.ndarray) -> np.ndarray:
        """Replace near-zero actions with current pose.
        
        Workaround for models trained with zeros for non-engaged arms.
        Computes the path length (sum of consecutive xyz distances) of each
        arm's trajectory. If below the threshold, replaces all of that arm's
        actions with the current ee pose.
        
        Also updates self._zero_arms to track which arms are outputting zeros,
        so we can zero out their observations to match training distribution.
        
        Args:
            actions: (T, 14) array of actions in YPR format
            ee_poses: (14,) current end-effector poses [left_7D, right_7D]
            
        Returns:
            Fixed actions array
        """
        if self.zero_action_threshold is None:
            return actions
        
        actions = actions.copy()
        threshold = self.zero_action_threshold
        alpha = self.action_lpf_alpha_xyz

        def _first_last_distance(xyz_seq: np.ndarray, arm: str) -> float:
            """Distance between first and last point of xyz trajectory (optionally EMA-smoothed)."""
            if xyz_seq.shape[0] <= 1:
                return 0.0
            if alpha is not None:
                prev_filtered = self._last_filtered_arm_action.get(arm)
                prev_xyz = prev_filtered[0:3].copy() if prev_filtered is not None else xyz_seq[0].copy()
                smoothed = np.empty_like(xyz_seq)
                for t in range(xyz_seq.shape[0]):
                    smoothed[t] = prev_xyz + alpha * (xyz_seq[t] - prev_xyz)
                    prev_xyz = smoothed[t]
                return float(np.linalg.norm(smoothed[-1] - smoothed[0]))
            return float(np.linalg.norm(xyz_seq[-1] - xyz_seq[0]))

        left_distance = _first_last_distance(actions[:, 0:3], "left")
        right_distance = _first_last_distance(actions[:, 7:10], "right")
        
        # Track which arms are outputting zeros (for observation zeroing)
        if left_distance < threshold:
            if not self._zero_arms.get("left", False):
                print(f"[DWMClient] Detected zero-action left arm (first_last_dist={left_distance:.4f}) - will zero observations")
            self._zero_arms["left"] = True
            actions[:, 0:7] = ee_poses[0:7]
        
        if right_distance < threshold:
            if not self._zero_arms.get("right", False):
                print(f"[DWMClient] Detected zero-action right arm (first_last_dist={right_distance:.4f}) - will zero observations")
            self._zero_arms["right"] = True
            actions[:, 7:14] = ee_poses[7:14]
        
        return actions
    
    def _zero_obs_for_bad_arms(self, obs: dict) -> dict:
        """Zero out ee_poses for arms that output near-zero actions.
        
        Workaround to match training distribution where non-engaged arms
        had zeros in their observations.
        
        Args:
            obs: Observation dictionary with 'ee_poses' key
            
        Returns:
            Modified observation dictionary
        """
        if self.zero_action_threshold is None:
            return obs
        
        ee_poses = obs.get('ee_poses')
        if ee_poses is None:
            return obs
        
        # Make a copy to avoid modifying the original
        obs = obs.copy()
        ee_poses = ee_poses.copy()
        
        if self._zero_arms.get("left", False):
            ee_poses[0:7] = 0.0  # Zero left arm ee_pose
        
        if self._zero_arms.get("right", False):
            ee_poses[7:14] = 0.0  # Zero right arm ee_pose
        
        obs['ee_poses'] = ee_poses
        return obs
    
    def step(self, i: int) -> bool:
        """Execute one control step.
        
        Args:
            i: Current timestep
            
        Returns:
            True if step succeeded, False to terminate
        """
        # Pause before capture when requesting new actions (reduces motion blur)
        need_new_actions = (self.actions is None or self.action_index >= self.execution_horizon)
        if need_new_actions and self.capture_pause_delay > 0:
            time.sleep(self.capture_pause_delay)

        # Get observations (with optional latency monitoring)
        if self.monitor_latency:
            obs, latency_info = self.robot.get_obs_with_latency()
            self._track_latency(latency_info, i)
        else:
            obs = self.robot.get_obs()

        # In single-arm mode, provide home proprio for the non-controlled arm.
        obs = self._set_inactive_arm_proprio_to_home(obs)
        
        # Store actual ee_poses before any modifications (for action fixing)
        actual_ee_poses = obs.get('ee_poses')
        if actual_ee_poses is not None:
            actual_ee_poses = actual_ee_poses.copy()
        
        if need_new_actions:
            # Print current robot state
            print(f"[DWMClient] Step {i}: Requesting new actions (action_index={self.action_index}, execution_horizon={self.execution_horizon})")
            print(f"[DWMClient] Current robot ee_poses: {actual_ee_poses}")
            
            # Zero out observations for arms that output near-zero actions
            # (matches training distribution where non-engaged arms had zeros)
            obs_for_model = self._zero_obs_for_bad_arms(obs)
            if self._zero_arms.get("left") or self._zero_arms.get("right"):
                print(f"[DWMClient] Zeroed obs ee_poses: {obs_for_model.get('ee_poses')}")
            
            raw_actions = self._request_actions(obs_for_model, i)
            print(f"[DWMClient] Raw actions from inference (shape {raw_actions.shape}):")
            print(f"  First action: {raw_actions[0]}")
            
            # Convert to YPR format if needed (auto-detects from dimension)
            self.actions = self._process_actions(raw_actions)
            
            # Fix near-zero actions (replace with actual current pose for safety)
            if self.zero_action_threshold is not None and actual_ee_poses is not None:
                self.actions = self._fix_zero_actions(self.actions, actual_ee_poses)
            
            print(f"[DWMClient] Final actions (shape {self.actions.shape}):")
            print(f"  First action [left_7D, right_7D]: {self.actions[0]}")
            
            # Reset action index for new chunk
            self.action_index = 0
            
            # Reset IK failure counts for new action chunk - give arms another chance
            for arm in self._ik_fail_count:
                if self._ik_fail_count[arm] >= self._ik_fail_skip_threshold:
                    print(f"[DWMClient] Resetting {arm} arm IK failure count for new action chunk")
                self._ik_fail_count[arm] = 0
        
        if self.actions is None:
            return False
        
        # Get action at current index
        act_idx = self.action_index
        if act_idx >= self.actions.shape[0]:
            return False
        print(f"[DWMClient] Executing action {act_idx} of {self.actions.shape[0]}")
        action = self.actions[act_idx]
        
        # Execute on robot (skip if dry_run - only read proprioception)
        if self.dry_run:
            if act_idx == 0:  # Only print once per action batch
                print(f"[DRY RUN] Would execute action: {action}")
        else:
            for arm in self.arms_list:
                arm_offset = 7 if arm == "right" and self.arms == "both" else 0
                if self.arms != "both" and arm == "right":
                    arm_offset = 0
                arm_action = action[arm_offset:arm_offset + 7].copy()
                arm_action = self._apply_action_filters(
                    arm_action,
                    arm,
                    apply_lpf=(self.motion_smooth_mode == "lpf"),
                )
                
                # Check if we should skip this arm due to small delta
                if self.skip_small_delta is not None and actual_ee_poses is not None:
                    current_xyz = actual_ee_poses[arm_offset:arm_offset + 3]
                    target_xyz = arm_action[0:3]
                    delta = np.linalg.norm(target_xyz - current_xyz)
                    
                    if delta < self.skip_small_delta:
                        # Only log once per arm when we start skipping
                        if not self._skipped_arms.get(arm, False):
                            print(f"[DWMClient] Skipping {arm} arm - delta {delta*1000:.1f}mm < threshold {self.skip_small_delta*1000:.1f}mm")
                            self._skipped_arms[arm] = True
                        continue  # Skip this arm entirely
                    else:
                        # Arm is moving again, clear skip flag
                        if self._skipped_arms.get(arm, False):
                            print(f"[DWMClient] Resuming {arm} arm - delta {delta*1000:.1f}mm")
                            self._skipped_arms[arm] = False
                
                # Check if arm is being skipped due to consecutive IK failures
                if self._ik_fail_count.get(arm, 0) >= self._ik_fail_skip_threshold:
                    # Only log occasionally to avoid spam
                    if act_idx == 0:
                        print(f"[DWMClient] Skipping {arm} arm - {self._ik_fail_count[arm]} consecutive IK failures")
                    continue
                
                print(f"[DWMClient] Executing {arm} arm action: xyz=[{arm_action[0]:.4f}, {arm_action[1]:.4f}, {arm_action[2]:.4f}] ypr=[{arm_action[3]:.4f}, {arm_action[4]:.4f}, {arm_action[5]:.4f}] grip={arm_action[6]:.3f}")
                result = self.robot.set_pose(arm_action, arm, velocity_limit=self.velocity_limit)
                
                # Track IK failures
                if result is None:
                    self._ik_fail_count[arm] = self._ik_fail_count.get(arm, 0) + 1
                    if self._ik_fail_count[arm] == self._ik_fail_skip_threshold:
                        print(f"[DWMClient] {arm} arm hit {self._ik_fail_skip_threshold} consecutive IK failures - will skip until next action chunk")
                else:
                    # Reset on success
                    if self._ik_fail_count.get(arm, 0) > 0:
                        print(f"[DWMClient] {arm} arm IK recovered after {self._ik_fail_count[arm]} failures")
                    self._ik_fail_count[arm] = 0
        
        # Increment action index for next step
        self.action_index += 1
        
        return True
    
    def _track_latency(self, latency_info: dict, step: int):
        """Track and report camera latency statistics.
        
        Args:
            latency_info: Dict mapping camera names to their latency info
            step: Current timestep
        """
        for cam_name, info in latency_info.items():
            if not info:
                continue
            
            # Initialize history for this camera
            if cam_name not in self._latency_history:
                self._latency_history[cam_name] = []
            
            frame_age_ms = info.get('frame_age_ms', 0)
            is_new = info.get('is_new_frame', True)
            
            # Track frame age
            self._latency_history[cam_name].append(frame_age_ms)
            if len(self._latency_history[cam_name]) > self._latency_window:
                self._latency_history[cam_name].pop(0)
            
            # Report when we're about to request new actions
            if self.actions is None or self.action_index >= self.execution_horizon:
                history = self._latency_history[cam_name]
                avg_age = np.mean(history) if history else 0
                max_age = max(history) if history else 0
                frame_num = info.get('frame_number', 0)
                capture_lat = info.get('capture_latency_ms', 0)
                
                # Warn if frame is stale
                stale_warning = ""
                if not is_new:
                    stale_warning = " [STALE - same frame as last read!]"
                elif frame_age_ms > 50:  # > 50ms is concerning at 30fps
                    stale_warning = " [HIGH LATENCY]"
                
                print(f"[Latency] {cam_name}: frame #{frame_num}, age={frame_age_ms:.1f}ms, "
                      f"avg={avg_age:.1f}ms, max={max_age:.1f}ms, capture_lat={capture_lat:.1f}ms{stale_warning}")
    
    def reset(self):
        """Reset robot to home position."""
        print("Resetting to home position...")
        self.robot.set_home()
        self.actions = None
        self.action_index = 0
        self._last_filtered_arm_action = {}
    
    def close(self):
        """Clean up resources."""
        self.sock.close()
        self.ctx.term()
        print("Client closed.")


# =============================================================================
# Rate Control
# =============================================================================

class RateController:
    """Simple rate limiter for control loop."""
    
    def __init__(self, freq_hz: float):
        self.period = 1.0 / freq_hz
        self.last_t = None
        
    def sleep(self):
        """Sleep to maintain desired rate."""
        now = time.time()
        if self.last_t is not None:
            elapsed = now - self.last_t
            if elapsed < self.period:
                time.sleep(self.period - elapsed)
        self.last_t = time.time()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DWM Streaming Client for YAM Robot")
    
    # Server connection
    parser.add_argument("--server", type=str, required=True,
                        help="Inference server address (e.g., tcp://192.168.1.100:5555)")
    
    # Robot configuration
    parser.add_argument("--arms", type=str, default="both",
                        choices=["left", "right", "both"],
                        help="Which arm(s) to control")
    parser.add_argument("--gripper-type", type=str, default="linear_4310",
                        choices=["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"],
                        help="YAM gripper type")
    parser.add_argument("--left-can", type=str, default="can0",
                        help="CAN interface for left arm")
    parser.add_argument("--right-can", type=str, default="can1",
                        help="CAN interface for right arm")
    
    # Control parameters
    parser.add_argument("--frequency", type=float, default=30.0,
                        help="Control loop frequency in Hz")
    parser.add_argument("--execution-horizon", type=int, required=True,
                        help="Number of actions to execute from each action chunk before "
                             "requesting new actions from the server.")
    parser.add_argument("--timeout-ms", type=int, default=5000,
                        help="Server request timeout in milliseconds")
    
    # Debug
    parser.add_argument("--dry-run", action="store_true",
                        help="Connect to real robot for proprioception but do NOT execute actions (robot will be homed)")
    
    # Safety
    parser.add_argument("--velocity-limit", type=float, default=1.0,
                        help="Maximum joint velocity limit in rad/s. If not specified, no limit is applied.")
    
    # Connection
    parser.add_argument("--wait-for-server", action="store_true",
                        help="Wait indefinitely for the inference server to become available")
    
    # Workarounds
    parser.add_argument("--zero-action-threshold", type=float, default=None,
                        help="Replace zero-action arms whose action sequence first-to-last "
                             "distance is below this threshold with current pose. "
                             "Workaround for models trained with zeros for non-engaged arms.")
    
    # IK parameters
    parser.add_argument("--max-ik-iters", type=int, default=500,
                        help="Maximum IK solver iterations. Lower = faster failure. Default: 500. "
                             "Try 50-100 for faster response when IK is failing.")
    parser.add_argument("--ik-error-threshold", type=float, default=None,
                        help="Skip arm commands when IK position error exceeds this value (meters). "
                             "Suggested: 0.02 (20mm) to skip badly-converged IK solutions.")
    
    # Small delta skip
    parser.add_argument("--skip-small-delta", type=float, default=None,
                        help="Skip arm execution when target position is within this distance (meters) "
                             "of current position. Avoids wasting IK on 'hold position' actions. "
                             "Suggested: 0.005-0.01 (5-10mm).")

    # Action smoothing (all disabled by default)
    parser.add_argument("--action-lpf-alpha-xyz", type=float, default=None,
                        help="Optional EMA alpha for xyz smoothing in (0, 1]. "
                             "Disabled if unset.")
    parser.add_argument("--action-lpf-alpha-ypr", type=float, default=None,
                        help="Optional EMA alpha for yaw/pitch/roll smoothing in (0, 1]. "
                             "Disabled if unset.")
    parser.add_argument("--action-lpf-alpha-grip", type=float, default=None,
                        help="Optional EMA alpha for gripper smoothing in (0, 1]. "
                             "Disabled if unset.")
    parser.add_argument("--action-slew-max-xyz-step", type=float, default=None,
                        help="Optional max xyz delta per control step (meters). Disabled if unset.")
    parser.add_argument("--action-slew-max-ypr-step", type=float, default=None,
                        help="Optional max ypr delta per control step (radians). Disabled if unset.")
    parser.add_argument("--action-slew-max-grip-step", type=float, default=None,
                        help="Optional max gripper delta per control step. Disabled if unset.")
    parser.add_argument("--action-deadband-xyz", type=float, default=None,
                        help="Optional xyz deadband relative to previous command (meters). Disabled if unset.")
    parser.add_argument("--action-deadband-ypr", type=float, default=None,
                        help="Optional ypr deadband relative to previous command (radians). Disabled if unset.")
    parser.add_argument("--action-deadband-grip", type=float, default=None,
                        help="Optional gripper deadband relative to previous command. Disabled if unset.")
    parser.add_argument("--motion-smooth-mode", type=str, default="lpf",
                        choices=["lpf", "interp"],
                        help="Motion smoothing mode: 'lpf' keeps per-step LPF/slew/deadband path, "
                             "'interp' smooths each action horizon via endpoint-preserving interpolation "
                             "then applies per-step slew/deadband.")
    parser.add_argument("--interp-xyz-keyframe-step", type=int, default=1,
                        help="Interpolation keyframe step for xyz, integer in [1, +inf). "
                             "Use 1 to keep default behavior (no interpolation smoothing).")
    parser.add_argument("--interp-orientation-keyframe-step", type=int, default=1,
                        help="Interpolation keyframe step for orientation (ypr), integer in [1, +inf). "
                             "Use 1 to keep default behavior (no interpolation smoothing).")
    parser.add_argument("--interp-gripper-keyframe-step", type=int, default=1,
                        help="Interpolation keyframe step for gripper, integer in [1, +inf). "
                             "Use 1 to keep default behavior (no interpolation smoothing).")
    
    # Latency monitoring
    parser.add_argument("--monitor-latency", action="store_true",
                        help="Enable camera latency monitoring. Reports frame age and capture "
                             "latency for each RealSense camera at every inference step.")
    parser.add_argument("--read-all-arms", action="store_true",
                        help="Read proprioception from both arms even when controlling one arm.")
    parser.add_argument("--disable-gripper-calibration", action="store_true",
                        help="Disable linear gripper auto-calibration at startup and use fixed limits.")
    parser.add_argument("--camera-fps-15", action="store_true",
                        help="Force all cameras to 15 FPS, overriding config values.")
    parser.add_argument("--capture-pause-delay", type=float, default=0.0,
                        help="Seconds to pause before capturing images when requesting new "
                             "actions. Reduces motion blur by allowing the robot to settle. "
                             "Suggested: 0.1-0.2. Default: 0 (no pause).")
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("DWM YAM Client Configuration")
    print("=" * 60)
    print(f"Server:            {args.server}")
    print(f"Arms:              {args.arms}")
    print(f"Gripper type:      {args.gripper_type}")
    print(f"Frequency:         {args.frequency} Hz")
    print(f"Execution horizon: {args.execution_horizon} actions")
    print(f"Dry run:           {args.dry_run}")
    print(f"Velocity limit:    {args.velocity_limit} rad/s" if args.velocity_limit else "Velocity limit:    None (unlimited)")
    print(f"Wait for server:   {args.wait_for_server}")
    print(f"Zero action fix:   {args.zero_action_threshold}m first-to-last distance threshold" if args.zero_action_threshold else "Zero action fix:   disabled")
    print(f"Max IK iters:      {args.max_ik_iters}")
    print(f"IK error thresh:   {args.ik_error_threshold}m" if args.ik_error_threshold else "IK error thresh:   None (no rejection)")
    print(f"Skip small delta:  {args.skip_small_delta}m" if args.skip_small_delta else "Skip small delta:  None (always execute)")
    print(f"LPF xyz alpha:     {args.action_lpf_alpha_xyz}" if args.action_lpf_alpha_xyz is not None else "LPF xyz alpha:     disabled")
    print(f"LPF ypr alpha:     {args.action_lpf_alpha_ypr}" if args.action_lpf_alpha_ypr is not None else "LPF ypr alpha:     disabled")
    print(f"LPF grip alpha:    {args.action_lpf_alpha_grip}" if args.action_lpf_alpha_grip is not None else "LPF grip alpha:    disabled")
    print(f"Slew xyz step:     {args.action_slew_max_xyz_step}m" if args.action_slew_max_xyz_step is not None else "Slew xyz step:     disabled")
    print(f"Slew ypr step:     {args.action_slew_max_ypr_step}rad" if args.action_slew_max_ypr_step is not None else "Slew ypr step:     disabled")
    print(f"Slew grip step:    {args.action_slew_max_grip_step}" if args.action_slew_max_grip_step is not None else "Slew grip step:    disabled")
    print(f"Deadband xyz:      {args.action_deadband_xyz}m" if args.action_deadband_xyz is not None else "Deadband xyz:      disabled")
    print(f"Deadband ypr:      {args.action_deadband_ypr}rad" if args.action_deadband_ypr is not None else "Deadband ypr:      disabled")
    print(f"Deadband grip:     {args.action_deadband_grip}" if args.action_deadband_grip is not None else "Deadband grip:     disabled")
    print(f"Motion smooth:     {args.motion_smooth_mode}")
    if args.motion_smooth_mode == "interp":
        print(f"Interp xyz step:   {args.interp_xyz_keyframe_step} (1 = no smoothing)")
        print(f"Interp orient step:{args.interp_orientation_keyframe_step} (1 = no smoothing)")
        print(f"Interp grip step:  {args.interp_gripper_keyframe_step} (1 = no smoothing)")
    else:
        print("Interp settings:   n/a (LPF mode)")
    print(f"Monitor latency:   {args.monitor_latency}")
    print(f"Read all arms:     {args.read_all_arms}")
    print(f"Disable grip calib:{args.disable_gripper_calibration}")
    print(f"Force cam fps 15:  {args.camera_fps_15}")
    print(f"Capture pause:     {args.capture_pause_delay}s" if args.capture_pause_delay > 0 else "Capture pause:     disabled")
    print("=" * 60)
    
    # Initialize client
    can_interfaces = {"left": args.left_can, "right": args.right_can}
    
    client = DWMYAMClient(
        server_addr=args.server,
        arms=args.arms,
        execution_horizon=args.execution_horizon,
        timeout_ms=args.timeout_ms,
        gripper_type=args.gripper_type,
        can_interfaces=can_interfaces,
        dry_run=args.dry_run,
        velocity_limit=args.velocity_limit,
        wait_for_server=args.wait_for_server,
        zero_action_threshold=args.zero_action_threshold,
        max_ik_iters=args.max_ik_iters,
        ik_error_threshold=args.ik_error_threshold,
        skip_small_delta=args.skip_small_delta,
        monitor_latency=args.monitor_latency,
        read_all_arms=args.read_all_arms,
        disable_gripper_calibration=args.disable_gripper_calibration,
        camera_fps_15=args.camera_fps_15,
        action_lpf_alpha_xyz=args.action_lpf_alpha_xyz,
        action_lpf_alpha_ypr=args.action_lpf_alpha_ypr,
        action_lpf_alpha_grip=args.action_lpf_alpha_grip,
        action_slew_max_xyz_step=args.action_slew_max_xyz_step,
        action_slew_max_ypr_step=args.action_slew_max_ypr_step,
        action_slew_max_grip_step=args.action_slew_max_grip_step,
        action_deadband_xyz=args.action_deadband_xyz,
        action_deadband_ypr=args.action_deadband_ypr,
        action_deadband_grip=args.action_deadband_grip,
        capture_pause_delay=args.capture_pause_delay,
        motion_smooth_mode=args.motion_smooth_mode,
        interp_xyz_keyframe_step=args.interp_xyz_keyframe_step,
        interp_orientation_keyframe_step=args.interp_orientation_keyframe_step,
        interp_gripper_keyframe_step=args.interp_gripper_keyframe_step,
    )
    
    # Initialize rate controller
    rate = RateController(args.frequency)
    
    # Reset to home
    client.reset()
    print("\nStarting control loop. Press Ctrl+C to stop.\n")
    
    step = 0
    try:
        while True:
            if not client.step(step):
                print("Rollout complete.")
                break
            rate.sleep()
            step += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        client.close()
        print("Done.")


if __name__ == "__main__":
    main()
