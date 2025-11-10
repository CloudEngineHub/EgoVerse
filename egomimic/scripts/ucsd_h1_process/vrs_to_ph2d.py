import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm, trange

from projectaria_tools.core import data_provider, mps, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

import h5py

def save_compressed_imgs_hdf5(imgs, image_key, hf, shape="CHW"):
    ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    compressed_len_list = []
    encoded_img_list = []
    num_imgs = len(imgs)
    for i in range(num_imgs):
        if shape == "CHW":
            img = imgs[i].transpose(1, 2, 0)
        elif shape == "HWC":
            img = imgs[i]
        else:
            raise ValueError(f"Invalid shape: {shape}")
        _, img_encode = cv2.imencode('.jpg', img, ENCODE_PARAM)
        encoded_img_list.append(img_encode)
        compressed_len_list.append(len(img_encode))
                
    max_len = max(compressed_len_list)
    # create dataset
    hf.create_dataset(image_key, (num_imgs, max_len), dtype=np.uint8)
    for i in range(num_imgs):
        hf[image_key][i, :compressed_len_list[i]] = encoded_img_list[i].flatten()
    
    return encoded_img_list, compressed_len_list, max_len

# Transform finger positions from world frame to local wrist frame
def transform_hand_keypoints_world_to_local(points, wrist_mat):
    """
    Transform hand keypoints from world frame to local wrist frame.
    This is the inverse of transform_hand_keypoints_local_to_world.
    
    Args:
        points: numpy array of shape (N, 3) containing world-frame hand keypoints
        wrist_mat: 4x4 transformation matrix from local wrist frame to world frame
    
    Returns:
        numpy array of shape (N, 3) containing local-frame hand keypoints
    """
    # Convert to homogeneous coordinates
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    # Apply inverse transformation
    wrist_mat_inv = np.linalg.inv(wrist_mat)
    transformed = np.dot(wrist_mat_inv, points_h.T).T
    return transformed[:, :3]  # Return only xyz coordinates

# ====== BEGIN ARIA CONSTANTS ======

NORMAL_VIS_LEN = 0.05  # meters
ARIA_SLAM_TO_CENTER_LEN = 0.2 # meters

Z_FRONT_TO_Z_UP = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]
])

ARIA_LEFT_HAND_TO_H1_LEFT_HAND = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

ARIA_RIGHT_HAND_TO_H1_RIGHT_HAND = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1]
])

time_domain: TimeDomain = TimeDomain.DEVICE_TIME
time_query_closest: TimeQueryOptions = TimeQueryOptions.CLOSEST

from cet.utils import fk_cmd_dict2policy

# wrist, thumb, index, middle, ring, pinky
# https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/hand_tracking
# https://facebookresearch.github.io/projectaria_tools/assets/images/21-keypoints-36dacdda7266325ce379b7e90c5a31b7.png
ARIA_FINGERTIP_INDICES = [5, 0, 1, 2, 3, 4]

# ====== END ARIA CONSTANTS ======

def process_one_vrs_file(vrs_sample_path, save_dir, meta_info):
    # VRS provider of raw data
    provider = data_provider.create_vrs_data_provider(vrs_sample_path)

    annotation_dir_name = "mps_" + os.path.basename(vrs_sample_path).replace(".vrs", "_vrs")
    mps_sample_path = os.path.join(os.path.dirname(vrs_sample_path), annotation_dir_name)

    # MPS data provider of annotations
    mps_data_provider = mps.MpsDataProvider(mps.MpsDataPathsProvider(mps_sample_path).get_data_paths())

    assert(mps_data_provider.has_general_eyegaze() and
        mps_data_provider.has_open_loop_poses() and
        mps_data_provider.has_closed_loop_poses() and
        mps_data_provider.has_semidense_point_cloud() and
        mps_data_provider.has_hand_tracking_results())

    ## Load device SLAM trajectory
    closed_loop_trajectory = os.path.join(
        mps_sample_path, "slam", "closed_loop_trajectory.csv"
    )
    mps_trajectory = mps.read_closed_loop_trajectory(closed_loop_trajectory)
    mps_trajectory_ns_timestamp_arr = np.zeros(len(mps_trajectory), dtype=np.uint64)

    for i in range(len(mps_trajectory)):
        mps_trajectory_ns_timestamp_arr[i] = mps_trajectory[i].tracking_timestamp.total_seconds() * 1e9

    # Hand tracking
    hand_tracking_results_path = os.path.join(
        mps_sample_path, "hand_tracking", "hand_tracking_results.csv"
    )

    ## Load hand tracking
    hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(
        hand_tracking_results_path
    )

    assert len(mps_trajectory) != 0 and len(hand_tracking_results) != 0

    device_calibration = provider.get_device_calibration()

    # Get stream ids, stream labels, stream timestamps, and camera calibrations for RGB and SLAM cameras
    stream_ids: Dict[str, StreamId] = {
        "rgb": StreamId("214-1"),
        "slam-left": StreamId("1201-1"),
        "slam-right": StreamId("1201-2"),
    }

    stream_labels: Dict[str, str] = {
        key: provider.get_label_from_stream_id(stream_id)
        for key, stream_id in stream_ids.items()
    }
    stream_timestamps_ns: Dict[str, List[int]] = {
        key: provider.get_timestamps_ns(stream_id, time_domain)
        for key, stream_id in stream_ids.items()
    }
    camera_calibrations = {
        key: device_calibration.get_camera_calib(stream_label)
        for key, stream_label in stream_labels.items()
    }
    for key, camera_calibration in camera_calibrations.items():
        assert camera_calibration is not None, f"no camera calibration for {key}"

    # Get device calibration and transform from device to sensor
    device_calibration = provider.get_device_calibration()

    def undistort_to_linear(provider, stream_ids, raw_image, camera_label="rgb"):
        camera_label = provider.get_label_from_stream_id(stream_ids[camera_label])
        calib = provider.get_device_calibration().get_camera_calib(camera_label)
        warped = calibration.get_linear_camera_calibration(
            480, 640, 133.25430222 * 2, camera_label, calib.get_transform_device_camera()
        )
        warped_image = calibration.distort_by_calibration(raw_image, warped, calib)
        warped_rot = np.rot90(warped_image, k=3)
        return warped_rot

    # ================================================
    # Begin main function
    # ================================================

    # Synchronize sensor observations and annotations
    max_frames = len(stream_timestamps_ns["rgb"])
    init_rotation_world_device = mps_trajectory[0].transform_world_device.rotation().to_matrix() @ Z_FRONT_TO_Z_UP
    init_translation_world_device = mps_trajectory[0].transform_world_device.translation()
    
    # Process the entire VRS file. Segment data by glass pitch and hand visibility
    all_frames_dict = {
        "rgb_image": [],
        "left_hand_wrist_pose": [],
        "right_hand_wrist_pose": [],
        "left_hand_keypoints": [],
        "right_hand_keypoints": [],
        "head_pose": [],
        "left_hand_available": [],
        "right_hand_available": [],
    }
    cur_episode_idx = 0
    for frame_idx in trange(max_frames):
        # Get landmarks data for current frame
        sample_timestamp_ns = stream_timestamps_ns["rgb"][frame_idx]

        sample_frames = {
            key: provider.get_image_data_by_time_ns(
                stream_id, sample_timestamp_ns, time_domain, time_query_closest
            )[0]
            for key, stream_id in stream_ids.items()
        }

        rgb_image = sample_frames["rgb"].to_numpy_array()

        rgb_image = undistort_to_linear(provider, stream_ids, rgb_image)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        frame_hand_tracking_result = mps_data_provider.get_hand_tracking_result(
            sample_timestamp_ns, time_query_closest
        )
        
        # Get head pose from trajectory
        timestamp_abs_offset = np.abs(mps_trajectory_ns_timestamp_arr - sample_timestamp_ns)
        closest_idx = np.argmin(timestamp_abs_offset)
        # timestamp_abs_offset is in nanoseconds
        if timestamp_abs_offset[closest_idx] > 0.1 * 1e9:
            print("Dropping frame {} due to MPS/VRS timestamp offset too large. Should only happen in beginning/end".format(frame_idx))
            continue
        
        rotation_world_device = mps_trajectory[closest_idx].transform_world_device.rotation().to_matrix() @ Z_FRONT_TO_Z_UP
        translation_world_device = mps_trajectory[closest_idx].transform_world_device.translation()

        # Translate head poses so that they are w.r.t. to the initial head pose
        rotation_world_device = init_rotation_world_device.T @ rotation_world_device
        # TODO(roger): for the current version, we mask the translation of the head pose
        translation_world_device = np.zeros_like(translation_world_device)
        
        # Create head transformation matrix
        head_mat = np.eye(4)
        head_mat[:3, :3] = rotation_world_device
        head_mat[:3, 3] = translation_world_device
        
        # Get hand landmarks
        left_landmarks = frame_hand_tracking_result.left_hand.landmark_positions_device if frame_hand_tracking_result.left_hand else []
        right_landmarks = frame_hand_tracking_result.right_hand.landmark_positions_device if frame_hand_tracking_result.right_hand else []
        
        # Convert to numpy arrays for easier handling
        left_array = np.array([landmark for landmark in left_landmarks]) if left_landmarks else np.empty((0, 3))
        right_array = np.array([landmark for landmark in right_landmarks]) if right_landmarks else np.empty((0, 3))

        # Convert to x-front, y-right, z-up coordinate system\
        LEFT_HAND_AVAILABLE = len(left_array) > 0
        RIGHT_HAND_AVAILABLE = len(right_array) > 0

        # Apply same transformations to left_hand_wrist_pose
        # Create coordinate transformation matrix: [z, y, -x] mapping
        coord_transform = np.eye(4)
        coord_transform[:3, :3] = np.array([
            [0, 0, 1],   # new x = old z
            [0, 1, 0],   # new y = old y  
            [-1, 0, 0]   # new z = -old x
        ])
        
        # Create rotation matrix
        rotation_transform = np.eye(4)
        rotation_transform[:3, :3] = rotation_world_device.T

        if LEFT_HAND_AVAILABLE:
            left_array = left_array[:, [2, 1, 0]] * [1, 1, -1]
            # Before process: coordinate is relative to camera-rgb on the left side of the glass
            left_array[:, 1] += ARIA_SLAM_TO_CENTER_LEN
            # Apply rotation to align with world coordinate system
            left_array = left_array @ rotation_world_device.T
            
            left_hand_wrist_pose = coord_transform @ frame_hand_tracking_result.left_hand.transform_device_wrist.to_matrix()
            left_hand_wrist_pose[1, 3] += ARIA_SLAM_TO_CENTER_LEN
            left_hand_wrist_pose = left_hand_wrist_pose.T @ rotation_transform
            left_hand_wrist_pose = left_hand_wrist_pose.T
        else:
            left_array = None
            left_hand_wrist_pose = None
        
        if RIGHT_HAND_AVAILABLE:
            right_array = right_array[:, [2, 1, 0]] * [1, 1, -1]
            # Before process: coordinate is relative to camera-rgb on the right side of the glass
            right_array[:, 1] += ARIA_SLAM_TO_CENTER_LEN
            # Apply rotation to align with world coordinate system
            right_array = right_array @ rotation_world_device.T

            right_hand_wrist_pose = coord_transform @ frame_hand_tracking_result.right_hand.transform_device_wrist.to_matrix()
            right_hand_wrist_pose[1, 3] += ARIA_SLAM_TO_CENTER_LEN
            right_hand_wrist_pose = right_hand_wrist_pose.T @ rotation_transform
            right_hand_wrist_pose = right_hand_wrist_pose.T
        else:
            right_array = None
            right_hand_wrist_pose = None

        all_frames_dict["rgb_image"].append(rgb_image)
        all_frames_dict["left_hand_wrist_pose"].append(left_hand_wrist_pose)
        all_frames_dict["right_hand_wrist_pose"].append(right_hand_wrist_pose)
        all_frames_dict["left_hand_keypoints"].append(left_array)
        all_frames_dict["right_hand_keypoints"].append(right_array)
        all_frames_dict["head_pose"].append(head_mat)
        all_frames_dict["left_hand_available"].append(LEFT_HAND_AVAILABLE)
        all_frames_dict["right_hand_available"].append(RIGHT_HAND_AVAILABLE)
    
    per_episode_dict = segment_episodes(all_frames_dict)

    for episode_idx, episode_dict in enumerate(per_episode_dict):
        post_process_one_episode(episode_dict, episode_idx, save_dir, meta_info)

def get_episode_idx(both_hands_available_list, sliding_window_length):
    cur_episode_idx = -1
    episode_idx_list = []
    for i in range(len(both_hands_available_list)):
        frame_start_idx = max(0, i - sliding_window_length // 2)
        frame_end_idx = min(len(both_hands_available_list), i + sliding_window_length // 2)
        if np.all(both_hands_available_list[frame_start_idx:frame_end_idx]):
            # In a valid frame
            if len(episode_idx_list) == 0 or episode_idx_list[-1] == -1:
                cur_episode_idx += 1
            episode_idx_list.append(cur_episode_idx)
        else:
            episode_idx_list.append(-1)
    return episode_idx_list

def segment_episodes(all_frames_dict, sliding_window_length=40):
    per_episode_dict = []

    both_hands_available_list = []
    for i in range(len(all_frames_dict["rgb_image"])):
        both_hands_available = all_frames_dict["left_hand_available"][i] and all_frames_dict["right_hand_available"][i]
        both_hands_available_list.append(both_hands_available)
    
    episode_idx_list = get_episode_idx(both_hands_available_list, sliding_window_length)

    assert len(episode_idx_list) == len(all_frames_dict["rgb_image"])

    num_episodes = max(episode_idx_list) + 1
    for episode_idx in range(num_episodes):
        episode_dict = {
            key: [all_frames_dict[key][i] for i in range(len(all_frames_dict[key])) if episode_idx_list[i] == episode_idx]
            for key in all_frames_dict.keys()
        }
        per_episode_dict.append(episode_dict)
    return per_episode_dict

def post_process_one_episode(episode_dict, episode_idx, save_dir, meta_info):
    episode_frame_cnt = len(episode_dict["rgb_image"])
    # Assume filtered dictionary that has continuous frames for a single episode
    episode_save_path = os.path.join(save_dir, f"aria_processed_episode_{episode_idx}.hdf5")

    fk_cmd_dict = {"head_mat":[], "rel_left_wrist_mat": [], "rel_right_wrist_mat": [], "rel_left_hand_keypoints": [], "rel_right_hand_keypoints": []}

    fk_cmd_dict["head_mat"] = np.array(episode_dict["head_pose"])  # (N, 4, 4)
    fk_cmd_dict["rel_left_wrist_mat"] = np.array(episode_dict["left_hand_wrist_pose"])  # (N, 4, 4)
    fk_cmd_dict["rel_right_wrist_mat"] = np.array(episode_dict["right_hand_wrist_pose"])  # (N, 4, 4)

    # NOTE(roger, 2025-6): Data from Aria has 21 keypoints
    # https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/hand_tracking
    aria_left_hand_keypoints = np.array(episode_dict["left_hand_keypoints"])  # (N, 21, 3)
    aria_right_hand_keypoints = np.array(episode_dict["right_hand_keypoints"])  # (N, 21, 3)

    # Handling transformations
    for frame_idx in range(episode_frame_cnt):
        # Transform from world to local wrist frame
        aria_left_hand_keypoints[frame_idx, :, :] = transform_hand_keypoints_world_to_local(
            aria_left_hand_keypoints[frame_idx, :, :],
            fk_cmd_dict["rel_left_wrist_mat"][frame_idx]
        )
        aria_right_hand_keypoints[frame_idx, :, :] = transform_hand_keypoints_world_to_local(
            aria_right_hand_keypoints[frame_idx, :, :],
            fk_cmd_dict["rel_right_wrist_mat"][frame_idx]
        )

    rel_left_hand_keypoints = aria_left_hand_keypoints[:, ARIA_FINGERTIP_INDICES, :]
    rel_right_hand_keypoints = aria_right_hand_keypoints[:, ARIA_FINGERTIP_INDICES, :]

    fk_cmd_dict["rel_left_hand_keypoints"] = rel_left_hand_keypoints
    fk_cmd_dict["rel_right_hand_keypoints"] = rel_right_hand_keypoints

    # Handling rotation convetnions. Make it consistent to H1 in pinnochio
    for frame_idx in range(episode_frame_cnt):
        left_wrist_mat = fk_cmd_dict['rel_left_wrist_mat'][frame_idx]
        left_fk_rot = left_wrist_mat[:3, :3] @ ARIA_LEFT_HAND_TO_H1_LEFT_HAND
        fk_cmd_dict['rel_left_wrist_mat'][frame_idx][:3, :3] = left_fk_rot

        # Left finger keypoints
        left_finger_keypoints_n3 = fk_cmd_dict['rel_left_hand_keypoints'][frame_idx]
        left_finger_keypoints_n3 = left_finger_keypoints_n3 @ ARIA_LEFT_HAND_TO_H1_LEFT_HAND
        fk_cmd_dict['rel_left_hand_keypoints'][frame_idx] = left_finger_keypoints_n3
        
        # Change rotation for right hand
        right_wrist_mat = fk_cmd_dict['rel_right_wrist_mat'][frame_idx]
        right_fk_rot = right_wrist_mat[:3, :3] @ ARIA_RIGHT_HAND_TO_H1_RIGHT_HAND
        fk_cmd_dict['rel_right_wrist_mat'][frame_idx][:3, :3] = right_fk_rot

        # Right finger keypoints
        right_finger_keypoints_n3 = fk_cmd_dict['rel_right_hand_keypoints'][frame_idx]
        right_finger_keypoints_n3 = right_finger_keypoints_n3 @ ARIA_RIGHT_HAND_TO_H1_RIGHT_HAND
        fk_cmd_dict['rel_right_hand_keypoints'][frame_idx] = right_finger_keypoints_n3

    human_actions = fk_cmd_dict2policy(fk_cmd_dict, episode_frame_cnt)
    human_states = np.zeros_like(human_actions)
    human_states[1:] = human_actions[:-1]
    human_states[0] = human_actions[0]
    assert human_actions.shape[0] == episode_frame_cnt

    with h5py.File(episode_save_path, 'w') as hf:
        assert len(episode_dict["rgb_image"]) == episode_frame_cnt
        encoded_list_left, _, _ = save_compressed_imgs_hdf5(episode_dict["rgb_image"], 'observation.image.left', hf, shape="HWC")

        # Save a string "observation.image.left" to "observation.image.right"
        hf.create_dataset('observation.image.right', data=b'observation.image.left')

        hf.create_dataset('action', data=human_actions.astype(np.float32))
        hf.create_dataset('observation.state', data=human_states.astype(np.float32))
        hf.attrs['sim'] = False
        hf.attrs['description'] = meta_info['description']
        hf.attrs['embodiment'] = meta_info['embodiment']
        
if __name__ == "__main__":
    vrs_sample_path = "/home/roger/EgoVerse/data/raw/bag_grocery/bag_groceries_wang_scene_5_recording_4.vrs"
    save_dir = "processed"
    os.makedirs(save_dir, exist_ok=True)

    meta_info = {
        "description": "Aria walking test",
        "embodiment": "Aria"
    }

    process_one_vrs_file(vrs_sample_path, save_dir, meta_info)
