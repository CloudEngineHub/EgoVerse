import argparse
import os
from pathlib import Path
import shutil
import traceback
import cv2
import h5py
import torch
import logging
from enum import Enum

import time

import numpy as np

from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
EPISODE_LENGTH = 1000
HORIZON_DEFAULT = 20
# UCSD H1 runs at 30 Hz
FPS = 30
H1_ARM_TYPE = "bimanual"
CHUNK_LENGTH_ACT = 100

RETARGETTING_INDICES = [0, 4, 9, 14, 19, 24]

OUTPUT_LEFT_EEF = np.arange(80, 89)
OUTPUT_RIGHT_EEF = np.arange(30, 39)
OUTPUT_HEAD_EEF = np.arange(0, 9)
OUTPUT_LEFT_KEYPOINTS = np.arange(10, 10 + 3 * len(RETARGETTING_INDICES))
assert OUTPUT_LEFT_KEYPOINTS[-1] < OUTPUT_RIGHT_EEF[0]
OUTPUT_RIGHT_KEYPOINTS = np.arange(40, 40 + 3 * len(RETARGETTING_INDICES))
assert OUTPUT_RIGHT_KEYPOINTS[-1] < OUTPUT_LEFT_EEF[0]

def cmd_dict_from_128dim(action, finger_tip_only):
    left_wrist_mat = np.eye(4)
    left_wrist_mat[0:3, 3] = action[OUTPUT_LEFT_EEF[0:3]]
    left_wrist_mat[0:3, 0:3] = rotation_6d_to_matrix(torch.tensor(action[OUTPUT_LEFT_EEF[3:]]).unsqueeze(0)).numpy()

    left_hand_keypoints = np.zeros((25,3))
    left_hand_keypoints[RETARGETTING_INDICES] = action[OUTPUT_LEFT_KEYPOINTS].reshape((6,3))

    right_wrist_mat = np.eye(4)
    right_wrist_mat[0:3, 3] = action[OUTPUT_RIGHT_EEF[0:3]]
    right_wrist_mat[0:3, 0:3] = rotation_6d_to_matrix(torch.tensor(action[OUTPUT_RIGHT_EEF[3:]]).unsqueeze(0)).numpy()

    right_hand_keypoints = np.zeros((25,3))
    right_hand_keypoints[RETARGETTING_INDICES] = action[OUTPUT_RIGHT_KEYPOINTS].reshape((6,3))

    head_mat = np.eye(4)
    head_mat[0:3, 3] = action[OUTPUT_HEAD_EEF[0:3]]
    head_rmat = rotation_6d_to_matrix(torch.tensor(action[OUTPUT_HEAD_EEF[3:]]).unsqueeze(0)).squeeze().numpy()
    head_mat[0:3, 0:3] = head_rmat

    if finger_tip_only:
        left_hand_keypoints = left_hand_keypoints[RETARGETTING_INDICES]
        right_hand_keypoints = right_hand_keypoints[RETARGETTING_INDICES]

    return {
        'left_wrist_mat': left_wrist_mat,
        'left_hand_kpts': left_hand_keypoints,
        'right_wrist_mat': right_wrist_mat,
        'right_hand_kpts': right_hand_keypoints,
        'head_mat': head_mat
    }

def rebuild_128dim_action(action_dict):

    B, T = action_dict["head_6d_rot"].shape[:2]
    action_128 = np.zeros((B, T, 128), dtype=np.float32)

    # ─────────── head ───────────
    action_128[:, :, OUTPUT_HEAD_EEF[3:9]] = action_dict["head_6d_rot"]

    # ───────── key‑points ───────
    action_128[:, :, OUTPUT_LEFT_KEYPOINTS]  = action_dict["left_hand_kpts"]
    action_128[:, :, OUTPUT_RIGHT_KEYPOINTS] = action_dict["right_hand_kpts"]

    # ─────── right arm EEF ──────
    action_128[:, :, OUTPUT_RIGHT_EEF[0:3]] = -action_dict["actions_cartesian"][:, :, 3:6]
    action_128[:, :, OUTPUT_RIGHT_EEF[3:9]] = action_dict["right_wrist_6d_rot"]

    # ─────── left  arm EEF ──────
    action_128[:, :, OUTPUT_LEFT_EEF[0:3]]  = -action_dict["actions_cartesian"][:, :, 0:3]
    action_128[:, :, OUTPUT_LEFT_EEF[3:9]]  = action_dict["left_wrist_6d_rot"]
    return action_128
    
def prepare_training_data(left_img, right_img, state, action = None):

    data_dict = {
        "observations.state.ee_pose": [],
        "observations.state.ucsd_h1.left_wrist_6d_rot": [],
        "observations.state.ucsd_h1.right_wrist_6d_rot": [],
        "observations.state.ucsd_h1.left_hand_kpts": [],
        "observations.state.ucsd_h1.right_hand_kpts": [],
        "observations.images.front_img_1": left_img[None],
        "observations.images.front_img_2": right_img[None],
        "observations.state.hp_human_states": state,
        # "actions_cartesian": [],
        # "actions.ucsd_h1.left_hand_kpts": [],
        # "actions.ucsd_h1.right_hand_kpts":[] ,
        # "actions.ucsd_h1.left_wrist_6d_rot": [],
        # "actions.ucsd_h1.right_wrist_6d_rot": [],
        # "actions.ucsd_h1.head_6d_rot": [],
        'metadata.embodiment': torch.tensor([6])
    }

    cur_cmd_dict = cmd_dict_from_128dim(state, finger_tip_only=True)
    # Translation
    cur_state = np.array([
        cur_cmd_dict["left_wrist_mat"][0:3, 3],
        cur_cmd_dict["right_wrist_mat"][0:3, 3]
    ]).flatten()
    data_dict["observations.state.ee_pose"].append(cur_state)
    # Left hand keypoints
    data_dict["observations.state.ucsd_h1.left_hand_kpts"].append(cur_cmd_dict["left_hand_kpts"].flatten())
    # Right hand keypoints
    data_dict["observations.state.ucsd_h1.right_hand_kpts"].append(cur_cmd_dict["right_hand_kpts"].flatten())
    # Left wrist 6d rot
    data_dict["observations.state.ucsd_h1.left_wrist_6d_rot"].append(state[OUTPUT_LEFT_EEF[3:]])
    # Right wrist 6d rot
    data_dict["observations.state.ucsd_h1.right_wrist_6d_rot"].append(state[OUTPUT_RIGHT_EEF[3:]])

    if action != None:
        cur_cmd_dict = cmd_dict_from_128dim(action, finger_tip_only=True)
        # Translation
        cur_action = np.array([
            cur_cmd_dict["left_wrist_mat"][0:3, 3],
            cur_cmd_dict["right_wrist_mat"][0:3, 3]
        ]).flatten()
        data_dict["actions_cartesian"].append(cur_action)
        # Left hand keypoints
        data_dict["actions.ucsd_h1.left_hand_kpts"].append(cur_cmd_dict["left_hand_kpts"].flatten())
        # Right hand keypoints
        data_dict["actions.ucsd_h1.right_hand_kpts"].append(cur_cmd_dict["right_hand_kpts"].flatten())
        # Left wrist 6d rot
        data_dict["actions.ucsd_h1.left_wrist_6d_rot"].append(action[OUTPUT_LEFT_EEF[3:]])
        # Right wrist 6d rot
        data_dict["actions.ucsd_h1.right_wrist_6d_rot"].append(action[OUTPUT_RIGHT_EEF[3:]])
        # Head 6d rot
        data_dict["actions.ucsd_h1.head_6d_rot"].append(action[OUTPUT_HEAD_EEF[3:]])

    for k in data_dict.keys():
        data_dict[k] = np.array(data_dict[k])
    if True:
        # TODO: temporary solution to match axes direction.
        # Better way is to fix rotations + reference frames to be consistent with Aria.
        data_dict["observations.state.ee_pose"] = -1 * data_dict["observations.state.ee_pose"]
        # data_dict["actions_cartesian"] = -1 * data_dict["actions_cartesian"]
    
    for key, value in data_dict.items():
        data_dict[key] = torch.from_numpy(value)
    
    return data_dict