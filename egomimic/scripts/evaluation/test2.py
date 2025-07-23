import argparse
import numpy as np
import time

import pybullet as p
import pybullet_data

from egomimic.utils.egomimicUtils import *
from egomimic.utils.real_utils import *

import torch
import torchvision

from rldb.utils import *

from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)

from eve.constants import DT, FOLLOWER_GRIPPER_JOINT_OPEN, START_ARM_POSE

from egomimic.utils.egomimicUtils import *

from eve.robot_utils import move_grippers
from eve.real_env import make_real_env

joint_type_dict = {
    p.JOINT_REVOLUTE: "Revolute",
    p.JOINT_PRISMATIC: "Prismatic",
    p.JOINT_SPHERICAL: "Spherical",
    p.JOINT_PLANAR: "Planar",
    p.JOINT_FIXED: "Fixed"
}

def init_pybullet(urdf_path, GUI=False):
    if GUI:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    p.setGravity(0, 0, 9.81)
    return robot_id

def print_joint_info(robot_id):
    num_joints = p.getNumJoints(robot_id)
    print(f"Number of Joints: {num_joints}")
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_index = joint_info[0]
        joint_name = joint_info[1].decode('utf-8')
        link_name = joint_info[12].decode('utf-8')
        joint_type = joint_info[2]
        print(f"Joint Index: {joint_index}, Joint Name: {joint_name}, Link Type: {link_name}, Joint Type: {joint_type_dict[joint_type]}")
        
def get_joint_limits(robot_id):
    num_joints = p.getNumJoints(robot_id)
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    rest_poses = []

    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[2] == p.JOINT_REVOLUTE or joint_info[2] == p.JOINT_PRISMATIC:
            lower_limits.append(joint_info[8])  # lower limit
            upper_limits.append(joint_info[9])  # upper limit
            joint_ranges.append(joint_info[9] - joint_info[8])
            rest_poses.append((joint_info[8] + joint_info[9]) / 2)
        else:
            lower_limits.append(-np.pi)
            upper_limits.append(np.pi)
            joint_ranges.append(2 * np.pi)
            rest_poses.append(0)
    
    return lower_limits, upper_limits, joint_ranges, rest_poses


def rollout_ee_pose_real(robot_id, env, left_ee_pose, right_ee_pose, joints=[0, 1, 2, 3, 4, 5, 10, 11, 12]):
    ts = env.reset()

    qpos_list = []
    image_list = []
    target_qpos_list = []
    #rewards = []

    t0 = time.time()
    rollout_images = []
    ik = AlohaIK()
    xyz = []
    quat = []
    cur_joint = []
    for t, (left_pose, right_pose) in enumerate(zip(left_ee_pose, right_ee_pose)):
        obs = ts.observation
        if "images" in obs:
            image_list.append(obs["images"])
        else:
            image_list.append({"main": obs["image"]})
        qpos = np.array(obs["qpos"])

        # joint_angles = ik.solve(qpos[7:13])
        qpos_left = qpos[:6]
        qpos_right = qpos[7:13] # waist                  shoulder             elbow                   forearm_roll        wrist angle      wrist rotate 

        qpos_left = qpos_left.tolist() + [0, 0.02239, -0.02239]
        qpos_right = qpos_right.tolist() + [0, 0.02239, -0.02239] # ee_gripper   left finger     right finger
        left_current_joint_positions = qpos_left
        right_current_joint_positions = qpos_right
        # print(f"CURRENT: {current_joint_positions}")

        left_current_joint_positions[0] += math.pi/2
        right_current_joint_positions[0] += math.pi/2

        # target_pos = [pose[2], pose[0], pose[1]]
        left_target_pos = left_pose[:3]
        left_target_orientation = left_pose[3:]
        t0 = time.time()
        left_joint_angles = p.calculateInverseKinematics(
            robot_id, 
            endEffectorLinkIndex=12, 
            targetPosition=left_target_pos, 
            targetOrientation=left_target_orientation, 
            currentPositions=left_current_joint_positions,
            maxNumIterations=100,
            residualThreshold=0.001
        )
        right_target_pos = right_pose[:3]
        right_target_orientation = right_pose[3:]
        right_joint_angles = p.calculateInverseKinematics(
            robot_id, 
            endEffectorLinkIndex=12, 
            targetPosition=right_target_pos, 
            targetOrientation=right_target_orientation, 
            currentPositions=right_current_joint_positions,
            maxNumIterations=100,
            residualThreshold=0.001
        )
        t1 = time.time() - t0
        print(f"Time per calcuation: {t1}")
        left_joint_angles = list(left_joint_angles)
        right_joint_angles = list(right_joint_angles)

        left_target_qpos = np.array(left_joint_angles[:7])
        right_target_qpos = np.array(right_joint_angles[:7])
        left_target_qpos[0] -= math.pi/2
        right_target_qpos[0] -= math.pi/2
        # print(f"TARGET: {target_qpos}")

        target_qpos = np.concatenate([left_target_qpos, right_target_qpos])

        # ts = env.step(target_qpos)

        qpos_list.append(qpos)
        target_qpos_list.append(target_qpos)

        # time.sleep(0.02)

    print("moving robot")
    move_grippers([env.follower_bot_right], [FOLLOWER_GRIPPER_JOINT_OPEN] * 2, moving_time=0.5)
    move_grippers([env.follower_bot_left], [FOLLOWER_GRIPPER_JOINT_OPEN] * 2, moving_time=0.5)

    return target_qpos_list


def main(args):
    
    root = args.dataset
    repo_id = "rpuns/test"
    
    urdf_path = os.path.join(
            os.path.dirname(egomimic.__file__), "resources/aloha_vx300s.urdf"
        )   
    robot_id = init_pybullet(urdf_path)
    print_joint_info(robot_id)

    aloha_fk = AlohaFK()
    
    node = create_interbotix_global_node('aloha')
    env = make_real_env(node, active_arms="both", setup_robots=True)
    robot_startup(node)

    if args.debug:
        episodes = [0]
        dataset = RLDBDataset(repo_id=repo_id, root=root, local_files_only=True, episodes=episodes, mode="sample")
        joint_positions = torch.stack([
            sample["actions_joints"][0] for sample in dataset
        ])
        cartesians = torch.stack([
            sample["actions_cartesian"][0] for sample in dataset
        ])
        left_out = aloha_fk.fk(joint_positions[..., :6]).numpy()
        right_out = aloha_fk.fk(joint_positions[..., 7:13]).numpy()
        right_fk_ee_pose = np.zeros((right_out.shape[0], 7))
        left_fk_ee_pose = np.zeros((left_out.shape[0], 7))
        for i, r in enumerate(right_out):
            right_fk_ee_pose[i] = transformation_matrix_to_pose(r)
        for i, l in enumerate(left_out):
            left_fk_ee_pose[i] = transformation_matrix_to_pose(l)

        joint_positions_reconstructed = rollout_ee_pose_real(robot_id, env, left_fk_ee_pose, right_fk_ee_pose)
        ###
        breakpoint()
        joint_positions_reconstructed = np.stack(joint_positions_reconstructed).astype(np.float32)
        right_reconstructed_ee_pose = aloha_fk.fk(joint_positions_reconstructed[..., 7:13])
        left_reconstructed_ee_pose = aloha_fk.fk(joint_positions_reconstructed[..., :6])

        tensor_left_out = torch.from_numpy(left_out).float()
        tensor_right_out = torch.from_numpy(right_out).float()

        left_diff = tensor_left_out - left_reconstructed_ee_pose
        right_diff = tensor_right_out - right_reconstructed_ee_pose

        print(f'Left diff mean: {left_diff.mean(dim=0)}')
        print(f'Left diff std: {left_diff.std(dim=0)}')
        print(f'Right diff mean: {right_diff.mean(dim=0)}')
        print(f'Right diff std: {right_diff.std(dim=0)}')

    p.disconnect()

# def cartesian_joints(actions_cartesians):
#     actions_gripper = actions_cartesian[..., -1].unsqueeze(-1)
#     actions_pos = actions_cartesian[..., :3]
#     actions_ypr = actions_cartesian[..., 3:6]
#     actions_ypr = actions_ypr[..., list(permutation)]
#     actions_rotmat = batched_euler_to_rot_matrix(actions_ypr)
#     extrinsics = torch.from_numpy(self.model.model.camera_transforms.extrinsics[arm]).to(actions_cartesian.device).float()
#     batch_shape = actions_cartesian.shape[:-1]
#     T = torch.zeros(*batch_shape, 4, 4, device=actions_cartesian.device)
#     T[..., :3, :3] = actions_rotmat
#     T[..., :3, 3] = actions_pos
#     T[..., 3, 3] = 1.0

#     T_base = T
#     T_base = extrinsics @ T # Should we multiply from base frame to cam frame?
    
#     # actions_rotmat = T_base[..., :3, :3]
#     # wxyz -> xyzw

#     # actions_quat = matrix_to_quaternion(actions_rotmat)
#     # actions_quat = torch.cat([actions_quat[..., 1:], actions_quat[..., :1]], dim=-1)

#     if T_base.dim() == 3:
#         T_base = T_base.unsqueeze(1) # add the T dim
#     if actions_gripper.dim() == 2:
#         actions_gripper = actions_gripper.unsqueeze(0)
#     actions = []
#     for b in range(T_base.shape[0]):
#         actions.append(transformation_matrix_to_pose(T_base[b].cpu().numpy()))

#     actions = np.stack(actions)

#     actions_pos = actions[..., :3]
#     actions_quat = actions[..., 3:]
#     target_joint_positions_l = []
#     for i in range(batch_shape[0]):
#         target_joint_positions = self.ik.solve(target_pos=actions_pos[i], 
#                                             target_orientation=actions_quat[i], 
#                                             current_joints=current_joint_positions)
#         target_joint_positions_l.append(target_joint_positions)
#     actions_joint_positions = np.stack(target_joint_positions_l)
#     actions_joint_positions = np.concatenate((actions_joint_positions, actions_gripper.cpu().numpy()), axis=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action='store_true',
        help="In debug mode, only one demo is rolled out",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="path to dataset",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="path to robot model"
    )
    
    args = parser.parse_args()
    main(args)

