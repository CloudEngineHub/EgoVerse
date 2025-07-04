import argparse
import numpy as np
import time
import os

import torch
import robomimic.utils.obs_utils as ObsUtils
from torchvision.utils import save_image
import cv2
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)

from eve.constants import DT, FOLLOWER_GRIPPER_JOINT_OPEN, START_ARM_POSE

from egomimic.utils.egomimicUtils import (
    cam_frame_to_cam_pixels,
    draw_dot_on_frame,
    general_unnorm,
    miniviewer,
    nds,
    ARIA_INTRINSICS,
    EXTRINSICS,
    ee_pose_to_cam_frame,
    AlohaFK,
)

from eve.robot_utils import move_grippers, move_arms  # requires EgoMimic-eve
from eve.real_env import make_real_env  # requires EgoMimic-eve

from egomimic.utils.realUtils import *

from omegaconf import DictConfig, OmegaConf
import hydra

CURR_INTRINSICS = ARIA_INTRINSICS
CURR_EXTRINSICS = EXTRINSICS["ariaJul29R"]
TEMPORAL_AGG = False

from rldb.utils import EMBODIMENT, get_embodiment, get_embodiment_id

from egomimic.pl_utils.pl_model import ModelWrapper

from egomimic.scripts.evaluation.eval import Eval
from egomimic.scripts.evaluation.utils import TemporalAgg

from egomimic.utils.pylogger import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)

class (Eval):
    def __init__(
        self,
        eval_path,
        ckpt_path,
        arm,
        query_frequency,
        num_rollouts,
        debug=False,
        **kwargs
    ):
        super().__init__(eval_path, debug, **kwargs)

        log.info(f"Instantiating model from checkpoint<{ckpt_path}>")
        self.model = ModelWrapper.load_from_checkpoint(ckpt_path)
        self.arm = arm
        self.query_frequency = query_frequency
        self.num_rollouts = num_rollouts
        
        self.plot_pred_freq = kwargs.get("plot_pred_freq", None)
        if self.data_schematic is None:
            raise ValueError("Data schematic is needed to be passed in.")
        self.model.eval()
        node = create_interbotix_global_node('aloha')
        self.env = make_real_env(node, active_arms=self.arm, setup_robots=True)

        robot_startup(node)

        if not os.path.exists(self.eval_path) and self.debug:
            os.mkdir(self.eval_path)

        if self.arm == "right":
            self.embodiment_name = "eve_right_arm"
            self.embodiment_id = get_embodiment_id(self.embodiment_name)
        elif self.arm == "left":
            self.embodiment_name = "eve_left_arm"
            self.embodiment_id = get_embodiment_id(self.embodiment_name)
        elif self.arm == "both":
            self.embodiment_name = "eve_bimanual"
            self.embodiment_id = get_embodiment_id(self.embodiment_name)
        else:
            raise ValueError("Invalid arm inputted")

    def process_batch_for_eval(self, batch):
        obs = batch
        processed_batch = {}
        qpos = np.array(obs["qpos"])
        qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = {
            "front_img_1" : (
                torch.from_numpy(
                obs["images"]["cam_high"][None, None, :]
            )).to(torch.uint8) / 255.0,
        }

        if self.arm == "right":
            data["right_wrist_img"] = torch.from_numpy(obs["images"]["cam_right_wrist"][None, None, :]).to(torch.uint8) / 255.0
            data["joint_positions"] =  qpos[..., 7:].reshape((1, 1, -1))
            data["embodiment"] = torch.Tensor([self.embodiment_id], dtype=torch.int64)
            processed_batch[self.embodiment_id] = data
            processed_batch = self.model.model.data_schematic.normalize_data(processed_batch, self.embodiment_id)
        elif self.arm == "left":
            data["left_wrist_img"] = torch.from_numpy(obs["images"]["cam_left_wrist"][None, None, :]).to(torch.uint8) / 255.0
            data["joint_positions"] = qpos[..., :7]
            data["embodiment"] = torch.Tensor([self.embodiment_id], dtype=torch.int64)
            processed_batch[self.embodiment_id] = data
            processed_batch = self.model.model.data_schematic.normalize_data(processed_batch, self.embodiment_id)
        elif self.arm == "both":
            data["right_wrist_img"] = torch.from_numpy(obs["images"]["cam_right_wrist"][None, None, :]).to(torch.uint8) / 255.0
            data["left_wrist_img"] = torch.from_numpy(obs["images"]["cam_left_wrist"][None, None, :]).to(torch.uint8) / 255.0
            data["joint_positions"] = qpos[..., :].reshape((1, 1, -1))
            data["embodiment"] = torch.Tensor([self.embodiment_id], dtype=torch.int64)
            processed_batch[self.embodiment_id] = data
            processed_batch = self.model.model.data_schematic.normalize_data(processed_batch, self.embodiment_id)

        return processed_batch
    
    def run_eval(self):
        device = torch.device("cuda")
        aloha_fk = AlohaFK()
        qpos_t, actions_t = [], []

        if TEMPORAL_AGG:
            TA = TemporalAgg()

        ts = self.env.reset()
        t0 = time.time()

        for rollout_id in self.num_rollouts:
            with torch.inference_mode():
                rollout_images = []
                for t in range(1000):
                    time.sleep(max(0, DT*2 - (time.time() - t0)))
                    t0 = time.time()
                    obs = ts.observation
                    inference_t = time.time()

                    if t % self.query_frequency == 0:
                        batch = self.process_batch_for_eval(obs)
                        preds = self.model.model.forward_eval(batch)
                        
                        ac_key = self.model.model.ac_keys[self.embodiment_id]
                        actions = preds[f"{self.embodiment_name}_{ac_key}"].cpu().numpy()

                        if TEMPORAL_AGG:
                            TA.add_action(actions[0])
                            actions = TA.smoothed_action()[None, :]

                        print(f"Inference time: {time.time() - inference_t}")

                    raw_action = actions[:, t % self.query_frequency]
                    raw_action = raw_action[0]
                    target_qpos = raw_action

                    if self.arm == "right":
                        target_qpos = np.concatenate([np.zeros(7), target_qpos])
                    
                    ts = self.env.step(target_qpos)
                    qpos_t.append(ts.observation["qpos"])
                    actions_t.append(target_qpos)

            rollout_images = []
            log.info("Moving Robot")

            if self.arm == "right":
                move_grippers(
                [self.env.follower_bot_right], [FOLLOWER_GRIPPER_JOINT_OPEN], moving_time=0.5
                )  # open
                move_arms([self.env.follower_bot_right], [START_ARM_POSE[:6]], moving_time=1.0)
            elif self.arm == "left":
                move_grippers(
                [self.env.follower_bot_left], [FOLLOWER_GRIPPER_JOINT_OPEN], moving_time=0.5
                ) 
                move_arms([self.env.follower_bot_left], [START_ARM_POSE[:6]], moving_time=1.0)
            elif self.arm == "both":
                move_grippers(
                    [self.env.follower_bot_left, self.env.follower_bot_right], [FOLLOWER_GRIPPER_JOINT_OPEN]*2, moving_time=0.5
                )  # open
                move_arms([self.env.follower_bot_left, self.env.follower_bot_right], [START_ARM_POSE[:6]]*2, moving_time=1.0)
        return
        
