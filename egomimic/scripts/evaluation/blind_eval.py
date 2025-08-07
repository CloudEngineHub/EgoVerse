from sre_constants import JUMP
import pandas as pd
import argparse
import numpy as np
import time
import os
import string
import random
from pathlib import Path

import gc
import torch
from torchvision.utils import save_image
import cv2
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

from egomimic.scripts.evaluation.eval import Eval

class BlindEval(Eval):
    def __init__(
        self,
        eval_path,
        models,
        **kwargs
    ):
        super().__init__(eval_path)
        self.models = models
        self.blind_eval_name = kwargs.get("blind_eval_name", None)
        self.blind_eval_path = kwargs.get("blind_eval_path", None)
        
        self.cur_ckpt = None
        self.cur_rollout_id = None
        self.cur_blind_id = None

        if self.models is not None and self.blind_eval_path is not None:
            raise ValueError("Only models or blind_eval_path should be specified")
        if self.models is None and self.blind_eval_path is None:
            raise ValueError("One of models or blind eval path needs to be specified")
        
        self.properties = ['ckpt_path', 'arm', 'frequency', 'cartesian']
        eval_dir = Path(self.eval_path).parent
        self.blind_eval_path = os.path.join(eval_dir, self.blind_eval_name)
        os.makedirs(self.blind_eval_path, exist_ok=True)

        if self.models:
            #instantiate models
            self.models_dict = {}
            for model_name in self.models:
                breakpoint()
                self.models_dict[key] = hydra.utils.instantiate(self.models[model_name])
            breakpoint()
            rows = []
            num_models = len(self.models)
            letters = list(string.ascii_uppercase[:num_models])
            random.shuffle(letters)
            for i, (key, value) in enumerate(self.models.items()):
                row = [value[prop] for prop in self.properties] + [letters[i]]
                rows.append(row)
            self.properties.append('blind_id')
            self.model_df = pd.DataFrame(rows, columns=self.properties)
            self.model_df.to_pickle(os.path.join(self.blind_eval_path, 'info.pkl'))
            self.result_df = pd.DataFrame([], columns=['episode_id', 'blind_id', 'success'])
            self.result_df.to_csv(os.path.join(self.blind_eval_path, 'results.csv'))
            breakpoint()
        else:
            self.df = pd.read_pickle(os.path.join(self.blind_eval_path, 'info.pkl'))
            self.result_df = pd.read_csv(os.path.join(self.blind_eval_path, 'results.csv')) 
    
    def ckpt_from_blind_id(self, blind_id):
        row = self.model_df[self.model_df['blind_id'] == blind_id]
        return row['ckpt_path'].iloc[0]
    
    def blind_ids(self):
        return self.model_df["blind_id"].unique()

    def select_models_episode(self):
        options_list = self.df['name']
        blind_id_valid = False
        while not blind_id_valid:
            blind_id = input('Choose models to eval' + ', '.join(options_list))
        if blind_id not in self.model_df.columns.unique():
            print("Invalid model inputted")
        else:
            blind_id_valid = True
        self.cur_blind_id = blind_id
        ckpt = self.ckpt_from_blind_id(blind_id)
        if ckpt != self.cur_ckpt:
            if self.model is not None:
                del self.model
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print("New cuda memory being cleaned: " + torch.cuda.memory_allocated())

            self.model = ModelWrapper.load_from_checkpoint(ckpt)
            self.cur_ckpt = ckpt
        
        rollout_id_valid = False
        added_string = f'prev id was {self.cur_rollout_id}' if self.cur_rollout_id else ''
        while not rollout_id_valid:
            rollout_id = input(f'Choose episode id {added_string}: ')
            if rollout_id.isdigit() and int(rollout_id) > 0:
                self.cur_rollout_id = int(rollout_id)
    
    def run_eval(self):
        aloha_fk = AlohaFK()
        qpos_t, actions_t = [], []

        if TEMPORAL_AGG:
            TA = TemporalAgg()

        ts = self.env.reset()
        t0 = time.time()

        while True:
            keep_rollout = True
            self.select_models_episode()
            print("Can interrupt rollout with any keyboard keys")
            
        return
        
            
        