from typing import Any, Dict, List, Optional, Tuple

import os
import time
import itertools
import copy
import numpy as np

import hydra
import torch
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer

from omegaconf import DictConfig, OmegaConf

from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras, task_wrapper
from egomimic.utils.egomimicUtils import AlohaFK, transformation_matrix_to_6dof, AlohaIK

from egomimic.pl_utils.pl_model import ModelWrapper

from egomimic.scripts.evaluation.eval import Eval

from egomimic.scripts.evaluation.utils import TemporalAgg, save_image, transformation_matrix_to_pose, batched_euler_to_rot_matrix

log = RankedLogger(__name__, rank_zero_only=True)

from rldb.utils import *

@task_wrapper
def eval(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    log.info(f"Instantiating data schematic <{cfg.multirun_cfg.data_schematic._target_}>")
    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.multirun_cfg.data_schematic)
    datamodule = None
    if cfg.datasets is not None:
        
        if cfg.datasets == "multirun":
            log.info(f"Using multirun validation datasets")
            eval_datasets = cfg.multirun_cfg.data.valid_datasets
            datasets_target = cfg.multirun_cfg.data._target_
        elif "eval_datasets" in cfg.datasets and cfg.datasets.eval_datasets is not None:
            log.info(f"Using specified yaml evaluation datasets")
            eval_datasets = cfg.datasets.data.eval_datasets
            datasets_target = cfg.datasets.data._target_
        elif "valid_datasets" in cfg.datasets and cfg.datasets.valid_datasets is not None:
            log.ingo(f"Using specified yaml validation datasets")
            eval_datasets = cfg.datasets.data.valid_datasets
            datasets_target = cfg.datasets.data._target_
        
        eval_datasets_dict = {}
        for dataset_name in eval_datasets:
            eval_datasets[dataset_name] = hydra.utils.instantiate(
                eval_datasets_dict[dataset_name]
            )
    
        log.info(f"Instantiating datamodule <{datasets_target}>")
        assert "MultiDataModuleWrapper" in datasets_target, "cfg.data._target_ must be 'MultiDataModuleWrapper'"
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, valid_datasets=eval_datasets_dict)
    
        for dataset_name, dataset in datamodule.valid_datasets.items():
            log.info(f"Inferring shapes for dataset <{dataset_name}>")
            data_schematic.infer_shapes_from_batch(dataset[0])
            data_schematic.infer_norm_from_dataset(dataset)
    
    eval = hydra.utils.instantiate(cfg.eval)
    eval.datamodule = datamodule
    eval.data_schematic = data_schematic # unsure if this is necessary to pass in
    dataset_path = '/home/rl2-eve/EgoVerse/logs/lerobot/lerobot'
    repo_id = "rpuns/test"
    episodes = [0]  
    dataset = RLDBDataset(repo_id=repo_id, root=dataset_path, local_files_only=True, episodes=episodes, mode="sample")
    joints_act = torch.stack([
            sample["actions_joints"][0] for sample in dataset
    ])
    cartesians_act = torch.stack([
        sample["actions_cartesian"][0] for sample in dataset
    ])
    fk = AlohaFK()
    ik = AlohaIK()
    left_fk = fk.fk(joints_act[:, :6])
    right_fk = fk.fk(joints_act[:, 7: 13])

    perms = list(itertools.permutations((0, 1, 2, 3)))
    results = []  # (perm, rms_err, per_joint_mean, elapsed)

    for perm in perms:
        solved_actions = []
        t0 = time.time()

        for i in range(cartesians_act.shape[0]):
            qpos = joints_act[i]

            act_left = transformation_matrix_to_pose(left_fk[i].cpu().numpy())
            act_right = transformation_matrix_to_pose(right_fk[i].cpu().numpy())

            act_left_xyz = act_left[:3]
            act_left_quat = act_left[3:]
            act_left_quat = act_left_quat[list(perm)]

            act_right_xyz = act_right[:3]
            act_right_quat = act_right[3:]
            act_right_quat = act_right_quat[list(perm)]
            sol_left = ik.solve(act_left_xyz[np.newaxis, :], act_left_quat[np.newaxis, :], qpos[:7])
            sol_right = ik.solve(act_right_xyz[np.newaxis, :], act_right_quat[np.newaxis, :], qpos[7:])
            sol_left = np.concatenate([sol_left.squeeze(0), [qpos[6]]])
            sol_right = np.concatenate([sol_right.squeeze(0), [qpos[13]]])

            solved_actions.append(np.concatenate([sol_left, sol_right], axis=-1))

        elapsed = time.time() - t0
        solved_actions = np.asarray(solved_actions).reshape(len(solved_actions), -1)

        diff = solved_actions - joints_act.cpu().numpy()
        rms_err = np.sqrt((diff ** 2).mean())          # scalar
        per_joint_mean = np.abs(diff)[:100,].mean(axis=0)             # 14‑element vector

        results.append((perm, rms_err, per_joint_mean, elapsed))

        # sort best→worst by RMS error
    results.sort(key=lambda x: x[1])

    print("Permutation ranking (best to worst):")
    for rank, (perm, rms, pj_mean, t) in enumerate(results, 1):
        print(f"{rank:>2}. perm={perm}  RMS={rms:.6f}  time={t:.3f}s")
        print("    per‑joint mean diff:", np.array2string(pj_mean, precision=4, separator=', '))
        
    best_perm = results[0][0]
    print("\nBest permutation:", best_perm)
    breakpoint()

@hydra.main(version_base="1.3", config_path="../../hydra_configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    
    extras(cfg)

    # if 'multirun_path' not in cfg:
    #     raise ValueError("Multirun path is required.")
    # if not os.path.exists(cfg.multirun_path):
    #     raise FileNotFoundError(f"Cannot locate multirun.yaml at {cfg.multirun_path}")
    if 'multirun_cfg' in cfg:
        multi_cfg = OmegaConf.load(cfg.multirun_cfg)
        OmegaConf.set_struct(cfg, False)
        cfg["multirun_cfg"] = copy.deepcopy(multi_cfg)
        OmegaConf.set_struct(cfg, True)
    
    print(OmegaConf.to_yaml(cfg))
    
    eval(cfg)

if __name__ == '__main__':
    main()from typing import Any, Dict, List, Optional, Tuple

import os
import time
import itertools
import copy
import numpy as np

import hydra
import torch
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer

from omegaconf import DictConfig, OmegaConf

from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras, task_wrapper
from egomimic.utils.egomimicUtils import AlohaFK, transformation_matrix_to_6dof, AlohaIK

from egomimic.pl_utils.pl_model import ModelWrapper

from egomimic.scripts.evaluation.eval import Eval

from egomimic.scripts.evaluation.utils import TemporalAgg, save_image, transformation_matrix_to_pose, batched_euler_to_rot_matrix

log = RankedLogger(__name__, rank_zero_only=True)

from rldb.utils import *

@task_wrapper
def eval(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    log.info(f"Instantiating data schematic <{cfg.multirun_cfg.data_schematic._target_}>")
    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.multirun_cfg.data_schematic)
    datamodule = None
    if cfg.datasets is not None:
        
        if cfg.datasets == "multirun":
            log.info(f"Using multirun validation datasets")
            eval_datasets = cfg.multirun_cfg.data.valid_datasets
            datasets_target = cfg.multirun_cfg.data._target_
        elif "eval_datasets" in cfg.datasets and cfg.datasets.eval_datasets is not None:
            log.info(f"Using specified yaml evaluation datasets")
            eval_datasets = cfg.datasets.data.eval_datasets
            datasets_target = cfg.datasets.data._target_
        elif "valid_datasets" in cfg.datasets and cfg.datasets.valid_datasets is not None:
            log.ingo(f"Using specified yaml validation datasets")
            eval_datasets = cfg.datasets.data.valid_datasets
            datasets_target = cfg.datasets.data._target_
        
        eval_datasets_dict = {}
        for dataset_name in eval_datasets:
            eval_datasets[dataset_name] = hydra.utils.instantiate(
                eval_datasets_dict[dataset_name]
            )
    
        log.info(f"Instantiating datamodule <{datasets_target}>")
        assert "MultiDataModuleWrapper" in datasets_target, "cfg.data._target_ must be 'MultiDataModuleWrapper'"
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, valid_datasets=eval_datasets_dict)
    
        for dataset_name, dataset in datamodule.valid_datasets.items():
            log.info(f"Inferring shapes for dataset <{dataset_name}>")
            data_schematic.infer_shapes_from_batch(dataset[0])
            data_schematic.infer_norm_from_dataset(dataset)
    
    eval = hydra.utils.instantiate(cfg.eval)
    eval.datamodule = datamodule
    eval.data_schematic = data_schematic # unsure if this is necessary to pass in
    dataset_path = '/home/rl2-eve/EgoVerse/logs/lerobot/lerobot'
    repo_id = "rpuns/test"
    episodes = [0]  
    dataset = RLDBDataset(repo_id=repo_id, root=dataset_path, local_files_only=True, episodes=episodes, mode="sample")
    joints_act = torch.stack([
            sample["actions_joints"][0] for sample in dataset
    ])
    cartesians_act = torch.stack([
        sample["actions_cartesian"][0] for sample in dataset
    ])
    fk = AlohaFK()
    ik = AlohaIK()
    left_fk = fk.fk(joints_act[:, :6])
    right_fk = fk.fk(joints_act[:, 7: 13])

    perms = list(itertools.permutations((0, 1, 2, 3)))
    results = []  # (perm, rms_err, per_joint_mean, elapsed)

    for perm in perms:
        solved_actions = []
        t0 = time.time()

        for i in range(cartesians_act.shape[0]):
            qpos = joints_act[i]

            act_left = transformation_matrix_to_pose(left_fk[i].cpu().numpy())
            act_right = transformation_matrix_to_pose(right_fk[i].cpu().numpy())

            act_left_xyz = act_left[:3]
            act_left_quat = act_left[3:]
            act_left_quat = act_left_quat[list(perm)]

            act_right_xyz = act_right[:3]
            act_right_quat = act_right[3:]
            act_right_quat = act_right_quat[list(perm)]
            sol_left = ik.solve(act_left_xyz[np.newaxis, :], act_left_quat[np.newaxis, :], qpos[:7])
            sol_right = ik.solve(act_right_xyz[np.newaxis, :], act_right_quat[np.newaxis, :], qpos[7:])
            sol_left = np.concatenate([sol_left.squeeze(0), [qpos[6]]])
            sol_right = np.concatenate([sol_right.squeeze(0), [qpos[13]]])

            solved_actions.append(np.concatenate([sol_left, sol_right], axis=-1))

        elapsed = time.time() - t0
        solved_actions = np.asarray(solved_actions).reshape(len(solved_actions), -1)

        diff = solved_actions - joints_act.cpu().numpy()
        rms_err = np.sqrt((diff ** 2).mean())          # scalar
        per_joint_mean = np.abs(diff)[:100,].mean(axis=0)             # 14‑element vector

        results.append((perm, rms_err, per_joint_mean, elapsed))

        # sort best→worst by RMS error
    results.sort(key=lambda x: x[1])

    print("Permutation ranking (best to worst):")
    for rank, (perm, rms, pj_mean, t) in enumerate(results, 1):
        print(f"{rank:>2}. perm={perm}  RMS={rms:.6f}  time={t:.3f}s")
        print("    per‑joint mean diff:", np.array2string(pj_mean, precision=4, separator=', '))
        
    best_perm = results[0][0]
    print("\nBest permutation:", best_perm)
    breakpoint()

@hydra.main(version_base="1.3", config_path="../../hydra_configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    
    extras(cfg)

    # if 'multirun_path' not in cfg:
    #     raise ValueError("Multirun path is required.")
    # if not os.path.exists(cfg.multirun_path):
    #     raise FileNotFoundError(f"Cannot locate multirun.yaml at {cfg.multirun_path}")
    if 'multirun_cfg' in cfg:
        multi_cfg = OmegaConf.load(cfg.multirun_cfg)
        OmegaConf.set_struct(cfg, False)
        cfg["multirun_cfg"] = copy.deepcopy(multi_cfg)
        OmegaConf.set_struct(cfg, True)
    
    print(OmegaConf.to_yaml(cfg))
    
    eval(cfg)

if __name__ == '__main__':
    main()