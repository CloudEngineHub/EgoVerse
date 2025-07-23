from typing import Any, Dict, List, Optional, Tuple

import os
import time
import itertools
import copy
import math
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
    right_fk_ee_pose = np.zeros((right_fk.shape[0], 7))
    left_fk_ee_pose= np.zeros((left_fk.shape[0], 7))
    for i, r in enumerate(right_fk):
        right_fk_ee_pose[i] = transformation_matrix_to_pose(r)
    for i, l in enumerate(left_fk):
        left_fk_ee_pose[i] = transformation_matrix_to_pose(l)
    # cartesians_act = []
    # for i in range(left_fk.shape[0]):
    #     left_pose = transformation_matrix_to_6dof(left_fk[i].cpu().numpy())
    #     right_pose = transformation_matrix_to_6dof(right_fk[i].cpu().numpy())
        # both_pose = torch.from_numpy(np.concatenate([left_pose,[joints_act[i][6]], right_pose, [joints_act[i][13]]], axis=0))
    #     cartesians_act.append(both_pose)
    # cartesians_act = torch.stack(cartesians_act)

    perms = list(itertools.permutations((0, 1, 2, 3)))
    results = []  # (perm, rms_err, per_joint_mean, elapsed)

    for perm in [perms[0]]:
        solved_actions = []
        t0 = time.time()
        left_xyz = []
        left_quat = []
        left_cur_joints = []
        for i in range(cartesians_act.shape[0]):
            # qpos = joints_act[i].to(eval.model.device)
    #         qpos = np.array([0.00000000e+00,
    #    -9.41864252e-01,  1.11520410e+00,  1.53398083e-03, -3.11398119e-01,
    #    -1.07378662e-02, -3.08226784e-01, 0.00000000e+00,
    #    -9.41864252e-01,  1.11520410e+00,  1.53398083e-03, -3.11398119e-01,
    #    -1.07378662e-02, -3.08226784e-01])
            load_np = np.load('qpos.npy')
            qpos = load_np[0, 7:]
            # qpos = np.concatenate([load_np[0, 7:], load_np[0, 7:]], axis=0)

            # act_left = transformation_matrix_to_pose(left_fk[i].cpu().numpy())
            # act_right = transformation_matrix_to_pose(right_fk[i].cpu().numpy())
            act_left = left_fk_ee_pose[i]
            act_right = right_fk_ee_pose[i]
            act_left_xyz = act_left[:3]
            act_left_quat = act_left[3:]
            # act_left_quat = act_left_quat[list(perm)]

            act_right_xyz = act_right[:3]
            act_right_quat = act_right[3:]
            # act_right_quat = act_right_quat[list(perm)]
            current_joints = qpos[:6].tolist() + [0, 0.02239, -0.02239]
            current_joints[0] += math.pi/2
            sol_left = ik.solve(act_left_xyz[np.newaxis, :], act_left_quat[np.newaxis, :], qpos[:7])
            sol_right = ik.solve(act_right_xyz[np.newaxis, :], act_right_quat[np.newaxis, :], qpos[:7])
            sol_left = np.concatenate([sol_left.squeeze(0), [joints_act[i,6]]])
            sol_right = np.concatenate([sol_right.squeeze(0), [joints_act[i, 13]]])
            left_xyz.append(act_left_xyz)
            left_quat.append(act_left_quat)
            left_cur_joints.append(current_joints)
            # act_left  = cartesians_act[i, :7].unsqueeze(0).to(eval.model.device)
            # act_right = cartesians_act[i, 7:].unsqueeze(0).to(eval.model.device)
            # cur_left, cur_right = qpos[:7], qpos[7:]

            # sol_left  = eval.solve_ik(act_left,  cur_left,  permutation=perm, arm="left")
            # sol_right = eval.solve_ik(act_right, cur_right, permutation=perm, arm="right")
            # sol_left = ik.solve(act_left[:3], act_left[3:6], cur_left)
            solved_actions.append(np.concatenate([sol_left, sol_right], axis=-1))
        left_xyz = np.array(left_xyz)
        left_quat = np.array(left_quat)
        current_joints = np.array(current_joints)
        breakpoint()
        elapsed = time.time() - t0
        left_pos_errs, left_ang_errs = [], []
        right_pos_errs, right_ang_errs = [], []

        for i, sol in enumerate(solved_actions):
            # original FK 4×4
            orig_L = left_fk[i].cpu().numpy()
            orig_R = right_fk[i].cpu().numpy()
            # reconstructed FK from your IK solution
            recon_L = fk.fk(torch.from_numpy(sol[:6]).unsqueeze(0).float()).cpu().numpy()[0]
            recon_R = fk.fk(torch.from_numpy(sol[7:13]).unsqueeze(0).float()).cpu().numpy()[0]

            # convert to [x,y,z,qx,qy,qz,qw]
            pL, qL = transformation_matrix_to_pose(orig_L)[:3], transformation_matrix_to_pose(orig_L)[3:]
            pLr, qLr = transformation_matrix_to_pose(recon_L)[:3], transformation_matrix_to_pose(recon_L)[3:]
            pR, qR = transformation_matrix_to_pose(orig_R)[:3], transformation_matrix_to_pose(orig_R)[3:]
            pRr, qRr = transformation_matrix_to_pose(recon_R)[:3], transformation_matrix_to_pose(recon_R)[3:]

            # position error (Euclidean)
            left_pos_errs .append(np.linalg.norm(pL  - pLr))
            right_pos_errs.append(np.linalg.norm(pR  - pRr))

            # orientation error (angle between quaternions)
            def ang_err(a, b):
                a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
                return 2 * np.arccos(np.clip(abs(np.dot(a, b)), -1, 1))
            left_ang_errs .append(ang_err(qL,  qLr))
            right_ang_errs.append(ang_err(qR,  qRr))

        # aggregate
        mean_L_pos = np.mean(left_pos_errs)
        mean_L_ang = np.mean(left_ang_errs)
        mean_R_pos = np.mean(right_pos_errs)
        mean_R_ang = np.mean(right_ang_errs)

        # include in your results tuple or just print:
        print(f"EE errors — left: pos {mean_L_pos:.4f} m, ang {mean_L_ang:.4f} rad; "
            f"right: pos {mean_R_pos:.4f} m, ang {mean_R_ang:.4f} rad")
        # sort best→worst by RMS error
    # results.sort(key=lambda x: x[1])

    # print("Permutation ranking (best to worst):")
    # for rank, (perm, rms, pj_mean, t) in enumerate(results, 1):
    #     print(f"{rank:>2}. perm={perm}  RMS={rms:.6f}  time={t:.3f}s")
    #     print("    per‑joint mean diff:", np.array2string(pj_mean, precision=4, separator=', '))
        
    # best_perm = results[0][0]
    # print("\nBest permutation:", best_perm)
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