from collections import OrderedDict
import logging

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from overrides import override
from omegaconf import DictConfig, OmegaConf
import importlib

from egomimic.algo.algo import Algo
from egomimic.rldb.utils import get_embodiment_id, get_embodiment

from cosmos_policy.models.policy_video2world_model import (
    CosmosPolicyVideo2WorldModel,
    CosmosPolicyVideo2WorldConfig,
)
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE
from cosmos_policy.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from cosmos_policy.config.conditioner.video2world_conditioner import Video2WorldConditioner
from cosmos_policy._src.predict2.conditioner import TextAttr
from cosmos_policy._src.predict2.models.text2world_model import EMAConfig
from cosmos_policy._src.predict2.networks.wan2pt1 import WanModel
from cosmos_policy._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L

logger = logging.getLogger(__name__)


class CosmosPolicy(Algo):
    def __init__(
        self,
        data_schematic,
        camera_transforms,
        domains,
        ac_keys,
        # ---------------------------
        # Image augmentations
        # ---------------------------
        train_image_augs,
        eval_image_augs,
        # ---------------------------
        # Cosmos Policy specific params
        # ---------------------------
        model_config,
        chunk_size: int = 8,
        use_proprio: bool = True,
        use_wrist_images: bool = True,
        use_third_person_images: bool = True,
        num_history_frames: int = 1,
        final_image_size: int = 224,
        **kwargs
    ):
        super().__init__()
        
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic
        self.camera_transforms = camera_transforms
        self.train_image_augs = train_image_augs
        self.eval_image_augs = eval_image_augs
        self.ac_keys = ac_keys
        self.domains = domains
        self.device = None
        
        # Cosmos Policy specific parameters
        self.chunk_size = chunk_size
        self.use_proprio = use_proprio
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.num_history_frames = num_history_frames
        self.final_image_size = final_image_size
        self.model_config = model_config
        
        # Initialize camera, proprio, and language keys per embodiment
        self.camera_keys = {}
        self.proprio_keys = {}
        self.lang_keys = {}
        
        for embodiment in self.domains:
            embodiment_id = get_embodiment_id(embodiment)
            self.camera_keys[embodiment_id] = []
            self.proprio_keys[embodiment_id] = []
            self.lang_keys[embodiment_id] = []
            for key in data_schematic.keys_of_type("action_keys"):
                if data_schematic.is_key_with_embodiment(key, embodiment_id) and key == self.ac_keys[embodiment]:
                    self.ac_keys[embodiment_id] = key
            for key in data_schematic.keys_of_type("camera_keys"):
                if data_schematic.is_key_with_embodiment(key, embodiment_id):
                    self.camera_keys[embodiment_id].append(key)
            for key in data_schematic.keys_of_type("proprio_keys"):
                if data_schematic.is_key_with_embodiment(key, embodiment_id):
                    self.proprio_keys[embodiment_id].append(key)
            for key in data_schematic.keys_of_type("lang_keys"):
                if data_schematic.is_key_with_embodiment(key, embodiment_id):
                    self.lang_keys[embodiment_id].append(key)
        
        # Cosmos Policy model initialization
        cosmos_config = self._get_cosmos_policy_config()
        self.model = CosmosPolicyVideo2WorldModel(config=cosmos_config)
        self.nets["policy"] = self.model
        self.model_config = self.model.config if hasattr(self.model, 'config') else None
    
    
    def _initialize_model(self, device):
        """Move the cosmos_policy model to the given device."""
        # Model is already initialized in __init__, just move to device
        if self.model is not None:
            self.model = self.model.to(device)
            self.device = device
    
    def _get_cosmos_policy_config(self):
        """
        Create a CosmosPolicyVideo2WorldConfig object from the model_config.
        
        This method handles the conversion of nested config dictionaries to their proper types:
        - sde: dict -> L(HybridEDMSDE)(...) LazyCall
        - tokenizer: dict -> L(Wan2pt1VAEInterface)(...) LazyCall
        - conditioner: dict -> L(Video2WorldConditioner)(...) LazyCall (with nested text -> L(TextAttr)(...))
        - ema: dict -> EMAConfig object
        - net: dict -> L(WanModel)(...) LazyCall (with nested sac_config -> L(SACConfig)(...))
        
        Returns:
            CosmosPolicyVideo2WorldConfig: Properly configured config object with all nested
                configs converted to the correct types
        """
        model_config_dict = OmegaConf.to_container(self.model_config, resolve=True)
        
        # Convert SDE config dict to LazyCall (required by config class)
        if "sde" in model_config_dict:
            sde_config = model_config_dict["sde"]
            model_config_dict["sde"] = L(HybridEDMSDE)(**sde_config)
        
        # Convert tokenizer config dict to LazyCall (required by config class - expects LazyDict)
        if "tokenizer" in model_config_dict:
            tokenizer_config = model_config_dict["tokenizer"]
            model_config_dict["tokenizer"] = L(Wan2pt1VAEInterface)(**tokenizer_config)
        
        # Convert conditioner config dict to LazyCall (handle nested text config)
        if "conditioner" in model_config_dict:
            conditioner_config = model_config_dict["conditioner"]
            if isinstance(conditioner_config, dict):
                processed_conditioner = {}
                for key, value in conditioner_config.items():
                    if key == "text":
                        text_config = {"input_key": ["t5_text_embeddings"], "use_empty_string": False}
                        text_config.update(value)  # Override with provided values
                        processed_conditioner[key] = L(TextAttr)(**text_config)
                    else:
                        processed_conditioner[key] = value
                model_config_dict["conditioner"] = L(Video2WorldConditioner)(**processed_conditioner)
        
        # Convert ema config dict to EMAConfig object (required by config class)
        if "ema" in model_config_dict:
            ema_config = model_config_dict["ema"]
            model_config_dict["ema"] = EMAConfig(**ema_config)
        
        # Convert net config dict to LazyCall (required by config class - expects LazyDict)
        if "net" in model_config_dict:
            net_config = model_config_dict["net"]
            if isinstance(net_config, dict):
                processed_net = {}
                for key, value in net_config.items():
                    if key == "sac_config":
                        processed_net[key] = L(SACConfig)(**value)
                    else:
                        processed_net[key] = value
                model_config_dict["net"] = L(WanModel)(**processed_net)
        
        # Filter out None values to use config class defaults
        filtered_config = {k: v for k, v in model_config_dict.items() if v is not None}
        cosmos_config = CosmosPolicyVideo2WorldConfig(**filtered_config)
        
        return cosmos_config
    
    
    @override
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            batch (dict): processed dict of batches in cosmos_policy format
        """
        pass
    
    def _robomimic_to_cosmos_policy_data(
        self, batch, cam_keys, proprio_keys, lang_keys, ac_key, embodiment
    ):
        pass
    
    @override
    def forward_training(self, batch):
        """
        One iteration of training. Compute forward pass and compute losses.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D), loss_key_name: torch.Tensor (1)}
        """
        if self.device is None:
            # Set device from batch
            first_tensor = next(iter(batch.values())) if isinstance(batch, dict) else None
            if first_tensor is not None:
                self.device = first_tensor.device
                self._initialize_model(self.device)
        
        predictions = OrderedDict()
        
        # batch: dict_keys(['front_img_1', 'left_wrist_img', 'right_wrist_img', 
        # 'joint_positions', 'ee_pose', 'actions_joints', 'actions_cartesian', 'embodiment', 'pad_mask'])
        for embodiment_id, _batch in batch.items():
            cam_keys = self.camera_keys[embodiment_id]
            proprio_keys = self.proprio_keys[embodiment_id]
            lang_keys = self.lang_keys[embodiment_id]
            ac_key = self.ac_keys[embodiment_id]
            embodiment_name = get_embodiment(embodiment_id).lower()
            
            # Transform to cosmos_policy format
            cosmos_batch = self._robomimic_to_cosmos_policy_data(
                _batch, cam_keys, proprio_keys, lang_keys, ac_key, embodiment_name
            )
            
            # Call cosmos_policy model training_step
            if self.model is None:
                # Placeholder: model not initialized
                # In real implementation, this should call model.training_step()
                logger.warning("CosmosPolicy model not initialized - using placeholder loss")
                loss = torch.tensor(0.0, device=self.device)
                output_batch = {}
            else:
                # Get current iteration (would need to track this)
                iteration = getattr(self, "_current_iteration", 0)
                output_batch, loss = self.model.training_step(cosmos_batch, iteration)
            
            predictions[f"{embodiment_name}_{ac_key}"] = _batch[ac_key]
            predictions[f"{embodiment_name}_loss"] = loss
        
        return predictions
    
    @override
    def forward_eval(self, batch):
        """
        Compute forward pass and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D)}
        """
        if self.device is None:
            first_tensor = next(iter(batch.values())) if isinstance(batch, dict) else None
            if first_tensor is not None:
                self.device = first_tensor.device
                self._initialize_model(self.device)
        
        predictions = OrderedDict()
        
        with torch.no_grad():
            for embodiment_id, _batch in batch.items():
                cam_keys = self.camera_keys[embodiment_id]
                proprio_keys = self.proprio_keys[embodiment_id]
                lang_keys = self.lang_keys[embodiment_id]
                ac_key = self.ac_keys[embodiment_id]
                embodiment_name = get_embodiment(embodiment_id).lower()
                
                # Transform to cosmos_policy format
                cosmos_batch = self._robomimic_to_cosmos_policy_data(
                    _batch, cam_keys, proprio_keys, lang_keys, ac_key, embodiment_name
                )
                
                # Call cosmos_policy model inference
                if self.model is None:
                    logger.warning("CosmosPolicy model not initialized - using placeholder prediction")
                    # Return original actions as placeholder
                    pred_actions = _batch[ac_key]
                else:
                    # Use model's generate_samples_from_batch or similar method
                    # This is a placeholder - actual implementation needs proper inference
                    pred_actions = _batch[ac_key]  # Placeholder
                
                # Unnormalize predictions
                pred_dict = {ac_key: pred_actions}
                unnorm_preds = self.data_schematic.unnormalize_data(pred_dict, embodiment_id)
                
                for key in unnorm_preds:
                    predictions[f"{embodiment_name}_{key}"] = unnorm_preds[key]
        
        return predictions
    
    @override
    def forward_eval_logging(self, batch):
        """
        Called by pl_model to generate a dictionary of metrics and an image visualization
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            metrics (dict): metricname: value (float)
            images_dict (dict): embodiment_id: images (np.ndarray)
        """
        from torchmetrics import MeanSquaredError
        
        preds = self.forward_eval(batch)
        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        
        for embodiment_id, _batch in batch.items():
            _batch = self.data_schematic.unnormalize_data(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys[embodiment_id]
            pred_key = f"{embodiment_name}_{ac_key}"
            
            if pred_key in preds:
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(),
                    _batch[ac_key].cpu()
                )
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu(),
                    _batch[ac_key][:, -1].cpu()
                )
            
            ims = self.visualize_preds(preds, _batch)
            images_dict[embodiment_id] = ims
        
        return metrics, images_dict
    
    @override
    def visualize_preds(self, predictions, batch):
        """
        Helper function to visualize predictions on top of images
        Args:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D)}
            batch (dict): {ac_key: torch.Tensor (B, Seq, D), front_img_1: torch.Tensor (B, 3, H, W)}
        Returns:
            ims (np.ndarray): (B, H, W, 3) - images with actions drawn on top
        """
        from egomimic.utils.egomimicUtils import draw_actions
        
        # Get embodiment from batch
        embodiment_id = batch.get("embodiment", [0])[0].item() if "embodiment" in batch else 0
        embodiment_name = get_embodiment(embodiment_id).lower()
        ac_key = self.ac_keys[embodiment_id]
        
        # Get visualization image key
        viz_img_key = self.data_schematic.viz_img_key()[embodiment_id]
        ims = (batch[viz_img_key].cpu().numpy().transpose((0, 2, 3, 1)) * 255).astype(np.uint8)
        
        # Draw predictions and ground truth
        pred_key = f"{embodiment_name}_{ac_key}"
        if pred_key in predictions:
            preds = predictions[pred_key]
            gt = batch[ac_key]
            
            for b in range(ims.shape[0]):
                if preds.shape[-1] == 7 or preds.shape[-1] == 14:
                    ac_type = "joints"
                elif preds.shape[-1] == 3 or preds.shape[-1] == 6:
                    ac_type = "xyz"
                else:
                    ac_type = "joints"  # Default
                
                arm = "right" if preds.shape[-1] == 7 or preds.shape[-1] == 3 else "both"
                ims[b] = draw_actions(
                    ims[b], ac_type, "Purples", preds[b].cpu().numpy(),
                    self.camera_transforms.extrinsics, self.camera_transforms.intrinsics, arm=arm
                )
                ims[b] = draw_actions(
                    ims[b], ac_type, "Greens", gt[b].cpu().numpy(),
                    self.camera_transforms.extrinsics, self.camera_transforms.intrinsics, arm=arm
                )
        
        return ims
    
    @override
    def compute_losses(self, predictions, batch):
        """
        Compute losses based on network outputs in @predictions dict.
        Args:
            predictions (dict): dictionary containing network outputs, from @forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        loss_dict = OrderedDict()
        total_action_loss = None
        
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            loss_key = f"{embodiment_name}_loss"
            
            if loss_key in predictions:
                bc_loss = predictions[loss_key]
                if total_action_loss is None:
                    total_action_loss = torch.tensor(0.0, device=bc_loss.device)
                total_action_loss += bc_loss
                loss_dict[loss_key] = bc_loss
        
        if total_action_loss is not None:
            loss_dict["action_loss"] = total_action_loss / len(self.domains)
        else:
            loss_dict["action_loss"] = torch.tensor(0.0)
        
        return loss_dict
    
    @override
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of losses returned by compute_losses
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = OrderedDict()
        
        if "losses" in info:
            if "action_loss" in info["losses"]:
                log["Loss"] = info["losses"]["action_loss"].item()
            
            for loss_key, loss in info["losses"].items():
                log[loss_key] = loss.item()
        
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        
        return log

