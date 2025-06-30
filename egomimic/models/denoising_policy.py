import math
from typing import Tuple
from functools import partial

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.denoising_nets import ConditionalUnet1D
from rldb.utils import get_embodiment, get_embodiment_id

class DenoisingPolicy(nn.Module):
    """
    A diffusion-based policy head.

    Args:
        model (ConditionalUnet1D): The model used for prediction.
        noise_scheduler: The noise scheduler used for the diffusion process.
        action_horizon (int): The number of time steps in the action horizon.
        output_dim (int): The dimension of the output.
        num_inference_steps (int, optional): The number of inference steps.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        action_horizon,
        infer_ac_dims,
        num_inference_steps=None,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.action_horizon = action_horizon
        self.kwargs = kwargs
        self.padding = kwargs.get("padding", None)
        self.pooling = kwargs.get("pooling", True)
        self.human_6dof = kwargs.get("human_6dof", False)
        self.robot_cartesian = kwargs.get("robot_cartesian", False)
        self.model_type = kwargs.get("model_type", None)
        self.robot_domain = kwargs.get("robot_domain", None)
        self.infer_ac_dims = infer_ac_dims
        self.num_inference_steps = num_inference_steps
        if len(self.infer_ac_dims) == 0:
            raise ValueError(f"Ac has invalid value of {self.infer_ac_dims}")

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(f"{name} has requires_grad=False")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[{self.__class__.__name__}] Total trainable parameters: {total_params / 1e6:.2f}M")
    
    def preprocess_sampling(self, global_cond, embodiment_name, generator=None):
        if self.pooling == "mean":
            global_cond = global_cond.mean(dim=1)
        elif self.pooling == "flatten":
            global_cond = global_cond.reshape(global_cond.shape[0], -1)
        noise = torch.randn(
            size=(len(global_cond), self.action_horizon, self.infer_ac_dims[embodiment_name]),
            dtype=global_cond.dtype,
            device=global_cond.device,
            generator=generator,
        ).to(global_cond.device)
        return noise, global_cond
    
    def inference(self, noise, global_cond, generator=None) -> torch.Tensor:
        pass
    
    def sample_action(self, global_cond, embodiment_name, generator=None):
        noise, global_cond = self.preprocess_sampling(global_cond, embodiment_name, generator)
        
        x_t = self.inference(noise, global_cond, generator)
        return x_t
        
    def forward(self, global_cond):
        cond, embodiment = global_cond
        return self.sample_action(cond, embodiment)
    
    def predict(self, actions, global_cond) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prototype of model prediction functions  (noise, mean, velocity, etc) used in training step
        Args:
            actions (torch.Tensor): the ground truth actions
            global_cond (torch.Tensor): conditioning for the action prediction

        """
        pass
    
    def loss_fn(self, pred, target):
        """
        Default loss function for denoising policies
        Args:
            pred (torch.Tensor): model prediction
            target (torch.Tensor): target prediction
        Returns:
            loss (torch.Tensor): total loss
        """
        if target.shape[-1] == 6:
            xyz = pred[..., :3]
            ypr = pred[..., 3:6]
            
            xyz_gt = target[..., :3]
            ypr_gt = target[..., 3:6]
            loss = F.smooth_l1_loss(xyz, xyz_gt)
            if self.human_6dof:
                loss += F.smooth_l1_loss(ypr, ypr_gt)
            return loss
        elif target.shape[-1] == 12:
            xyz1 = pred[..., :3]
            ypr1 = pred[..., 3:6]
            xyz2 = pred[..., 6:9]
            ypr2 = pred[..., 9:12]

            xyz1_gt = target[..., :3]
            ypr1_gt = target[..., 3:6]
            xyz2_gt = target[..., 6:9]
            ypr2_gt = target[..., 9:12]

            loss = F.smooth_l1_loss(xyz1, xyz1_gt) + F.smooth_l1_loss(xyz2, xyz2_gt)
            if self.human_6dof:
                loss += F.smooth_l1_loss(ypr1, ypr1_gt) + F.smooth_l1_loss(ypr2, ypr2_gt)
            return loss
        if self.robot_cartesian:
            xyz1 = pred[..., :3]
            ypr1 = pred[..., 3:6]
            grip1 = pred[..., 6]
            xyz1_gt = target[..., :3]
            ypr1_gt = target[..., 3:6]
            grip1_gt = target[..., 6]
            loss = F.smooth_l1_loss(xyz1, xyz1_gt) + 0.5 * F.mse_loss(ypr1, ypr1_gt) + F.smooth_l1_loss(grip1, grip1_gt)
            if target.shape[-1] == 14:
                xyz2 = pred[..., 7:10]
                ypr2 = pred[..., 10:13]
                grip2 = pred[..., 13]
                xyz2_gt = target[..., 7:10]
                ypr2_gt = target[..., 10:13]
                grip2_gt = target[..., 13]

                loss += F.smooth_l1_loss(xyz2, xyz2_gt) + 0.5 * F.mse_loss(ypr2, ypr2_gt) + F.smooth_l1_loss(grip2, grip2_gt)
        else:
            loss = F.smooth_l1_loss(pred, target)
        return loss
    
    def preprocess_compute_loss(self, global_cond, data):
        """
        Preprocess the conditions and data
        Args:
            global_cond (torch.Tensor): The global condition tensor of shape (batch_size, global_cond_dim).
            data (dict): The input data dictionary containing the following keys:
                - "action" (torch.Tensor): The action tensor of shape (batch_size, action_horizon * action_dim).
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - loss (torch.Tensor): The computed loss.
            - pred (torch.Tensor): The predicted action.
            - target (torch.Tensor): The target action.
        """
        if self.pooling == "mean":
            global_cond = global_cond.mean(dim=1)
        elif self.pooling == "flatten":
            global_cond = global_cond.reshape(global_cond.shape[0], -1)

        embodiment_name = get_embodiment(data["embodiment"][0].item())

        actions = data["action"].reshape((len(global_cond), self.action_horizon, -1)) # Reshape the action tensor
        
        if self.padding is not None:
            if actions.shape[-1] == 6 or actions.shape[-1] == 12:
                pad_shape = (*actions.shape[:-1], 1)
                if self.padding == "gaussian":
                    padding = torch.randn(pad_shape, device=actions.device)
                elif self.padding == "zero":
                    padding = torch.zeros(pad_shape, device=actions.device)
                
                if actions.shape[-1] == 6:
                    actions = torch.concat((actions, padding), dim=-1)
                else:
                    actions = torch.concat((actions[..., :6], padding, actions[..., 6:], padding), dim=-1)
        return actions, global_cond
    
    def compute_loss(self, global_cond, data):
        """
        Compute the loss for the flow matching policy head.

        Args:
            global_cond (torch.Tensor): The global condition tensor of shape (batch_size, global_cond_dim).
            data (dict): The input data dictionary containing the following keys:
                - "action" (torch.Tensor): The action tensor of shape (batch_size, action_horizon * action_dim).

        Returns:
            torch.Tensor: The computed loss.

        """
        actions, global_cond = self.preprocess_compute_loss(global_cond, data)
        
        pred, target = self.predict(actions, global_cond)
        return self.loss_fn(pred, target)
    