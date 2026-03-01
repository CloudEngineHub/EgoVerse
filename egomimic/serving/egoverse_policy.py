"""
EgoVerse policy wrapper for inference over WebSocket.

Implements infer(obs) and infer_batch(obs_list) by reusing ModelWrapper + HPT
forward_eval. Observation preprocessing is derived from PolicyRollout.process_obs_for_policy
and generalized for multiple embodiments (RBY1, EVA, etc.).
"""

import numpy as np
import torch

from egomimic.rldb.utils import get_embodiment_id


def _prepare_image(img: np.ndarray, device: torch.device, bgr_to_rgb: bool = True) -> torch.Tensor:
    """
    Convert numpy image to model input format.

    Args:
        img: (H, W, 3) uint8, BGR or RGB
        device: target device
        bgr_to_rgb: if True, assume input is BGR (e.g. from OpenCV)

    Returns:
        (1, 3, H, W) float32 tensor in [0, 1]
    """
    arr = torch.from_numpy(img[None, ...])
    if bgr_to_rgb:
        arr = arr[..., [2, 1, 0]]
    arr = arr.permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
    return arr


class EgoVersePolicy:
    """
    Policy wrapper that exposes infer/infer_batch for serving.

    Expects observation dict with keys matching the model's data_schematic
    (e.g. front_img_1, robot0_joint_pos for RBY1; front_img_1, right_wrist_img,
    left_wrist_img, joint_positions for EVA bimanual).
    Images: (H, W, 3) uint8, BGR preferred (converted to RGB internally).
    """

    def __init__(self, model_wrapper, device: str | None = None):
        self._model = model_wrapper
        self._model.eval()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.model.device = torch.device(self._device)

        if len(self._model.model.domains) != 1:
            raise ValueError(
                f"EgoVersePolicy supports single-domain models only, got domains={self._model.model.domains}"
            )
        self._embodiment_name = self._model.model.domains[0].lower()
        self._embodiment_id = get_embodiment_id(self._embodiment_name)
        self._ac_key = self._model.model.ac_keys[self._embodiment_id]
        self._cam_keys = self._model.model.camera_keys[self._embodiment_id]
        self._proprio_keys = self._model.model.proprio_keys[self._embodiment_id]
        self._data_schematic = self._model.model.data_schematic

        action_horizon = 10
        action_dim = 49
        heads = self._model.model.nets.policy.heads
        head = heads[self._embodiment_name] if self._embodiment_name in heads else None
        if head is not None:
            action_horizon = getattr(head, "action_horizon", action_horizon)
            infer_ac_dims = getattr(head, "infer_ac_dims", {})
            action_dim = infer_ac_dims.get(self._embodiment_name, action_dim)

        self._action_horizon = action_horizon
        self._action_dim = action_dim

    @property
    def metadata(self) -> dict:
        return {
            "methods": ["infer", "infer_batch"],
            "embodiment": self._embodiment_name,
            "action_horizon": self._action_horizon,
            "action_dim": self._action_dim,
            "camera_keys": self._cam_keys,
            "proprio_keys": self._proprio_keys,
        }

    def _obs_to_batch(self, obs: dict, batch_size: int = 1) -> dict:
        """
        Convert raw observation dict to normalized batch for forward_eval.

        obs keys must match data_schematic (front_img_1, robot0_joint_pos, etc.).
        """
        device = torch.device(self._device)
        data = {}

        for key in self._cam_keys:
            if key in obs:
                img = obs[key]
                if isinstance(img, np.ndarray):
                    data[key] = _prepare_image(img, device, bgr_to_rgb=True)
                elif torch.is_tensor(img):
                    data[key] = img.to(device).float()
                    if data[key].dim() == 3:
                        data[key] = data[key].unsqueeze(0)
                    if data[key].shape[1] != 3:
                        data[key] = data[key].permute(0, 3, 1, 2)
                    data[key] = data[key] / 255.0 if data[key].max() > 1 else data[key]
                else:
                    raise TypeError(f"Unsupported image type for {key}: {type(img)}")

        for key in self._proprio_keys:
            if key in obs:
                val = obs[key]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val).float()
                val = val.reshape(1, 1, -1).to(device)
                data[key] = val

        pad_mask = torch.ones(
            batch_size, self._action_horizon, 1, device=device, dtype=torch.bool
        )
        data["pad_mask"] = pad_mask
        data["embodiment"] = torch.tensor(
            [self._embodiment_id] * batch_size, dtype=torch.int64, device=device
        )

        action_placeholder = torch.zeros(
            batch_size, self._action_horizon, self._action_dim,
            device=device, dtype=torch.float32
        )
        data[self._ac_key] = action_placeholder

        for key, val in data.items():
            if key not in self._cam_keys and torch.is_tensor(val):
                data[key] = val.to(device, dtype=torch.float32)

        processed = {self._embodiment_id: data}
        processed[self._embodiment_id] = self._data_schematic.normalize_data(
            processed[self._embodiment_id], self._embodiment_id
        )
        return processed

    def infer(self, obs: dict) -> dict:
        """
        Run inference on a single observation.

        Args:
            obs: dict with keys matching embodiment (e.g. front_img_1, robot0_joint_pos for RBY1)

        Returns:
            dict with "actions" (np.ndarray), "embodiment", "server_timing"
        """
        batch = self._obs_to_batch(obs)
        with torch.inference_mode():
            preds = self._model.model.forward_eval(batch)
        ac_key = f"{self._embodiment_name}_{self._ac_key}"
        actions = preds[ac_key].detach().cpu().numpy()
        return {
            "actions": actions,
            "embodiment": self._embodiment_name,
        }

    def infer_batch(self, obs_list: list[dict]) -> list[dict]:
        """
        Run inference on a batch of observations.

        Args:
            obs_list: list of observation dicts

        Returns:
            list of result dicts (one per observation)
        """
        return [self.infer(obs) for obs in obs_list]
