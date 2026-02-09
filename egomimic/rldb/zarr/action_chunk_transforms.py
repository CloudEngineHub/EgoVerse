"""
Embodiment-dependent action chunk transforms for ZarrDataset.

Replicates the prestacking transformations from aria_to_lerobot.py / eva_to_lerobot.py,
applied at load time instead of at data creation time. Raw action frames are loaded
as (action_horizon, action_dim) and interpolated to (chunk_length, action_dim).

Translation (xyz) and gripper dimensions use linear interpolation.
Rotation (euler ypr) dimensions use np.unwrap before interpolation and rewrap after,
matching the behaviour of egomimicUtils.interpolate_arr_euler.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Helper interpolation functions
# ---------------------------------------------------------------------------

def _interpolate_euler(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Euler-aware interpolation for a single (T, 6) or (T, 7) sequence.

    Layout: [x, y, z, yaw, pitch, roll, (optional gripper)]

    - xyz: linear interpolation
    - ypr: np.unwrap + linear interp + rewrap to [-pi, pi)
    - gripper: linear interpolation (if present)
    - Bad-data sentinel: any value >= 1e8 -> fill with 1e9
    """
    T, D = seq.shape
    assert D in (6, 7), f"Expected 6 or 7 dims, got {D}"

    # Bad-data sentinel check
    if np.any(seq >= 1e8):
        return np.full((chunk_length, D), 1e9)

    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)

    # Translation
    trans_interp = interp1d(old_time, seq[:, :3], axis=0, kind="linear")(new_time)

    # Rotation (euler) – unwrap before interp, rewrap after
    rot_unwrapped = np.unwrap(seq[:, 3:6], axis=0)
    rot_interp = interp1d(old_time, rot_unwrapped, axis=0, kind="linear")(new_time)
    rot_interp = (rot_interp + np.pi) % (2 * np.pi) - np.pi

    if D == 6:
        return np.concatenate([trans_interp, rot_interp], axis=-1)

    # Gripper
    grip_interp = interp1d(old_time, seq[:, 6:7], axis=0, kind="linear")(new_time)
    return np.concatenate([trans_interp, rot_interp, grip_interp], axis=-1)


def _interpolate_linear(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Simple linear interpolation for arbitrary (T, D) arrays."""
    T, D = seq.shape
    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)
    return interp1d(old_time, seq, axis=0, kind="linear")(new_time)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseActionChunkTransform:
    """Transform raw action frames into an interpolated action chunk."""

    def transform(self, actions: np.ndarray, chunk_length: int, key: str) -> np.ndarray:
        """
        Args:
            actions: (T, action_dim) raw consecutive action frames
            chunk_length: target number of frames after interpolation
            key: action key name (e.g. "actions_cartesian", "actions_joints")
        Returns:
            (chunk_length, action_dim) interpolated action chunk
        """
        raise NotImplementedError

    def transform_cartesian(self, actions: np.ndarray, chunk_length: int) -> np.ndarray:
        raise NotImplementedError

    def transform_joints(self, actions: np.ndarray, chunk_length: int) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete transform classes
# ---------------------------------------------------------------------------

class SingleArmEulerTransform(BaseActionChunkTransform):
    """Single-arm 6D (xyz + euler ypr) transform."""

    def transform_cartesian(self, actions: np.ndarray, chunk_length: int) -> np.ndarray:
        return _interpolate_euler(actions, chunk_length)

    def transform_joints(self, actions: np.ndarray, chunk_length: int) -> np.ndarray:
        return _interpolate_linear(actions, chunk_length)

    def transform(self, actions: np.ndarray, chunk_length: int, key: str) -> np.ndarray:
        if "cartesian" in key:
            return self.transform_cartesian(actions, chunk_length)
        return self.transform_joints(actions, chunk_length)


class SingleArmEulerGripTransform(BaseActionChunkTransform):
    """Single-arm 7D (xyz + euler ypr + gripper) transform."""

    def transform_cartesian(self, actions: np.ndarray, chunk_length: int) -> np.ndarray:
        return _interpolate_euler(actions, chunk_length)

    def transform_joints(self, actions: np.ndarray, chunk_length: int) -> np.ndarray:
        return _interpolate_linear(actions, chunk_length)

    def transform(self, actions: np.ndarray, chunk_length: int, key: str) -> np.ndarray:
        if "cartesian" in key:
            return self.transform_cartesian(actions, chunk_length)
        return self.transform_joints(actions, chunk_length)


class BimanualEulerTransform(BaseActionChunkTransform):
    """Bimanual 12D (2x 6D: xyz + euler ypr) transform."""

    def transform_cartesian(self, actions: np.ndarray, chunk_length: int) -> np.ndarray:
        left = _interpolate_euler(actions[:, :6], chunk_length)
        right = _interpolate_euler(actions[:, 6:], chunk_length)
        return np.concatenate([left, right], axis=-1)

    def transform_joints(self, actions: np.ndarray, chunk_length: int) -> np.ndarray:
        return _interpolate_linear(actions, chunk_length)

    def transform(self, actions: np.ndarray, chunk_length: int, key: str) -> np.ndarray:
        if "cartesian" in key:
            return self.transform_cartesian(actions, chunk_length)
        return self.transform_joints(actions, chunk_length)


class BimanualEulerGripTransform(BaseActionChunkTransform):
    """Bimanual 14D (2x 7D: xyz + euler ypr + gripper) transform."""

    def transform_cartesian(self, actions: np.ndarray, chunk_length: int) -> np.ndarray:
        left = _interpolate_euler(actions[:, :7], chunk_length)
        right = _interpolate_euler(actions[:, 7:], chunk_length)
        return np.concatenate([left, right], axis=-1)

    def transform_joints(self, actions: np.ndarray, chunk_length: int) -> np.ndarray:
        return _interpolate_linear(actions, chunk_length)

    def transform(self, actions: np.ndarray, chunk_length: int, key: str) -> np.ndarray:
        if "cartesian" in key:
            return self.transform_cartesian(actions, chunk_length)
        return self.transform_joints(actions, chunk_length)


# ---------------------------------------------------------------------------
# Registry: EMBODIMENT.value -> transform instance
# ---------------------------------------------------------------------------

ACTION_CHUNK_TRANSFORMS: dict[int, BaseActionChunkTransform] = {
    # EVE (6D per arm)
    0: SingleArmEulerTransform(),      # EVE_RIGHT_ARM
    1: SingleArmEulerTransform(),      # EVE_LEFT_ARM
    2: BimanualEulerTransform(),       # EVE_BIMANUAL
    # ARIA (6D per arm)
    3: SingleArmEulerTransform(),      # ARIA_RIGHT_ARM
    4: SingleArmEulerTransform(),      # ARIA_LEFT_ARM
    5: BimanualEulerTransform(),       # ARIA_BIMANUAL
    # EVA (7D per arm – includes gripper)
    6: SingleArmEulerGripTransform(),  # EVA_RIGHT_ARM
    7: SingleArmEulerGripTransform(),  # EVA_LEFT_ARM
    8: BimanualEulerGripTransform(),   # EVA_BIMANUAL
    # MECKA (6D per arm)
    9: BimanualEulerTransform(),       # MECKA_BIMANUAL
    10: SingleArmEulerTransform(),     # MECKA_RIGHT_ARM
    11: SingleArmEulerTransform(),     # MECKA_LEFT_ARM
}


def get_action_chunk_transform(embodiment_id: int) -> BaseActionChunkTransform:
    """Look up the action chunk transform for a given embodiment ID.

    Raises:
        KeyError: if the embodiment_id is not registered.
    """
    if embodiment_id not in ACTION_CHUNK_TRANSFORMS:
        raise KeyError(
            f"No action chunk transform registered for embodiment_id={embodiment_id}"
        )
    return ACTION_CHUNK_TRANSFORMS[embodiment_id]
