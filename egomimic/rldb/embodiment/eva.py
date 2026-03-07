from __future__ import annotations

from typing import Literal

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolateLinear,
    InterpolatePose,
    NumpyToTensor,
    PoseCoordinateFrameTransform,
    Transform,
    XYZWXYZ_to_XYZYPR,
)
from egomimic.utils.egomimicUtils import (
    EXTRINSICS,
)
from egomimic.utils.pose_utils import (
    _matrix_to_xyzwxyz,
)
from egomimic.utils.type_utils import _to_numpy
from egomimic.utils.viz_utils import (
    _viz_axes,
    _viz_traj,
)


class Eva(Embodiment):
    VIZ_INTRINSICS_KEY = "base"
    VIZ_IMAGE_KEY = "observations.images.front_img_1"

    @staticmethod
    def get_transform_list() -> list[Transform]:
        return _build_eva_bimanual_transform_list()

    @classmethod
    def viz_transformed_batch(cls, batch, mode=""):
        """
        Visualize one transformed EVA batch sample.

        Modes:
            - traj: draw left/right trajectories from actions_cartesian.
            - axes: draw local xyz axes at each anchor using ypr.
        """
        image_key = cls.VIZ_IMAGE_KEY
        action_key = "actions_cartesian"
        intrinsics_key = "base"
        mode = (mode or "traj").lower()

        images = _to_numpy(batch[image_key][0])
        actions = _to_numpy(batch[action_key][0])

        return cls.viz(
            images=images, actions=actions, mode=mode, intrinsics_key=intrinsics_key
        )

    @classmethod
    def viz(cls, images, actions, mode=Literal["traj", "axes"], intrinsics_key=None):
        intrinsics_key = intrinsics_key or cls.VIZ_INTRINSICS_KEY
        if mode == "traj":
            return _viz_traj(
                images=images,
                actions=actions,
                intrinsics_key=intrinsics_key,
            )
        if mode == "axes":
            return _viz_axes(
                images=images,
                actions=actions,
                intrinsics_key=intrinsics_key,
            )
        raise ValueError(
            f"Unsupported mode '{mode}'. Expected one of: " f"('traj', 'axes')."
        )

    @classmethod
    def get_keymap(cls):
        return {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "observations.images.right_wrist_img": {
                "key_type": "camera_keys",
                "zarr_key": "images.right_wrist",
            },
            "observations.images.left_wrist_img": {
                "key_type": "camera_keys",
                "zarr_key": "images.left_wrist",
            },
            "right.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_ee_pose",
            },
            "right.obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "right.gripper",
            },
            "left.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "left.obs_ee_pose",
            },
            "left.obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "left.gripper",
            },
            "right.gripper": {
                "key_type": "action_keys",
                "zarr_key": "right.gripper",
                "horizon": 45,
            },
            "left.gripper": {
                "key_type": "action_keys",
                "zarr_key": "left.gripper",
                "horizon": 45,
            },
            "right.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_ee_pose",
                "horizon": 45,
            },
            "left.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "left.cmd_ee_pose",
                "horizon": 45,
            },
        }


def _build_eva_bimanual_transform_list(
    *,
    left_target_world: str = "left_extrinsics_pose",
    right_target_world: str = "right_extrinsics_pose",
    left_cmd_world: str = "left.cmd_ee_pose",
    right_cmd_world: str = "right.cmd_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_obs_gripper: str = "left.obs_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    left_gripper: str = "left.gripper",
    right_gripper: str = "right.gripper",
    left_cmd_camframe: str = "left.cmd_ee_pose_camframe",
    right_cmd_camframe: str = "right.cmd_ee_pose_camframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
    extrinsics_key: str = "x5Dec13_2",
    is_quat: bool = True,
) -> list[Transform]:
    """Canonical EVA bimanual transform pipeline used by tests and notebooks."""
    extrinsics = EXTRINSICS[extrinsics_key]
    left_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["left"][None, :])[0]
    right_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["right"][None, :])[0]
    left_extra_batch_key = {"left_extrinsics_pose": left_extrinsics_pose}
    right_extra_batch_key = {"right_extrinsics_pose": right_extrinsics_pose}

    mode = "xyzwxyz" if is_quat else "xyzypr"
    transform_list = [
        ActionChunkCoordinateFrameTransform(
            target_world=left_target_world,
            chunk_world=left_cmd_world,
            transformed_key_name=left_cmd_camframe,
            extra_batch_key=left_extra_batch_key,
            mode=mode,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_target_world,
            chunk_world=right_cmd_world,
            transformed_key_name=right_cmd_camframe,
            extra_batch_key=right_extra_batch_key,
            mode=mode,
        ),
        PoseCoordinateFrameTransform(
            target_world=left_target_world,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_pose,
            mode=mode,
        ),
        PoseCoordinateFrameTransform(
            target_world=right_target_world,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_pose,
            mode=mode,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_cmd_camframe,
            output_action_key=left_cmd_camframe,
            stride=stride,
            mode=mode,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_cmd_camframe,
            output_action_key=right_cmd_camframe,
            stride=stride,
            mode=mode,
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=left_gripper,
            output_action_key=left_gripper,
            stride=stride,
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=right_gripper,
            output_action_key=right_gripper,
            stride=stride,
        ),
    ]

    if is_quat:
        transform_list.append(
            XYZWXYZ_to_XYZYPR(
                keys=[
                    left_cmd_camframe,
                    right_cmd_camframe,
                    left_obs_pose,
                    right_obs_pose,
                ]
            )
        )

    transform_list.extend(
        [
            ConcatKeys(
                key_list=[
                    left_cmd_camframe,
                    left_gripper,
                    right_cmd_camframe,
                    right_gripper,
                ],
                new_key_name=actions_key,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[
                    left_obs_pose,
                    left_obs_gripper,
                    right_obs_pose,
                    right_obs_gripper,
                ],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(
                keys_to_delete=[
                    left_cmd_world,
                    right_cmd_world,
                    left_target_world,
                    right_target_world,
                ]
            ),
            NumpyToTensor(
                keys=[
                    actions_key,
                    obs_key,
                ]
            ),
        ]
    )
    return transform_list
