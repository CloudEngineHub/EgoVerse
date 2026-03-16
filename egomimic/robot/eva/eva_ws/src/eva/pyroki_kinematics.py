"""
Drop-in replacement for i2rt Kinematics using PyRoKi + JAX backend.

Provides the same API surface as I2RTKinematics (fk, ik) so it can be
swapped in robot_interface.py and collect_demo_2.py with minimal changes.
"""

import os
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as prk
import yourdfpy


_GRASP_LINK_XML = """
  <link name="grasp_link"/>
  <joint name="joint_grasp" type="fixed">
    <origin xyz="0 0 0.1347" rpy="0 0 -1.5707963268"/>
    <parent link="link_6"/>
    <child link="grasp_link"/>
  </joint>
"""

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
TARGET_LINK = "grasp_link"


def _load_urdf_with_grasp_link(urdf_path: str) -> yourdfpy.URDF:
    """Load a YAM URDF and append a fixed grasp_link matching the MuJoCo grasp_site."""
    import tempfile

    with open(urdf_path, "r") as f:
        xml_str = f.read()

    xml_str = xml_str.replace("</robot>", _GRASP_LINK_XML + "</robot>")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", delete=False, dir=os.path.dirname(urdf_path)
    ) as tmp:
        tmp.write(xml_str)
        tmp_path = tmp.name

    try:
        urdf_obj = yourdfpy.URDF.load(
            tmp_path,
            load_meshes=False,
            build_scene_graph=False,
            load_collision_meshes=False,
        )
    finally:
        os.unlink(tmp_path)

    # build_scene_graph=False leaves _base_link as None, which breaks
    # PyRoKi's topological sort. Compute it manually.
    if urdf_obj.base_link is None:
        urdf_obj._base_link = urdf_obj._determine_base_link()

    return urdf_obj


@jdc.jit
def _solve_ik_jax(
    robot: prk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    initial_state: jax.Array | None,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    init_vals = jaxls.VarValues.make({joint_var: initial_state})

    costs = [
        prk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz),
                target_position,
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        prk.costs.limit_constraint(robot, joint_var),
    ]

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            initial_vals=init_vals,
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]


class PyRoKiKinematics:
    """Drop-in replacement for I2RTKinematics with PyRoKi backend.

    API matches I2RTKinematics so callers (robot_interface, collect_demo_2)
    need only swap the constructor.
    """

    def __init__(self, urdf_path: str, site_name: str = "grasp_site"):
        self.jax_device = jax.devices("cpu")[0]

        with jax.default_device(self.jax_device):
            urdf_obj = _load_urdf_with_grasp_link(urdf_path)
            self.robot = prk.Robot.from_urdf(urdf_obj)

        self.joint_names = JOINT_NAMES
        self.target_link_name = TARGET_LINK
        self.num_actuated_joints = self.robot.joints.num_actuated_joints

        actuated_names = self.robot.joints.actuated_names
        name_to_idx = {n: i for i, n in enumerate(actuated_names)}
        self._to_local = [name_to_idx[n] for n in self.joint_names]

        local_to_ext = {ext_pos: loc for loc, ext_pos in enumerate(self._to_local)}
        self._from_local = [local_to_ext[i] for i in range(len(self.joint_names))]

        self.target_link_index = self.robot.links.names.index(self.target_link_name)

        self._warmup()

    def _to_local_order(self, joint_values: np.ndarray) -> np.ndarray:
        local = np.zeros(self.num_actuated_joints)
        local[self._to_local] = joint_values
        return local

    def _from_local_order(self, local_values: np.ndarray) -> np.ndarray:
        ext = np.zeros(len(self.joint_names))
        ext[self._from_local] = local_values
        return ext

    def _warmup(self):
        dummy = np.zeros(len(self.joint_names), dtype=np.float64)
        self.fk(dummy)
        T_dummy = np.eye(4)
        self.ik(target_pose=T_dummy, init_q=dummy)

    def fk(self, joint_states: np.ndarray) -> np.ndarray:
        """Forward kinematics.

        Args:
            joint_states: (6,) joint angles in JOINT_NAMES order.

        Returns:
            4x4 homogeneous transform of grasp_link in world frame.
        """
        with jax.default_device(self.jax_device):
            local_q = self._to_local_order(joint_states).astype(np.float64)
            link_poses = self.robot.forward_kinematics(local_q).squeeze()
            target_pose = link_poses[self.target_link_index]

            wxyz = np.array(target_pose[:4], dtype=np.float64)
            pos = np.array(target_pose[-3:], dtype=np.float64)

            T = np.eye(4)
            T[:3, :3] = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
            T[:3, 3] = pos
            return T

    def ik(
        self,
        target_pose: np.ndarray = None,
        site_name: str = None,
        init_q: Optional[np.ndarray] = None,
        damping: float = 1e-3,
        max_iters: int = 500,
        pos_threshold: float = 1e-3,
        ori_threshold: float = 1e-3,
        verbose: bool = False,
    ) -> tuple:
        """Inverse kinematics.

        Args:
            target_pose: 4x4 homogeneous target transform.
            site_name: Ignored (kept for API compat).
            init_q: (6,) initial joint guess in JOINT_NAMES order.
            damping, max_iters, pos_threshold, ori_threshold: Solver params
                (damping/max_iters not used by jaxls but kept for API compat).
            verbose: Ignored.

        Returns:
            (success: bool, solved_joints: ndarray(6,))
        """
        pos_xyz = target_pose[:3, 3]
        rot_mat = target_pose[:3, :3]
        quat_xyzw = R.from_matrix(rot_mat).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )

        initial_state = None
        if init_q is not None:
            initial_state = self._to_local_order(init_q).astype(np.float64)

        with jax.default_device(self.jax_device):
            local_cfg = _solve_ik_jax(
                self.robot,
                jnp.array(self.target_link_index),
                jnp.array(quat_wxyz),
                jnp.array(pos_xyz.astype(np.float64)),
                initial_state,
            )

        solved_joints = self._from_local_order(np.array(local_cfg))

        achieved_T = self.fk(solved_joints)
        pos_err = np.linalg.norm(achieved_T[:3, 3] - pos_xyz)
        success = pos_err < pos_threshold

        return success, solved_joints
