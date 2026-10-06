"""Stateful SMP reward term for MJLab environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from .prior import SMPPrior

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_KEY_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "head_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)


@dataclass
class _SmpState:
  """Raw quantities required to rebuild a frame relative to the newest root."""

  root_pos_w: torch.Tensor
  root_quat_w: torch.Tensor
  joint_pos: torch.Tensor
  key_body_pos_w: torch.Tensor
  root_lin_vel_w: torch.Tensor
  root_ang_vel_w: torch.Tensor


class SmpRewardTerm(ManagerTermBase):
  """Evaluate a frozen SMP prior on a 30 Hz history of robot states.

  The history stores raw world-frame states rather than ready-made observations.
  This is required because every frame in an SMP window must be expressed relative
  to the position and heading of the newest root in that window.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    params = cfg.params

    asset_cfg = params.get("asset_cfg", SceneEntityCfg("robot"))
    self._robot: Entity = env.scene[asset_cfg.name]
    self._key_body_names = tuple(
      params.get("key_body_names", _DEFAULT_KEY_BODY_NAMES)
    )
    # The MJLab G1 XML represents the head as a geom rigidly attached to the
    # torso, while the SMP training model exposed the same point as head_link.
    self._virtual_head_index: int | None = None
    resolved_body_names = list(self._key_body_names)
    if "head_link" in resolved_body_names and "head_link" not in self._robot.body_names:
      self._virtual_head_index = resolved_body_names.index("head_link")
      resolved_body_names[self._virtual_head_index] = params.get(
        "head_parent_body_name", "torso_link"
      )
    key_body_ids, found_names = self._robot.find_bodies(
      resolved_body_names, preserve_order=True
    )
    if tuple(found_names) != tuple(resolved_body_names):
      raise ValueError(
        "SMP key-body order does not match the checkpoint: "
        f"expected {tuple(resolved_body_names)}, found {tuple(found_names)}"
      )
    self._key_body_ids = key_body_ids
    self._virtual_head_offset = torch.tensor(
      params.get("head_offset", (0.0, 0.0, 0.43)),
      device=env.device,
      dtype=torch.float32,
    )

    self.prior = SMPPrior.load(
      params["config_file"],
      params["checkpoint_file"],
      device=env.device,
      diffusion_steps=tuple(params.get("diffusion_steps", (22, 15, 8))),
      sds_loss_scale=float(params.get("sds_loss_scale", 6.0)),
      reward_scale=float(params.get("reward_scale", 1.0)),
    )
    self._update_sds_stats = bool(params.get("update_sds_stats", True))
    self._sample_period = 1.0 / float(params.get("sample_frequency", 30.0))
    if self._sample_period <= 0.0:
      raise ValueError("sample_frequency must be positive")

    if self._robot.num_joints != 29:
      raise ValueError(
        f"The G1 SMP checkpoint expects 29 joints, got {self._robot.num_joints}"
      )
    # MimicKit's G1 model contains a fixed head_link in addition to the 29
    # actuated links. Its identity local rotation still occupies six features.
    self._num_joint_rotations = (self.prior.feature_dim - 30) // 6
    if 30 + 6 * self._num_joint_rotations != self.prior.feature_dim:
      raise ValueError(
        f"Cannot infer joint rotations from {self.prior.feature_dim} SMP features"
      )
    self._fixed_head_rotation_index: int | None = None
    if self._num_joint_rotations == self._robot.num_joints + 1:
      if self._virtual_head_index is None:
        raise ValueError("SMP checkpoint has one more joint rotation than the robot")
      self._fixed_head_rotation_index = (
        self._robot.joint_names.index("waist_pitch_joint") + 1
      )
    elif self._num_joint_rotations != self._robot.num_joints:
      raise ValueError(
        "SMP joint-rotation count does not match the robot: "
        f"checkpoint={self._num_joint_rotations}, robot={self._robot.num_joints}"
      )

    # MuJoCo stores hinge axes in the same natural joint order used by Entity.joint_pos.
    model_joint_ids = self._robot.data.indexing.joint_ids
    self._joint_axes = self._robot.data.model.jnt_axis[0, model_joint_ids].to(
      device=env.device, dtype=torch.float32
    )

    self._history = self._allocate_history()
    self._previous = self._allocate_state()
    self._time_since_sample = torch.zeros(self.num_envs, device=self.device)
    self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._cached_reward = torch.zeros(self.num_envs, device=self.device)

  def _allocate_state(self) -> _SmpState:
    """Allocate one raw state for every parallel environment."""
    n = self.num_envs
    return _SmpState(
      root_pos_w=torch.zeros(n, 3, device=self.device),
      root_quat_w=torch.zeros(n, 4, device=self.device),
      joint_pos=torch.zeros(n, 29, device=self.device),
      key_body_pos_w=torch.zeros(n, 5, 3, device=self.device),
      root_lin_vel_w=torch.zeros(n, 3, device=self.device),
      root_ang_vel_w=torch.zeros(n, 3, device=self.device),
    )

  def _allocate_history(self) -> _SmpState:
    """Allocate a chronological raw-state history for the SMP window."""
    n, steps = self.num_envs, self.prior.num_steps
    return _SmpState(
      root_pos_w=torch.zeros(n, steps, 3, device=self.device),
      root_quat_w=torch.zeros(n, steps, 4, device=self.device),
      joint_pos=torch.zeros(n, steps, 29, device=self.device),
      key_body_pos_w=torch.zeros(n, steps, 5, 3, device=self.device),
      root_lin_vel_w=torch.zeros(n, steps, 3, device=self.device),
      root_ang_vel_w=torch.zeros(n, steps, 3, device=self.device),
    )

  def _read_state(self) -> _SmpState:
    """Read the current simulated robot state without retaining simulator views."""
    root_pose = self._robot.data.root_link_pose_w
    root_velocity = self._robot.data.root_link_vel_w
    key_body_pos_w = self._robot.data.body_link_pos_w[:, self._key_body_ids].clone()
    if self._virtual_head_index is not None:
      parent_id = self._key_body_ids[self._virtual_head_index]
      parent_quat_w = self._robot.data.body_link_quat_w[:, parent_id]
      head_offset = self._virtual_head_offset.expand(self.num_envs, -1)
      key_body_pos_w[:, self._virtual_head_index] += quat_apply(
        parent_quat_w, head_offset
      )
    return _SmpState(
      root_pos_w=root_pose[:, :3].clone(),
      root_quat_w=root_pose[:, 3:7].clone(),
      joint_pos=self._robot.data.joint_pos.clone(),
      key_body_pos_w=key_body_pos_w,
      root_lin_vel_w=root_velocity[:, :3].clone(),
      root_ang_vel_w=root_velocity[:, 3:6].clone(),
    )

  @staticmethod
  def _copy_state(destination: _SmpState, source: _SmpState, env_ids: Any) -> None:
    """Copy selected environments between equally shaped state containers."""
    for field_name in _SmpState.__dataclass_fields__:
      getattr(destination, field_name)[env_ids] = getattr(source, field_name)[env_ids]

  def _fill_history(self, state: _SmpState, env_ids: torch.Tensor) -> None:
    """Initialize a history by repeating the first state after an environment reset."""
    for field_name in _SmpState.__dataclass_fields__:
      source = getattr(state, field_name)[env_ids].unsqueeze(1)
      target = getattr(self._history, field_name)
      target[env_ids] = source.expand_as(target[env_ids])

  def _interpolate(
    self, previous: _SmpState, current: _SmpState, env_ids: torch.Tensor, alpha: torch.Tensor
  ) -> _SmpState:
    """Interpolate a state at an exact 30 Hz sampling boundary."""
    alpha_vector = alpha[:, None]
    alpha_body = alpha[:, None, None]

    # Normalized lerp is sufficient here because adjacent control frames are close.
    q0 = previous.root_quat_w[env_ids]
    q1 = current.root_quat_w[env_ids]
    q1 = torch.where((q0 * q1).sum(dim=-1, keepdim=True) < 0.0, -q1, q1)
    quat = torch.nn.functional.normalize(q0 + alpha_vector * (q1 - q0), dim=-1)

    def lerp(field_name: str, factor: torch.Tensor) -> torch.Tensor:
      start = getattr(previous, field_name)[env_ids]
      end = getattr(current, field_name)[env_ids]
      return start + factor * (end - start)

    return _SmpState(
      root_pos_w=lerp("root_pos_w", alpha_vector),
      root_quat_w=quat,
      joint_pos=lerp("joint_pos", alpha_vector),
      key_body_pos_w=lerp("key_body_pos_w", alpha_body),
      root_lin_vel_w=lerp("root_lin_vel_w", alpha_vector),
      root_ang_vel_w=lerp("root_ang_vel_w", alpha_vector),
    )

  def _append_history(self, sample: _SmpState, env_ids: torch.Tensor) -> None:
    """Shift selected histories left and append the newest sampled state."""
    for field_name in _SmpState.__dataclass_fields__:
      history = getattr(self._history, field_name)
      history[env_ids, :-1] = history[env_ids, 1:].clone()
      history[env_ids, -1] = getattr(sample, field_name)

  @staticmethod
  def _quat_to_tan_norm(quaternion: torch.Tensor) -> torch.Tensor:
    """Match MimicKit's X-axis then Z-axis quaternion encoding."""
    basis_x = torch.zeros(*quaternion.shape[:-1], 3, device=quaternion.device)
    basis_z = torch.zeros_like(basis_x)
    basis_x[..., 0] = 1.0
    basis_z[..., 2] = 1.0
    return torch.cat(
      (quat_apply(quaternion, basis_x), quat_apply(quaternion, basis_z)), dim=-1
    )

  def _joint_rotation_features(self, joint_pos: torch.Tensor) -> torch.Tensor:
    """Encode link-local rotations using MimicKit's six-value convention."""
    half_angle = 0.5 * joint_pos
    axes = self._joint_axes.view(1, 1, 29, 3)
    quaternion = torch.cat(
      (torch.cos(half_angle).unsqueeze(-1), torch.sin(half_angle).unsqueeze(-1) * axes),
      dim=-1,
    )
    features = self._quat_to_tan_norm(quaternion)
    if self._fixed_head_rotation_index is not None:
      # A fixed link has identity local rotation: rotated X followed by rotated Z.
      identity = features.new_tensor((1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
      identity = identity.view(1, 1, 1, 6).expand(
        features.shape[0], features.shape[1], 1, -1
      )
      index = self._fixed_head_rotation_index
      features = torch.cat((features[:, :, :index], identity, features[:, :, index:]), dim=2)
    return features.flatten(start_dim=2)

  def _build_observation(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Build chronological [N, 10, 204] observations from raw history."""
    root_pos = self._history.root_pos_w[env_ids]
    root_quat = self._history.root_quat_w[env_ids]
    newest_pos = root_pos[:, -1]
    newest_heading = yaw_quat(root_quat[:, -1])
    heading = newest_heading[:, None].expand(-1, self.prior.num_steps, -1)

    relative_root_pos = quat_apply_inverse(heading, root_pos - newest_pos[:, None])
    # SMP preserves absolute root height instead of the heading-relative Z value.
    relative_root_pos[..., 2] = root_pos[..., 2]

    relative_root_quat = quat_mul(quat_inv(heading), root_quat)
    root_rotation = self._quat_to_tan_norm(relative_root_quat)
    joint_rotation = self._joint_rotation_features(self._history.joint_pos[env_ids])

    key_body_offset = self._history.key_body_pos_w[env_ids] - root_pos[:, :, None]
    key_heading = heading[:, :, None].expand(-1, -1, 5, -1)
    key_body_position = quat_apply_inverse(key_heading, key_body_offset).flatten(start_dim=2)

    root_lin_vel = quat_apply_inverse(heading, self._history.root_lin_vel_w[env_ids])
    root_ang_vel = quat_apply_inverse(heading, self._history.root_ang_vel_w[env_ids])
    observation = torch.cat(
      (
        relative_root_pos,
        root_rotation,
        joint_rotation,
        key_body_position,
        root_lin_vel,
        root_ang_vel,
      ),
      dim=-1,
    )
    if observation.shape[1:] != (self.prior.num_steps, self.prior.feature_dim):
      raise RuntimeError(f"Unexpected SMP observation shape: {tuple(observation.shape)}")
    return observation.to(dtype=torch.float32)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    """Invalidate reset environments; their first post-reset state seeds the history."""
    if env_ids is None:
      env_ids = slice(None)
    self._initialized[env_ids] = False
    self._time_since_sample[env_ids] = 0.0
    self._cached_reward[env_ids] = 0.0

  def __call__(self, env: ManagerBasedRlEnv, **_: Any) -> torch.Tensor:
    """Update 30 Hz histories and return the most recently evaluated SMP reward."""
    current = self._read_state()
    new_env_ids = torch.nonzero(~self._initialized, as_tuple=False).flatten()
    if new_env_ids.numel() > 0:
      self._fill_history(current, new_env_ids)
      self._copy_state(self._previous, current, new_env_ids)
      self._initialized[new_env_ids] = True

    # Locate the exact SMP sample between the preceding and current control states.
    self._time_since_sample += env.step_dt
    sampled_env_ids = torch.nonzero(
      self._time_since_sample >= self._sample_period, as_tuple=False
    ).flatten()
    if sampled_env_ids.numel() > 0:
      elapsed_before_step = self._time_since_sample[sampled_env_ids] - env.step_dt
      alpha = (self._sample_period - elapsed_before_step) / env.step_dt
      sample = self._interpolate(self._previous, current, sampled_env_ids, alpha)
      self._append_history(sample, sampled_env_ids)
      self._time_since_sample[sampled_env_ids] -= self._sample_period

    # Newly initialized environments also need an initial cached reward.
    reward_env_ids = torch.unique(torch.cat((new_env_ids, sampled_env_ids)))
    if reward_env_ids.numel() > 0:
      history = self._build_observation(reward_env_ids)
      self._cached_reward[reward_env_ids] = self.prior.compute_reward(
        history,
        update_sds_stats=self._update_sds_stats,
      )

    self._copy_state(self._previous, current, slice(None))
    return self._cached_reward
