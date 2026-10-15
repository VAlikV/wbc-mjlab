"""World-frame XY navigation command and SMP-compatible task rewards."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .command import _dir_world_to_local

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class WorldXyGoalCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  distance_range: tuple[float, float] = (1.0, 10.0)
  angle_range: tuple[float, float] = (-math.pi, math.pi)
  goal_radius: float = 0.3

  def __post_init__(self) -> None:
    if not 0 < self.distance_range[0] <= self.distance_range[1]:
      raise ValueError("distance_range must be positive and ordered")
    if self.angle_range[0] > self.angle_range[1]:
      raise ValueError("angle_range must be ordered")
    if self.goal_radius <= 0:
      raise ValueError("goal_radius must be positive")

  def build(self, env: ManagerBasedRlEnv) -> WorldXyGoalCommand:
    return WorldXyGoalCommand(self, env)


class WorldXyGoalCommand(CommandTerm):
  """Hold a world XY goal; expose [heading-frame delta X, delta Y]."""

  cfg: WorldXyGoalCommandCfg

  def __init__(self, cfg: WorldXyGoalCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.goal_xy_w = env.scene.env_origins[:, :2].clone()
    self.command_b = torch.zeros(self.num_envs, 2, device=self.device)
    self.goal_achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.metrics["distance_to_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goal_reached"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    self.goal_achieved[env_ids] = False
    return extras

  def record_goal_success(self) -> None:
    """Latch proximity-based arrival for terrain curriculum until reset."""
    self.goal_achieved |= self.distance <= self.cfg.goal_radius

  @property
  def delta_xy_w(self) -> torch.Tensor:
    return self.goal_xy_w - self.robot.data.root_link_pos_w[:, :2]

  @property
  def distance(self) -> torch.Tensor:
    return torch.linalg.vector_norm(self.delta_xy_w, dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = env_ids.numel()
    radius = torch.empty(n, device=self.device).uniform_(*self.cfg.distance_range)
    angle = torch.empty(n, device=self.device).uniform_(*self.cfg.angle_range)
    offset = radius[:, None] * torch.stack((angle.cos(), angle.sin()), dim=-1)
    # Match temp_smp/location: sample around the current robot position.
    self.goal_xy_w[env_ids] = self.robot.data.root_link_pos_w[env_ids, :2] + offset

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.command_b[ids, :2] = _dir_world_to_local(
      self.delta_xy_w[ids], self.robot.data.heading_w[ids]
    )

  def _update_metrics(self) -> None:
    self.metrics["distance_to_goal"][:] = self.distance
    self.metrics["goal_reached"][:] = (self.distance <= self.cfg.goal_radius).float()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    indices = visualizer.get_env_indices(self.num_envs)
    if not indices:
      return
    goals = self.goal_xy_w.detach().cpu().numpy()
    roots = self.robot.data.root_link_pos_w.detach().cpu().numpy()
    for index in indices:
      target = roots[index].copy()
      target[:2] = goals[index]
      visualizer.add_sphere(target, self.cfg.goal_radius, (0.2, 0.8, 0.2, 0.5))
      visualizer.add_arrow(roots[index], target, color=(0.2, 0.8, 0.2, 0.6), width=0.015)


def goal_position_tracking(
  env: ManagerBasedRlEnv, command_name: str, pos_err_scale: float = 0.25,
) -> torch.Tensor:
  """Reward proximity with exp(-scale * XY distance), matching temp_smp."""
  if pos_err_scale <= 0:
    raise ValueError("pos_err_scale must be positive")
  command: WorldXyGoalCommand = env.command_manager.get_term(command_name)
  # Keep curriculum success tracking active even when this is the only task
  # reward. Recording success does not change the position reward value.
  command.record_goal_success()
  env.extras.setdefault("log", {})["Metrics/goal_achieved_fraction"] = (
    command.goal_achieved.float().mean()
  )
  return torch.exp(-pos_err_scale * command.distance)


def goal_success(
  env: ManagerBasedRlEnv, command_name: str,
) -> torch.Tensor:
  """Reward every step inside the goal radius, independent of velocity.

  Use the current position rather than latched episode success: leaving the
  target region immediately removes the bonus.
  """
  command: WorldXyGoalCommand = env.command_manager.get_term(command_name)
  reward = (command.distance <= command.cfg.goal_radius).float()
  env.extras.setdefault("log", {})["Metrics/goal_success_reward"] = reward.mean()
  return reward
