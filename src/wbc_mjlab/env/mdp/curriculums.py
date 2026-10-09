"""Curriculum terms for WBC motion tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terrain_levels_motion(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
  """Move terrain level from the outcome of the finished episode.

  Time-outs move up one row, early terminations move down one row. The initial
  reset keeps the levels drawn from ``max_init_terrain_level``.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return {}

  move_up = env.termination_manager.time_outs[env_ids].clone()
  move_down = env.termination_manager.terminated[env_ids] & ~move_up
  if env.common_step_counter == 0:
    move_up.zero_()
    move_down.zero_()

  terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  return {
    "mean": levels.mean(),
    "max": levels.max(),
  }
