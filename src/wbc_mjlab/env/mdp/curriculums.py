"""Terrain curriculum driven by successful completed episodes."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv


def terrain_levels_motion(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  success_threshold: float = 0.7,
  min_episodes: int = 1024,
  goal_command_name: str | None = None,
) -> dict[str, torch.Tensor]:
  """Promote each terrain type when enough completed episodes succeed.

  A timeout counts as success only when no failure termination also occurred.
  With goal_command_name, success requires the command's latched goal arrival
  instead of a timeout; a failed episode never counts as successful.
  Evaluate non-overlapping windows spanning at least one full episode duration,
  so early failures cannot close a window before survivors have time to finish.
  Apply the current stage at reset; never teleport active robots. Stages only
  increase and stop at the last row instead of wrapping to a random level.
  """
  if not 0.0 < success_threshold <= 1.0 or min_episodes <= 0:
    raise ValueError("success_threshold must be in (0, 1] and min_episodes positive")
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return {}

  step = env.common_step_counter
  num_types = terrain.terrain_origins.shape[1]
  state = getattr(env, "_terrain_timeout_curriculum", None)
  if state is None:
    state = {
      "stage": torch.zeros(num_types, dtype=torch.long, device=env.device),
      "episodes": torch.zeros(num_types, device=env.device),
      "successes": torch.zeros(num_types, device=env.device),
      "rate": torch.zeros(num_types, device=env.device),
      "window_start": torch.full(
        (num_types,), step, dtype=torch.long, device=env.device
      ),
    }
    # Respect the initial terrain placement rather than forcing row zero.
    state["stage"].scatter_reduce_(
      0, terrain.terrain_types, terrain.terrain_levels, reduce="amax"
    )
    env._terrain_timeout_curriculum = state

  types = terrain.terrain_types[env_ids]
  promoted = torch.zeros(num_types, dtype=torch.bool, device=env.device)
  if step > 0:
    timeouts = env.termination_manager.time_outs[env_ids]
    failures = env.termination_manager.terminated[env_ids]
    # Episodes still running on the previous stage must not influence the
    # success rate of the newly promoted stage.
    current_stage = terrain.terrain_levels[env_ids] == state["stage"][types]
    completed = (timeouts | failures) & current_stage
    if goal_command_name is None:
      successes = timeouts & ~failures & current_stage
    else:
      goal = env.command_manager.get_term(goal_command_name)
      successes = goal.goal_achieved[env_ids] & ~failures & completed
    state["episodes"].scatter_add_(0, types, completed.float())
    state["successes"].scatter_add_(0, types, successes.float())
    rate = state["successes"] / state["episodes"].clamp_min(1)
    min_steps = env.max_episode_length
    ready = (state["episodes"] >= min_episodes) & (
      step - state["window_start"] >= min_steps
    )
    promoted = ready & (rate >= success_threshold) & (
      state["stage"] < terrain.max_terrain_level - 1
    )
    state["stage"] += promoted.long()
    state["rate"] = torch.where(ready, rate, state["rate"])
    state["episodes"].masked_fill_(ready, 0)
    state["successes"].masked_fill_(ready, 0)
    state["window_start"].masked_fill_(ready, step)

    terrain.terrain_levels[env_ids] = state["stage"][types]
    terrain.env_origins[env_ids] = terrain.terrain_origins[
      terrain.terrain_levels[env_ids], types
    ]

  levels = terrain.terrain_levels.float()
  success_metric = (
    "timeout_success_rate" if goal_command_name is None else "goal_success_rate"
  )
  metrics = {
    "mean": levels.mean(),
    "max": levels.max(),
    "promoted": promoted.float().mean(),
    success_metric: state["rate"].mean(),
    "window_episodes": state["episodes"].mean(),
  }
  for terrain_type in range(num_types):
    metrics[f"type_{terrain_type}/level"] = state["stage"][terrain_type].float()
    metrics[f"type_{terrain_type}/{success_metric}"] = state["rate"][terrain_type]
  return metrics
