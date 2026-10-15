"""Reward functions for SMP RL tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.sensor import ContactSensor

from .runtime import DiffNormalizer, MotionFeatureBuffer

if TYPE_CHECKING:
  from collections.abc import Callable

  from mjlab.envs import ManagerBasedRlEnv

  TaskTerm = tuple["Callable[..., torch.Tensor]", float, dict]


def _update_buffer_from_sim(env: ManagerBasedRlEnv) -> None:
  """Push current sim kinematics onto the buffer tail, env-origin-relative
  (matching ``_prime_sim_and_buffer``) so features are placement-invariant."""
  robot = env.scene["robot"]
  ee_indexes = env._smp_ee_indexes  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  origins = env.scene.env_origins
  buffer.update(
    robot.data.root_link_pos_w - origins,
    robot.data.root_link_quat_w,
    robot.data.root_link_lin_vel_w,
    robot.data.root_link_ang_vel_w,
    robot.data.body_link_pos_w[:, ee_indexes] - origins[:, None, :],
    robot.data.joint_pos,
    robot.data.joint_vel,
  )


def smp_guidance_reward(
  env: ManagerBasedRlEnv,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 4.0,
  normalize: bool = True,
) -> torch.Tensor:
  """SDS-style guidance reward over fixed timesteps ``K``:
  ``exp(-w_s/|K| · Σ_{i∈K} ‖ε̂_i − ε_i‖²)``.  ``normalize`` divides each MSE by a
  ``DiffNormalizer`` running mean (policy-relative) vs. raw (absolute scale);
  always stashes the mean raw MSE on ``env._smp_raw_err``."""
  device = torch.device(env.device)
  model, scheduler, q_low, q_high, _, _ = env._smp_bundle  # type: ignore[attr-defined]
  normalizer: DiffNormalizer = env._smp_normalizer  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  _update_buffer_from_sim(env)

  features = buffer.compute_features()
  x_0 = 2.0 * (features - q_low) / (q_high - q_low + 1e-8) - 1.0
  num_envs = x_0.shape[0]

  total_err = torch.zeros(num_envs, device=device)
  total_raw = torch.zeros(num_envs, device=device)
  with torch.no_grad():
    for t_scalar in fixed_timesteps:
      if not 0 <= t_scalar < scheduler.num_timesteps:
        msg = f"fixed_timestep {t_scalar} out of range [0, {scheduler.num_timesteps})"
        raise ValueError(msg)
      t = torch.full((num_envs,), t_scalar, dtype=torch.long, device=device)
      noise = torch.randn_like(x_0)
      x_t = scheduler.add_noise(x_0, noise, t)
      eps_hat = model(x_t, t)
      mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
      total_raw += mse_per_env
      if normalize:
        total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)
      else:
        total_err += mse_per_env

  env._smp_raw_err = total_raw / len(fixed_timesteps)  # type: ignore[attr-defined]
  err = total_err / len(fixed_timesteps)
  return torch.exp(-err * ws)


def task_smp_product(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
) -> torch.Tensor:
  """``(Σ wᵢ · taskᵢ(env)) · r_smp`` — multiplicative SMP gating; ``task_terms`` is
  a tuple of ``(func, weight, kwargs)``.  Calls ``smp_guidance_reward`` once (the
  sole SMP-buffer update), so it must be the task's only SMP reward term."""
  task = sum(w * func(env, **kw) for func, w, kw in task_terms)
  return task * smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)


class AlternativeTaskSmpProductTerm(ManagerTermBase):
  """Manager reward term for the 59-feature SMP implementation.

  This keeps the alternative implementation isolated from the legacy
  210-feature MimicKit reward term while retaining the same task × SMP API.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    task_terms: tuple[TaskTerm, ...],
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    ws: float = 6.0,
    contact_sensor_name: str = "feet_ground_contact",
  ) -> torch.Tensor:
    task = sum(weight * func(env, **kwargs) for func, weight, kwargs in task_terms)
    smp = smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)
    product = task * smp

    # Keep the two factors visible in TensorBoard. The product alone cannot
    # distinguish actual locomotion learning from velocity gained while falling.
    # log = env.extras.setdefault("log", {})
    # log["Metrics/task_reward"] = task.mean()
    # log["Metrics/smp_reward"] = smp.mean()
    # log["Metrics/smp_raw_error"] = env._smp_raw_err.mean()
    # log["Metrics/task_smp_product"] = product.mean()
    # self._log_foot_forces(env, log, contact_sensor_name)
    # self._log_terrain_contact_diagnostics(env, log)
    return product

  @staticmethod
  def _log_foot_forces(
    env: ManagerBasedRlEnv,
    log: dict[str, torch.Tensor],
    sensor_name: str,
  ) -> None:
    """Log ground reaction forces for the left and right foot.

    The contact sensor returns net world-frame force for each ankle subtree.
    Slot order follows the robot body order: left foot, then right foot.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    force = sensor.data.force
    found = sensor.data.found
    if force is None or found is None:
      return
    if force.shape[1] != 2:
      raise RuntimeError(
        f"{sensor_name!r} must expose two foot slots, got shape {tuple(force.shape)}"
      )

    magnitude = torch.linalg.vector_norm(force, dim=-1)
    tangential = torch.linalg.vector_norm(force[..., :2], dim=-1)
    normal = force[..., 2].abs()
    in_contact = found > 0

    for index, side in enumerate(("left", "right")):
      log[f"Contact/{side}_force_mean"] = magnitude[:, index].mean()
      log[f"Contact/{side}_force_max"] = magnitude[:, index].max()
      log[f"Contact/{side}_normal_mean"] = normal[:, index].mean()
      log[f"Contact/{side}_tangential_mean"] = tangential[:, index].mean()
      log[f"Contact/{side}_contact_fraction"] = in_contact[:, index].float().mean()

    log["Contact/force_imbalance_mean"] = (
      magnitude[:, 0] - magnitude[:, 1]
    ).abs().mean()

  @staticmethod
  def _log_terrain_contact_diagnostics(
    env: ManagerBasedRlEnv,
    log: dict[str, torch.Tensor],
  ) -> None:
    """Log robot/terrain geometry independently of the contact sensor."""
    robot = env.scene["robot"]
    log["Contact/pelvis_z_mean"] = robot.data.root_link_pos_w[:, 2].mean()

    for side in ("left", "right"):
      sensor = env.scene.sensors[f"{side}_foot_terrain_height"]
      data = sensor.data
      sole_z = data.frame_pos_w[..., 2].flatten(start_dim=1).mean(dim=1)
      clearance = data.heights.flatten(start_dim=1).amin(dim=1)
      terrain_z = sole_z - clearance
      log[f"Contact/{side}_sole_z_mean"] = sole_z.mean()
      log[f"Contact/{side}_terrain_z_mean"] = terrain_z.mean()
      log[f"Contact/{side}_clearance_mean"] = clearance.mean()
      log[f"Contact/{side}_clearance_min"] = clearance.min()

    # MJWarp's nacon is a single total across all worlds, including contacts
    # outside the foot sensor. Divide by num_envs for a comparable mean.
    total_contacts = env.sim.data.nacon.float().sum()
    log["Contact/sim_ncon_total"] = total_contacts
    log["Contact/sim_ncon_mean"] = total_contacts / env.num_envs
