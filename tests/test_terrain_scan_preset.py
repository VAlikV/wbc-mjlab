"""Cfg-level checks for the terrain + height-scan preset."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disable")

import wbc_mjlab.env.mdp as mdp
from wbc_mjlab.export.tracking_params_yaml import build_wbc_tracking_params
from wbc_mjlab.presets.terrain import CLEARANCE_SENSOR, HEIGHT_SCAN_SENSOR
from wbc_mjlab.robots.g1.constants import (
  G1_ANCHOR_BODY_NAME,
  G1_EE_TERMINATION_BODY_NAMES,
)
from wbc_mjlab.robots.g1.tasks import (
  g1_wbc_env_cfg,
  g1_wbc_terrain_scan_env_cfg,
  make_g1_wbc_env_cfg,
)
from wbc_mjlab.tasks import list_wbc_task_ids


def test_terrain_scan_task_registered() -> None:
  assert "Wbc-G1-Terrain-Scan" in list_wbc_task_ids()


def test_flat_task_untouched() -> None:
  cfg = g1_wbc_env_cfg()
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.commands["motion"].ground_reference is False
  names = {s.name for s in cfg.scene.sensors}
  assert HEIGHT_SCAN_SENSOR not in names
  assert CLEARANCE_SENSOR not in names
  assert "height_scan" not in cfg.observations["actor"].terms
  assert "terrain_levels" not in cfg.curriculum
  assert cfg.sim.nconmax == 35


def test_terrain_scan_cfg() -> None:
  cfg = g1_wbc_terrain_scan_env_cfg()
  wbc = g1_wbc_env_cfg()

  assert cfg.scene.terrain.terrain_type == "generator"
  assert cfg.scene.terrain.max_init_terrain_level == 0
  assert cfg.commands["motion"].ground_reference is True

  sensors = {s.name: s for s in cfg.scene.sensors}
  clearance_frames = [f.name for f in sensors[CLEARANCE_SENSOR].frame]
  assert clearance_frames[0] == G1_ANCHOR_BODY_NAME
  assert set(G1_EE_TERMINATION_BODY_NAMES) <= set(clearance_frames)

  for group in ("actor", "critic"):
    term = cfg.observations[group].terms["height_scan"]
    assert term.func is mdp.height_scan
    assert term.params["sensor_name"] == HEIGHT_SCAN_SENSOR

  assert cfg.terminations["anchor_pos"].func is mdp.bad_anchor_pos_z_terrain
  assert cfg.terminations["ee_body_pos"].func is mdp.bad_motion_body_pos_z_terrain
  # Thresholds carried over from the flat task, not loosened.
  for name in ("anchor_pos", "ee_body_pos"):
    assert (
      cfg.terminations[name].params["threshold"]
      == wbc.terminations[name].params["threshold"]
    )
  assert "terrain_levels" in cfg.curriculum

  for name, term in wbc.rewards.items():
    assert cfg.rewards[name].weight == term.weight


def test_terrain_scan_play_spreads_levels() -> None:
  cfg = make_g1_wbc_env_cfg(play=True, task_id="Wbc-G1-Terrain-Scan")
  assert cfg.curriculum == {}
  assert cfg.scene.terrain.max_init_terrain_level is None


def test_height_scan_export_layout() -> None:
  cfg = g1_wbc_terrain_scan_env_cfg()
  params = build_wbc_tracking_params(cfg, robot_id="g1", has_state_estimation=False)
  scan = params["actor_observations"]["height_scan"]
  ny, nx = scan["params"]["grid_shape_yx"]
  assert scan["dim"] == nx * ny == scan["params"]["num_rays"]
  assert scan["params"]["frame_body"] == "pelvis"
  assert scan["params"]["ray_alignment"] == "yaw"
