"""Terrain + height-scan preset (perceptive tracking on procedural terrain).

Compose on top of a tracking preset (``apply_wbc``). Adds a curriculum terrain
generator, a base-mounted height scan (actor + critic), per-body clearance rays
for terrain-aware terminations, and grounds the reference on each sub-terrain.
Robot-specific frames are passed in (see ``wire_g1_terrain_scan``).

Flat-ground mocap cannot match foot heights on bumps, so tracking rewards stay
world-frame (small bias on rough rows) while the anchor / EE terminations
compare heights *above the terrain* instead of world z.
"""

from __future__ import annotations

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, TerrainHeightSensorCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import wbc_mjlab.env.mdp as mdp
from wbc_mjlab.env.mdp.commands import MotionCommandCfg

_MOTION = "motion"

HEIGHT_SCAN_SENSOR = "height_scan"
CLEARANCE_SENSOR = "keybody_clearance"

HEIGHT_SCAN_PATTERN = GridPatternCfg(size=(1.6, 1.0), resolution=0.2)
"""9 x 6 grid (x forward), yaw-aligned under the scan frame."""


def terrain_generator_cfg() -> TerrainGeneratorCfg:
  """Mild curriculum terrain: flat, uniform noise, Perlin hills, random grid."""
  return TerrainGeneratorCfg(
    size=(15.0, 15.0),
    num_rows=10,
    border_width=20.0,
    curriculum=True,
    sub_terrains={
      "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),
      "rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.25,
        noise_range=(0.0, 0.1),
        noise_step=0.02,
        scale_with_difficulty=True,
      ),
      "perlin": terrain_gen.HfPerlinNoiseTerrainCfg(
        proportion=0.25,
        height_range=(0.0, 0.25),
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
        scale=10.0,
        horizontal_scale=0.1,
        resolution=0.05,
        base_thickness_ratio=1.0,
        border_width=0.0,
      ),
      "rgrid": terrain_gen.BoxRandomGridTerrainCfg(
        proportion=0.25,
        grid_width=0.5,
        grid_height_range=(0.0, 0.10),
        platform_width=0.5,
        holes=False,
        merge_similar_heights=True,
        height_merge_threshold=0.05,
        max_merge_distance=3,
        border_width=0.25,
      ),
    },
  )


def apply_terrain_scan(
  cfg: ManagerBasedRlEnvCfg,
  *,
  scan_body_name: str,
  scan_offset: float,
  ee_termination_bodies: tuple[str, ...],
) -> None:
  """Procedural terrain + actor height scan + terrain-aware terminations.

  Args:
    cfg: Env config to mutate in place (after ``apply_wbc``).
    scan_body_name: Body the height-scan grid hangs from (e.g. pelvis).
    scan_offset: Nominal scan-frame height above flat ground; subtracted so the
      observation is ~0 on flat terrain.
    ee_termination_bodies: Bodies for the clearance-based EE height termination.
  """
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=terrain_generator_cfg(),
    max_init_terrain_level=0,
  )

  motion_cmd = cfg.commands[_MOTION]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.ground_reference = True
  anchor = motion_cmd.anchor_body_name

  height_scan = TerrainHeightSensorCfg(
    name=HEIGHT_SCAN_SENSOR,
    frame=ObjRef(type="body", name=scan_body_name, entity="robot"),
    pattern=HEIGHT_SCAN_PATTERN,
    ray_alignment="yaw",
    reduction="none",
    max_distance=3.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
  )
  clearance_bodies = (anchor, *(b for b in ee_termination_bodies if b != anchor))
  clearance = TerrainHeightSensorCfg(
    name=CLEARANCE_SENSOR,
    frame=tuple(ObjRef(type="body", name=b, entity="robot") for b in clearance_bodies),
    pattern=GridPatternCfg(size=(0.0, 0.0)),  # Single ray straight down per body.
    ray_alignment="world",
    reduction="min",
    max_distance=3.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )
  cfg.scene.sensors = (*cfg.scene.sensors, height_scan, clearance)

  scan_params = {"sensor_name": HEIGHT_SCAN_SENSOR, "offset": scan_offset}
  cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
    func=mdp.height_scan,
    params=scan_params,
    noise=Unoise(n_min=-0.05, n_max=0.05),
  )
  cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
    func=mdp.height_scan,
    params=dict(scan_params),
  )

  terms = cfg.terminations
  terms["anchor_pos"] = TerminationTermCfg(
    func=mdp.bad_anchor_pos_z_terrain,
    params={
      "command_name": _MOTION,
      "sensor_name": CLEARANCE_SENSOR,
      "threshold": terms["anchor_pos"].params["threshold"],
    },
  )
  terms["ee_body_pos"] = TerminationTermCfg(
    func=mdp.bad_motion_body_pos_z_terrain,
    params={
      "command_name": _MOTION,
      "sensor_name": CLEARANCE_SENSOR,
      "threshold": terms["ee_body_pos"].params["threshold"],
      "body_names": ee_termination_bodies,
    },
  )

  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=mdp.terrain_levels_motion)

  # Box / heightfield terrain produces far more contacts than a plane.
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 1024
