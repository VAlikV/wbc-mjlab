"""G1 WBC task builders and mjlab task registry entries."""

from __future__ import annotations

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import wbc_mjlab.env.mdp as mdp
from wbc_mjlab.env.mdp.commands import MotionCommandCfg
from wbc_mjlab.env.mdp.curriculums import terrain_levels_motion
from wbc_mjlab.env.wbc_env_cfg import make_smp_wbc_env_cfg
from wbc_mjlab.presets.binary_failure import apply_binary_failure
from wbc_mjlab.presets.end_effector import apply_end_effector
from wbc_mjlab.presets.se_actor import apply_se_actor
from wbc_mjlab.presets.wbc import apply_wbc
from wbc_mjlab.presets.zest import apply_zest
from wbc_mjlab.robots.g1.actuators import G1_ACTION_SCALE
from wbc_mjlab.robots.g1.base import g1_base_cfg, wire_g1_imu_sensors
from wbc_mjlab.robots.g1.constants import (
  G1_EE_TERMINATION_BODY_NAMES,
  G1_ENDEFFECTOR_BODY_NAMES,
  G1_MOTION_BODY_NAMES,
  get_g1_robot_cfg,
)
from wbc_mjlab.tasks.config import WbcTaskConfig

DEFAULT_G1_TASK_ID = "Wbc-G1"


def _wire_g1_smp_contact_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add ground-contact and sole-clearance sensors used by SMP diagnostics."""
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  sole_height_cfgs = tuple(
    TerrainHeightSensorCfg(
      name=f"{side}_foot_terrain_height",
      frame=ObjRef(type="site", name=f"{side}_foot", entity="robot"),
      ray_alignment="yaw",
      reduction="min",
      pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.25),
      max_distance=5.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),
      debug_vis=False,
    )
    for side in ("left", "right")
  )
  cfg.scene.sensors = (*cfg.scene.sensors, feet_ground_cfg, *sole_height_cfgs)


def g1_wbc_env_cfg() -> ManagerBasedRlEnvCfg:
  cfg = g1_base_cfg()
  apply_wbc(
    cfg,
    motion_body_names=G1_MOTION_BODY_NAMES,
    ee_termination_bodies=G1_EE_TERMINATION_BODY_NAMES,
  )
  return cfg


def g1_smp_env_cfg() -> ManagerBasedRlEnvCfg:
  """G1 world-XY goal navigation with an SMP style reward."""
  cfg = make_smp_wbc_env_cfg()
  cfg.scene.entities = {"robot": get_g1_robot_cfg()}
  cfg.actions["joint_pos"].scale = G1_ACTION_SCALE
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  _wire_g1_smp_contact_sensor(cfg)
  cfg.viewer.body_name = "torso_link"
  return cfg


def g1_wbc_terrain_env_cfg() -> ManagerBasedRlEnvCfg:
  cfg = g1_base_cfg()

  apply_wbc(
    cfg,
    motion_body_names=G1_MOTION_BODY_NAMES,
    ee_termination_bodies=G1_EE_TERMINATION_BODY_NAMES,
  )

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
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
        # "tgrid": terrain_gen.BoxTiltedGridTerrainCfg(
        #   proportion=0.3,
        #   grid_width=0.5,
        #   tilt_range_deg=10.0,
        #   height_range=0.1,
        #   platform_width=0.5,
        #   border_width=0.25,
        #   floor_depth=2.0,
        # ),
      },
    ),
    max_init_terrain_level=0,
  )

  cfg.observations["actor"].terms["height_map"] = ObservationTermCfg(
    func=mdp.height_map,
    params={"sensor_name": "terrain_scan"},
    noise=Unoise(n_min=-0.05, n_max=0.05),
  )

  cfg.observations["critic"].terms["height_map"] = ObservationTermCfg(
    func=mdp.height_map,
    params={"sensor_name": "terrain_scan"},
  )

  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=terrain_levels_motion,
  )
  cfg.terminations["anchor_pos"].params["threshold"] = 0.5
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.4

  return cfg

def g1_smp_terrain_env_cfg() -> ManagerBasedRlEnvCfg:
  """G1 world-XY goal navigation with SMP style reward on random terrain."""
  cfg = make_smp_wbc_env_cfg()
  cfg.scene.entities = {"robot": get_g1_robot_cfg()}
  cfg.actions["joint_pos"].scale = G1_ACTION_SCALE
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  _wire_g1_smp_contact_sensor(cfg)
  cfg.viewer.body_name = "torso_link"

  terrain_scan_cfg = TerrainHeightSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    ray_alignment="yaw",
    reduction="none",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.25),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )
  root_terrain_height_cfg = TerrainHeightSensorCfg(
    name="root_terrain_height",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    ray_alignment="yaw",
    reduction="min",
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.25),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, terrain_scan_cfg, root_terrain_height_cfg)

  # Large box grids select SAP_SEGMENTED automatically in MJWarp. Its int32
  # work-package scan can overflow across many worlds, skipping all collisions.
  # NXN uses prefiltered pairs and avoids that global SAP scan.
  cfg.sim.broadphase = "nxn"

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    # collisions=(
    #   CollisionCfg(
    #     geom_names_expr=(r"^terrain.*",),
    #     contype=1,
    #     conaffinity=1,
    #     condim=3,
    #     priority=0,
    #     friction=(0.8,),
    #   ),
    # ),
    terrain_generator=TerrainGeneratorCfg(
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
         horizontal_scale = 0.2,
        ),
        "perlin": terrain_gen.HfPerlinNoiseTerrainCfg(
          proportion=0.25,
          height_range=(0.0, 0.4),
          octaves=4,
          persistence=0.5,
          lacunarity=2.0,
          scale=10.0,
          horizontal_scale=0.1,
          resolution=0.1,
          base_thickness_ratio=1.0,
          border_width=0.0,
        ),
        "rgrid": terrain_gen.BoxRandomGridTerrainCfg(
          proportion=0.25,
          grid_width=0.5,
          grid_height_range=(0.0, 0.10),
          platform_width=1.5,
          holes=False,
          merge_similar_heights=True,
          height_merge_threshold=0.05,
          max_merge_distance=64,
          border_width=0.25,
        ),
        # "tgrid": terrain_gen.BoxTiltedGridTerrainCfg(
        #   proportion=0.2,
        #   grid_width=0.5,
        #   tilt_range_deg=10.0,
        #   height_range=0.1,
        #   platform_width=1.5,
        #   border_width=0.25,
        #   floor_depth=2.0,
        # ),
      },
    ),
    max_init_terrain_level=0,
  )

  cfg.observations["actor"].terms["height_map"] = ObservationTermCfg(
    func=mdp.height_map,
    params={"sensor_name": "terrain_scan"},
    noise=Unoise(n_min=-0.05, n_max=0.05),
  )

  cfg.observations["critic"].terms["height_map"] = ObservationTermCfg(
    func=mdp.height_map,
    params={"sensor_name": "terrain_scan"},
  )

  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=terrain_levels_motion,
    params={"goal_command_name": "xy_goal", "success_threshold": 0.5},
  )

  cfg.terminations["base_height"] = TerminationTermCfg(
    func=mdp.root_height_above_terrain_below_minimum,
    params={
      "sensor_name": "root_terrain_height",
      "minimum_height": 0.3,
    },
  )

  return cfg

def g1_wbc_se_env_cfg() -> ManagerBasedRlEnvCfg:
  cfg = g1_wbc_env_cfg()
  apply_se_actor(cfg)
  wire_g1_imu_sensors(cfg)
  return cfg


def g1_wbc_zest_env_cfg() -> ManagerBasedRlEnvCfg:
  cfg = g1_base_cfg()
  apply_zest(
    cfg,
    reward_body_names=G1_ENDEFFECTOR_BODY_NAMES,
    contact_body_names=G1_MOTION_BODY_NAMES,
  )
  return cfg


def g1_wbc_zest_se_env_cfg() -> ManagerBasedRlEnvCfg:
  cfg = g1_wbc_zest_env_cfg()
  apply_se_actor(cfg)
  wire_g1_imu_sensors(cfg)
  return cfg


def g1_wbc_binary_failure_env_cfg() -> ManagerBasedRlEnvCfg:
  cfg = g1_base_cfg()
  apply_binary_failure(cfg)
  return cfg


def g1_wbc_ee_env_cfg() -> ManagerBasedRlEnvCfg:
  cfg = g1_wbc_env_cfg()
  apply_end_effector(cfg, body_names=G1_ENDEFFECTOR_BODY_NAMES)
  return cfg


def g1_wbc_ee_se_env_cfg() -> ManagerBasedRlEnvCfg:
  cfg = g1_wbc_ee_env_cfg()
  apply_se_actor(cfg)
  wire_g1_imu_sensors(cfg)
  return cfg

G1_WBC_TASKS: tuple[WbcTaskConfig, ...] = (
  WbcTaskConfig(
    task_id="Wbc-G1-SMP",
    robot_id="g1",
    description="World-frame XY goal navigation with a frozen SMP motion-style prior.",
    experiment_name="wbc_g1_smp",
    build_env_cfg=g1_smp_env_cfg,
  ),
  WbcTaskConfig(
    task_id="Wbc-G1-SMP-Terrain",
    robot_id="g1",
    description="World-frame XY goal navigation with SMP on procedural terrain.",
    experiment_name="wbc_g1_smp",
    build_env_cfg=g1_smp_terrain_env_cfg,
  ),
  WbcTaskConfig(
    task_id="Wbc-G1",
    robot_id="g1",
    description=(
      "Zest Table S4 tracking + RSI, EE z resets, light foot slip / anti-shake, history=1."
    ),
    experiment_name="wbc_g1",
    build_env_cfg=g1_wbc_env_cfg,
  ),
  WbcTaskConfig(
    task_id="Wbc-G1-Terrain",
    robot_id="g1",
    description="WBC G1 with procedural terrain and height-map observations.",
    experiment_name="wbc_g1_terrain",
    build_env_cfg=g1_wbc_terrain_env_cfg,
  ),
  WbcTaskConfig(
    task_id="Wbc-G1-SE",
    robot_id="g1",
    description="Wbc-G1 + SE obs (anchor pose tracking error, base lin vel).",
    experiment_name="wbc_g1_se",
    build_env_cfg=g1_wbc_se_env_cfg,
  ),
  WbcTaskConfig(
    task_id="Wbc-G1-Zest",
    robot_id="g1",
    description="Zest paper repro: no SE, reward-aligned RSI.",
    experiment_name="wbc_g1_zest",
    build_env_cfg=g1_wbc_zest_env_cfg,
  ),
  WbcTaskConfig(
    task_id="Wbc-G1-Zest-SE",
    robot_id="g1",
    description="Zest + SE obs (anchor pose tracking error, base lin vel).",
    experiment_name="wbc_g1_zest_se",
    build_env_cfg=g1_wbc_zest_se_env_cfg,
  ),
  WbcTaskConfig(
    task_id="Wbc-G1-BinaryFailure",
    robot_id="g1",
    description="Full obs, whole-body RSI with binary failure (BeyondMimic paper).",
    experiment_name="wbc_g1_binary",
    build_env_cfg=g1_wbc_binary_failure_env_cfg,
  ),
  WbcTaskConfig(
    task_id="Wbc-G1-EE",
    robot_id="g1",
    description=(
      "Wbc-G1 + actor EE ref_body_* command (no joint refs); critic/rewards keep full tracking."
    ),
    experiment_name="wbc_g1_ee",
    build_env_cfg=g1_wbc_ee_env_cfg,
  ),
  WbcTaskConfig(
    task_id="Wbc-G1-EE-SE",
    robot_id="g1",
    description="Wbc-G1-EE + SE obs (anchor pose tracking error, base lin vel).",
    experiment_name="wbc_g1_ee_se",
    build_env_cfg=g1_wbc_ee_se_env_cfg,
  ),
)

G1_TASK_BY_ID: dict[str, WbcTaskConfig] = {t.task_id: t for t in G1_WBC_TASKS}


def get_g1_task_config(task_id: str = DEFAULT_G1_TASK_ID) -> WbcTaskConfig:
  try:
    return G1_TASK_BY_ID[task_id]
  except KeyError as exc:
    known = ", ".join(sorted(G1_TASK_BY_ID))
    raise KeyError(f"Unknown G1 task {task_id!r}. Known: {known}") from exc


def make_g1_wbc_env_cfg(
  *,
  play: bool = False,
  task_id: str = DEFAULT_G1_TASK_ID,
  **kwargs,
) -> ManagerBasedRlEnvCfg:
  if kwargs:
    unknown = ", ".join(sorted(kwargs))
    raise TypeError(
      f"Unknown env cfg kwargs for G1: {unknown}. Pass task_id=<Wbc-G1-…>."
    )
  cfg = get_g1_task_config(task_id).build_env_cfg()

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events.pop("push_robot", None)
    if "init_smp_state" in cfg.events:
      # Match the alternative SMP play setup: avoid an expensive compile and
      # use a smaller one-shot GSI pool without background regeneration.
      cfg.events.pop("gsi_refresh", None)
      cfg.events["init_smp_state"].params["compile_model"] = False
      cfg.events["init_smp_state"].params["compile_mode"] = None
      cfg.events["init_smp_state"].params["gsi_buffer_size"] = 1024
    if "motion" in cfg.commands:
      motion_cmd = cfg.commands["motion"]
      assert isinstance(motion_cmd, MotionCommandCfg)
      motion_cmd.pose_range = {}
      motion_cmd.velocity_range = {}
      motion_cmd.assistive_wrench_enabled = False
      if "assistive_wrench" in cfg.events:
        cfg.events["assistive_wrench"].params["enabled"] = False
    elif "twist" in cfg.commands:
      twist_cmd = cfg.commands["twist"]
      assert isinstance(twist_cmd, UniformVelocityCommandCfg)
      twist_cmd.debug_vis = True

  return cfg


__all__ = [
  "DEFAULT_G1_TASK_ID",
  "G1_TASK_BY_ID",
  "G1_WBC_TASKS",
  "get_g1_task_config",
  "g1_smp_env_cfg",
  "g1_wbc_binary_failure_env_cfg",
  "g1_wbc_ee_env_cfg",
  "g1_wbc_ee_se_env_cfg",
  "g1_wbc_env_cfg",
  "g1_wbc_se_env_cfg",
  "g1_wbc_zest_env_cfg",
  "g1_wbc_zest_se_env_cfg",
  "make_g1_wbc_env_cfg",
]
