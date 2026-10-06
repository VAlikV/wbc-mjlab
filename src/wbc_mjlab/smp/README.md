# Standalone SMP prior

Runtime for the existing MimicKit TinyMDM checkpoint. It imports only PyTorch,
diffusers and PyYAML; no simulator, motion dataset or MimicKit import is needed.
The `arch.py`, `modules.py` and `activations.py` files are local copies of the
training architecture so checkpoint parameter names and calculations remain
compatible. Keep the same `diffusers` version as the training environment when
checking numerical equivalence.

Install the three dependencies in your Python environment:

```bash
pip install 'torch>=2.0' 'diffusers>=0.36.0' pyyaml
```

From the repository root:

```python
import torch
from SMP_standalon import SMPPrior

prior = SMPPrior.load(
    "output/smp_prior_g1_lafan/diffusion_config.yaml",
    "output/smp_prior_g1_lafan/model.pt",
    device="cuda",
)
history = torch.zeros(32, prior.num_steps, prior.feature_dim, device="cuda")
normalized = prior.normalize(history)                     # [32, input_dim]
raw_losses = prior.compute_sds_loss(history)              # [32, 3]
rewards = prior.compute_reward(history)                   # [32]
```

`history` must contain the **same** SMP observations, ordered oldest to newest,
as `compute_disc_obs` produced during training. This module does not construct
observations from MJLab state. Use the G1 joint order, key bodies, quaternion
convention, 30 Hz sampling and 10 frame history from training. A flattened
`[N, input_dim]` tensor is also accepted. The shape is inferred from checkpoint
weights and `env_config.yaml`; no dataset file is loaded.

## Observation vector for the G1/LAFAN checkpoint

The trained configuration has `global_obs: false`, `root_height_obs: true`,
`disc_dof_vel_obs: false`, five `key_bodies`, and 29 joints. One frame has
**204 float32 values** in this exact order (Python slices, end exclusive):

| Slice | Size | Contents |
| --- | ---: | --- |
| `0:3` | 3 | Root position relative to the **newest** root position, rotated by the inverse heading of that newest root. The `z` component is then replaced with the frame's **absolute root height** above the world origin. |
| `3:9` | 6 | Root orientation after removing the newest root's heading, encoded as `quat_to_tan_norm`. |
| `9:183` | 174 | 29 joint rotations, each encoded as six values, in the G1 MJCF/MimicKit joint order. |
| `183:198` | 15 | World positions of the five key bodies minus the root position **of this frame**, rotated by the inverse heading of the newest root; three coordinates per body. |
| `198:201` | 3 | Root linear velocity rotated by the inverse heading of the newest root. |
| `201:204` | 3 | Root angular velocity rotated the same way. |

The five key bodies, in order, are `left_ankle_roll_link`,
`right_ankle_roll_link`, `head_link`, `left_wrist_yaw_link`, and
`right_wrist_yaw_link`. Joint rotations use the 29 hinge joints of
`data/assets/g1/g1.xml`, in that model's order. The six-value rotation
encoding is **not** a generic 6D library convention: MimicKit's
`quat_to_tan_norm(q)` concatenates the vectors obtained by rotating the
unit X axis and unit Z axis by `q`, in that order. MimicKit quaternions are
`(x, y, z, w)`; convert from MJLab's convention if needed. Joint angles
themselves, joint velocities, contact flags and commands are **not** fields
of this checkpoint's observation.

The complete history is `[N, 10, 204]`, with timestamps
`t-9/30, ..., t-1/30, t` seconds. Its flattened form is `[N, 2040]`:
concatenate whole frames from oldest to newest. The reference root position
and heading for **every** frame in the window come from the newest frame.
The most recent frame therefore has root X/Y equal to zero, while its Z
retains the absolute root height. This reference must be recomputed as the
history window advances; independently normalizing each frame to its own
heading would change the input seen by the prior.

For numerical parity with training, compare the output of your MJLab
observation builder against MimicKit's `compute_disc_obs` on the *same* saved
trajectory before using reward. The standalone prior only checks shape and
device; a tensor with the right shape but a different field order or
coordinate convention will still produce a number.

`compute_sds_loss` implements the original `ESM_SDS_loss` math with independent
random noise per diffusion step. Set `torch.manual_seed(...)` before calls when
comparing implementations. The default timesteps are `(22, 15, 8)`.

`compute_reward` uses `exp(-mean(normalized_sds_losses) * sds_loss_scale)`.
The checkpoint contains *observation* mean/std, but not the separate SDS loss
normalizer used while training a policy. The latter starts with mean absolute
loss of one, matching MimicKit. To maintain running SDS statistics:

```python
rewards = prior.compute_reward(history, update_sds_stats=True)
# After all reward batches for the rollout, before the next rollout:
prior.sds_normalizer.update()
# Save prior.sds_normalizer.state_dict() in the policy checkpoint.
```

`update_sds_stats=True` records each batch and applies the *previous* statistics
to that batch, as in MimicKit. The normalizer is local to this process; if using
distributed PPO, aggregate its sums/counts across ranks before `update()`.
Set `normalize_sds=False` only to inspect the raw loss scale.

Quick smoke check in the environment with the required dependencies:

```bash
python -m SMP_standalon.smoke_test
```
