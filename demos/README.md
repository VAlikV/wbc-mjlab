# Demo

Bundled **G1 checkpoint** for interactive play (`demos/wbc_g1/model.pt`, ~44 MB).

Motions come from the version-controlled **samples** CSVs — convert once to NPZ, then run the demo.

## Run

```bash
# 1. Convert bundled samples (one time)
uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 8

# 2. Interactive viewer (8 envs, uniform clip sampling)
uv run wbc-mjlab-demo
```

Opens the Viser viewer in the browser. Play also writes deploy artifacts under `demos/wbc_g1/params/` (gitignored).

## Manual play

```bash
uv run wbc-mjlab-play --task Wbc-G1 --dataset samples \
  --checkpoint-file demos/wbc_g1/model.pt --viewer viser --num-envs 8
```

## Checkpoint

| File | Source |
|------|--------|
| `demos/wbc_g1/model.pt` | `Wbc-G1` trained on LAFAN1 and BONES-SEED (`model_349999.pt` from `logs/rsl_rl/wbc_g1/2026-07-15_11-38-15`). BONES-SEED clips are not in the repo. |

More checkpoints may move to Git LFS later.
