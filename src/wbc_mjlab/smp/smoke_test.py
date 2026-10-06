"""Load the trained G1 checkpoint and exercise the standalone inference API."""

import argparse

import torch

from .prior import SMPPrior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="output/smp_prior_g1_lafan/diffusion_config.yaml")
    parser.add_argument("--checkpoint", default="output/smp_prior_g1_lafan/model.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    prior = SMPPrior.load(args.config, args.checkpoint, args.device)
    history = torch.zeros(2, prior.num_steps, prior.feature_dim, device=args.device)
    losses = prior.compute_sds_loss(history)
    reward = prior.compute_reward(history, update_sds_stats=True)
    assert losses.shape == (2, len(prior.diffusion_steps))
    assert reward.shape == (2,)
    assert torch.isfinite(losses).all() and torch.isfinite(reward).all()
    prior.sds_normalizer.update()
    print(f"input={prior.input_dim}, losses={tuple(losses.shape)}, reward={tuple(reward.shape)}")


if __name__ == "__main__":
    main()
