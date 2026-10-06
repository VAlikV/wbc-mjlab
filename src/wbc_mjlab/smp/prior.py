"""Standalone TinyMDM/SMP inference compatible with MimicKit checkpoints."""

from pathlib import Path

import torch
import yaml
from diffusers.schedulers import DDIMScheduler, DDPMScheduler

from .arch import CondTinyStableMotionDiTModel, TinyStableMotionDiTModel


class SDSNormalizer:
    """Per-timestep mean-absolute-loss normalization used by MimicKit."""

    def __init__(self, steps, device, min_diff=1e-4):
        self.steps = tuple(steps)
        self.mean_abs = torch.ones(len(self.steps), device=device)
        self.count = torch.zeros(1, dtype=torch.long, device=device)
        self.min_diff = min_diff
        self._sum_abs = torch.zeros_like(self.mean_abs)
        self._new_count = 0

    def record(self, losses):
        if losses.ndim != 2 or losses.shape[1] != len(self.steps):
            raise ValueError("losses must have shape [batch, num_diffusion_steps]")
        self._sum_abs += losses.detach().abs().sum(dim=0)
        self._new_count += losses.shape[0]

    def update(self):
        if not self._new_count:
            return
        total = self.count + self._new_count
        self.mean_abs.copy_((self.mean_abs * self.count + self._sum_abs) / total)
        self.count.copy_(total)
        self._sum_abs.zero_()
        self._new_count = 0

    def normalize(self, losses):
        return losses / self.mean_abs.clamp_min(self.min_diff)

    def state_dict(self):
        return {"steps": self.steps, "mean_abs": self.mean_abs.clone(), "count": self.count.clone()}

    def load_state_dict(self, state):
        if tuple(state["steps"]) != self.steps:
            raise ValueError("SDS normalization steps do not match")
        self.mean_abs.copy_(state["mean_abs"].to(self.mean_abs.device))
        self.count.copy_(state["count"].to(self.count.device))


class SMPPrior:
    """Frozen prior. Input is flattened [N, D] or chronological [N, steps, features]."""

    def __init__(self, config, checkpoint, device="cpu", diffusion_steps=(22, 15, 8),
                 sds_loss_scale=6.0, reward_scale=1.0):
        self.device = torch.device(device)
        with open(config, encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Expected a TinyMDM state_dict checkpoint")

        self.num_steps = self._history_steps(config)
        self.feature_dim = state["dmodel.proj_in.weight"].shape[1]
        self.input_dim = self.feature_dim * self.num_steps
        arch = self.config["arch_name"]
        options = dict(in_channels=self.feature_dim, out_channels=self.feature_dim,
                       num_layers=self.config["num_layers"],
                       attention_head_dim=self.config.get("attention_head_dim", 64),
                       num_attention_heads=self.config.get("num_attention_heads", 4),
                       dropout=self.config.get("dropout", 0.0))
        if arch == "DiT":
            model_class = TinyStableMotionDiTModel
        elif arch == "CondDiT":
            model_class = CondTinyStableMotionDiTModel
            options.update(num_class=self.config.get("num_class", 0),
                           cfg_dropout=self.config.get("cfg_dropout", 0.0))
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        self.dmodel = model_class(**options).to(self.device)
        prefix = "ema_dmodel.ema_model." if self.config.get("model_ema", False) else "dmodel."
        model_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not model_state:
            raise ValueError(f"Checkpoint has no weights under {prefix}")
        self.dmodel.load_state_dict(model_state, strict=True)
        self.dmodel.eval().requires_grad_(False)

        self.mean = state["obs_normalizer._mean"].to(self.device)
        self.std = state["obs_normalizer._std"].to(self.device)
        if self.mean.shape != (self.feature_dim,) or self.std.shape != self.mean.shape:
            raise ValueError("Checkpoint observation normalization has wrong dimensions")
        if torch.any(self.std <= 0):
            raise ValueError("Checkpoint observation std must be positive")

        self.T = int(self.config["T"])
        scheduler_options = dict(num_train_timesteps=self.T,
                                 beta_schedule=self.config["noise_schedule_mode"],
                                 prediction_type=self.config["estimate_mode"], clip_sample=False)
        self.diffusion_scheduler = DDPMScheduler(**scheduler_options)
        self.ddim_scheduler = DDIMScheduler(**scheduler_options)
        self.ddim_scheduler.set_timesteps(self.T, device=self.device)
        self.diffusion_scheduler.alphas_cumprod = self.diffusion_scheduler.alphas_cumprod.to(self.device)
        self.diffusion_steps = tuple(diffusion_steps)
        if not self.diffusion_steps or any(not isinstance(t, int) or t < 0 or t >= self.T
                                           for t in self.diffusion_steps):
            raise ValueError(f"diffusion_steps must be integers in [0, {self.T})")
        self.sds_normalizer = SDSNormalizer(self.diffusion_steps, self.device)
        self.sds_loss_scale = float(sds_loss_scale)
        self.reward_scale = float(reward_scale)

    @classmethod
    def load(cls, config_file, checkpoint_file, device="cpu", **kwargs):
        return cls(config_file, checkpoint_file, device, **kwargs)

    def _history_steps(self, config_path):
        env_path = Path(self.config["env_config"])
        candidates = (env_path, Path(config_path).parent / env_path,
                      Path(config_path).parent / env_path.name)
        for candidate in candidates:
            if candidate.is_file():
                with open(candidate, encoding="utf-8") as stream:
                    return int(yaml.safe_load(stream)["num_disc_obs_steps"])
        raise FileNotFoundError(f"Cannot find env_config {env_path}; it is needed for history length")

    def _flatten(self, history):
        if history.device != self.device:
            raise ValueError(f"Expected input on {self.device}, got {history.device}")
        if history.dtype != torch.float32:
            raise ValueError("Expected float32 observations")
        if history.ndim == 3 and history.shape[1:] == (self.num_steps, self.feature_dim):
            return history.reshape(history.shape[0], self.input_dim)
        if history.ndim == 2 and history.shape[1] == self.input_dim:
            return history
        raise ValueError(f"Expected [N, {self.input_dim}] or [N, {self.num_steps}, {self.feature_dim}]")

    def normalize(self, history):
        x = self._flatten(history)
        return ((x.reshape(-1, self.feature_dim) - self.mean) / self.std).reshape(x.shape)

    @torch.no_grad()
    def compute_sds_loss(self, history, diffusion_steps=None, normalized=False):
        """Return raw ESM SDS losses [N, K], matching TinyMDMModel.ESM_SDS_loss."""
        x = self._flatten(history) if normalized else self.normalize(history)
        steps = self.diffusion_steps if diffusion_steps is None else tuple(diffusion_steps)
        if not steps or any(not isinstance(t, int) or t < 0 or t >= self.T for t in steps):
            raise ValueError(f"diffusion_steps must be integers in [0, {self.T})")
        losses = []
        for step in steps:
            t = torch.full((x.shape[0],), step, device=self.device, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t = self.diffusion_scheduler.add_noise(x, noise, t)
            prediction = self.dmodel(x_t, timestep=t)
            x0 = self.ddim_scheduler.step(prediction, step, x_t).pred_original_sample
            alpha = self.diffusion_scheduler.alphas_cumprod[t].unsqueeze(-1)
            eps = (x_t - x0 * alpha.sqrt()) / (1 - alpha).sqrt()
            losses.append(((eps - noise) ** 2).mean(dim=1))
        return torch.stack(losses, dim=1)

    @torch.no_grad()
    def compute_reward(self, history, *, normalize_sds=True, update_sds_stats=False):
        """Return reward [N]. Call sds_normalizer.update() after the rollout."""
        losses = self.compute_sds_loss(history)
        if update_sds_stats:
            self.sds_normalizer.record(losses)
        if normalize_sds:
            losses = self.sds_normalizer.normalize(losses)
        return torch.exp(-losses.mean(dim=1) * self.sds_loss_scale) * self.reward_scale
