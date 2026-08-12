"""Diffusion backbone for DECAF's causal generator.

Same move as `_decaf_ctgan.py` and `_decaf_dp.py`: keep DECAF's causal
generator structure (one sub-network per attribute, masked to its DAG
parents, generated in topological order) and its generation-time fairness
mechanism (`biased_edges` surrogate-value substitution), and swap only *how
each per-attribute sub-network is trained and sampled* -- a small conditional
denoising diffusion model (DDPM, Ho et al. 2020) instead of a GAN sub-network.

Why not a stock joint TabDDPM: TabDDPM denoises every column at once each
diffusion step, so there is no per-edge input to cut for `biased_edges` --
fairness surgery needs a conditioning input attached to one specific parent,
which only a *per-attribute* conditional decomposition provides. That
decomposition already exists in the tabular-diffusion literature --
CausalDiffTab (arXiv 2506.14206) and TabSCM (arXiv 2604.22337) both generate
column-by-column in topological order over a causal DAG -- but neither
implements DECAF's fairness mechanism (edge removal + surrogate
substitution) or a fairness taxonomy at all. This module is the missing
combination: DECAF's specific mechanism, on that kind of backbone. See
`notes/2026-08-12-decaf-diffusion-backbone.md` and the README's "Diffusion
backbones" section for the literature survey this design decision rests on.

Representation: reuses `_decaf_ctgan.ColumnBlocks` (one-hot column blocks) --
the columns here are ordinal category codes, same as the CTGAN backbone, and
Gaussian diffusion over a one-hot block is a standard simplification (the
"analog bits" / simplex-diffusion family) rather than TabDDPM's separate
multinomial-diffusion process for categoricals. That full categorical
diffusion is a documented possible upgrade, not implemented here -- see the
notes file.

Training is genuinely simpler than the GAN backbones, and that simplicity is
itself a finding worth reporting: DDPM training is a single supervised
noise-prediction regression (teacher-forced on real parent values, the
standard SCM factorization p(x) = prod_j p(x_j | parents(x_j))), not an
adversarial min-max. There is no discriminator, so there is no "only half the
model reads private data through post-processing" argument the way
`_decaf_dp.py` needs for DP-GAN -- DP-SGD (Abadi et al. 2016) applies
directly to the one loss, clipping and noising the *whole* model's per-sample
gradient every step. No weight clipping vs. gradient-penalty tradeoff, no
disabled privacy term, no PacGAN/conditional-vector soundness issues. The
per-sample cost is a T-step reverse loop at *generation* time instead of one
forward pass -- cheap here because each per-column network is small.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from ._decaf_ctgan import ColumnBlocks, topological_order


class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer/DDPM sinusoidal embedding of the diffusion
    timestep, shared by every per-column network."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class DiffusionSchedule:
    """Linear-beta DDPM schedule (Ho et al. 2020) with a fixed posterior
    variance (`sigma_t^2 = beta_t`, the paper's simpler of its two options).

    Defaults (`timesteps=50`, `beta_end=0.2`) are an untuned starting point,
    not a measured setting -- the standard 1000-step / beta_end=0.02 schedule
    is calibrated for large image models; a per-attribute MLP over a handful
    of categories converges over far fewer steps, but the exact schedule
    should be swept per dataset the same way epoch counts were for the GAN
    backbones (see `notes/2026-08-12-decaf-diffusion-backbone.md`).
    """

    def __init__(
        self,
        timesteps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.2,
        device: str = "cpu",
    ):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1.0 - alpha_bars).sqrt()

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) noise."""
        sab = self.sqrt_alpha_bars[t].unsqueeze(-1)
        somab = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        return sab * x0 + somab * noise

    def p_mean(
        self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor
    ) -> torch.Tensor:
        """Reverse step mean, given the model's predicted noise at `x_t`."""
        beta_t = self.betas[t].unsqueeze(-1)
        alpha_t = self.alphas[t].unsqueeze(-1)
        somab_t = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        return (x_t - beta_t / somab_t * eps_pred) / alpha_t.sqrt()

    def posterior_std(self, t: torch.Tensor) -> torch.Tensor:
        return self.betas[t].sqrt()


class BlockCausalDenoiser(nn.Module):
    """One noise-prediction network per attribute, masked to its DAG parents.

    The diffusion analogue of `_decaf_ctgan.BlockCausalGenerator`: same fixed
    parent mask over one-hot column blocks, same `biased_edges` shuffling
    applied to the *conditioning* input. The difference is what each
    per-column network computes -- not a sampled value, but the noise added
    to that column's own block at diffusion step `t` -- so sampling needs a
    T-step reverse loop per column (`DDPMCausalTrainer.generate`) rather than
    one forward pass.
    """

    def __init__(
        self,
        blocks: ColumnBlocks,
        dag_seed: Sequence[Sequence[int]],
        h_dim: int = 128,
        time_dim: int = 32,
    ):
        super().__init__()
        self.blocks = blocks
        self.n_cols = len(blocks.columns)
        self.total = blocks.total
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        mask = torch.zeros(self.total, self.n_cols)
        for parent, child in dag_seed:
            start, end = blocks.span(int(parent))
            mask[start:end, int(child)] = 1.0
        self.register_buffer("M", mask)

        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.total + size + time_dim, h_dim),
                    nn.SiLU(),
                    nn.Linear(h_dim, h_dim),
                    nn.SiLU(),
                    nn.Linear(h_dim, size),
                )
                for size in blocks.sizes
            ]
        )

    def masked_parents(
        self,
        generated: torch.Tensor,
        j: int,
        biased_edges: Optional[Dict[int, List[int]]] = None,
    ) -> torch.Tensor:
        """Zero out everything but column `j`'s parents, then apply DECAF's
        fairness shuffling (row-permute a listed parent's block) -- identical
        in effect to `BlockCausalGenerator.forward`'s inline masking, lifted
        out so both training (no shuffling) and generation (with shuffling)
        share it."""
        n = generated.shape[0]
        masked = generated * self.M[:, j]
        for parent in (biased_edges or {}).get(j, []):
            start, end = self.blocks.span(parent)
            perm = torch.randperm(n, device=generated.device)
            masked = masked.clone()
            masked[:, start:end] = masked[perm, start:end]
        return masked

    def forward(
        self, j: int, parent_cond: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        temb = self.time_embed(t)
        return self.nets[j](torch.cat([parent_cond, x_t, temb], dim=-1))


class DDPMCausalTrainer:
    """Trains and samples the per-attribute causal denoiser.

    `fit`: plain supervised training, teacher-forced on real parent values
    (the standard SCM chain-rule factorization -- no adversary, no generated
    rollout during training).

    `fit_dp`: the same loss under DP-SGD, clipping and noising the *whole*
    model's per-sample gradient every step (see the module docstring for why
    this needs none of DP-GAN's deviations).

    `generate`: topological, per-column T-step reverse diffusion, with
    `biased_edges` shuffling applied to each column's parent conditioning --
    fairness means exactly what it means for the GAN backbones.
    """

    def __init__(
        self,
        blocks: ColumnBlocks,
        dag_seed: Sequence[Sequence[int]],
        h_dim: int = 128,
        time_dim: int = 32,
        timesteps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 100,
        device: str = "cpu",
        dp_max_grad_norm: float = 1.0,
    ):
        self.blocks = blocks
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.dp_max_grad_norm = dp_max_grad_norm

        self.model = BlockCausalDenoiser(
            blocks, dag_seed, h_dim=h_dim, time_dim=time_dim
        ).to(self.device)
        self.schedule = DiffusionSchedule(
            timesteps, beta_start, beta_end, device=self.device
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gen_order = topological_order(dag_seed, len(blocks.columns))

        self.noise_multiplier: Optional[float] = None
        self.steps_taken = 0

    def _column_loss(
        self,
        real_onehot: torch.Tensor,
        t: torch.Tensor,
        noise_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Mean per-column noise-prediction MSE, conditioned on each column's
        *real* parent values (teacher forcing) -- one shared timestep `t` per
        row across all columns, the usual DDPM convention for a multi-feature
        example."""
        losses = []
        for j in range(self.model.n_cols):
            start, end = self.blocks.span(j)
            x0 = real_onehot[:, start:end]
            eps = noise_flat[:, start:end]
            x_t = self.schedule.q_sample(x0, t, eps)
            parent_cond = real_onehot * self.model.M[:, j]
            pred = self.model(j, parent_cond, x_t, t)
            losses.append(F.mse_loss(pred, eps))
        return torch.stack(losses).mean()

    def fit(self, codes: np.ndarray, seed: int = 0) -> None:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        data = torch.as_tensor(self.blocks.encode(codes), device=self.device)
        n = len(codes)
        steps_per_epoch = max(1, n // self.batch_size)

        for _ in range(self.max_epochs):
            for _ in range(steps_per_epoch):
                idx = rng.integers(0, n, size=self.batch_size)
                real = data[torch.as_tensor(idx, device=self.device)]
                t = torch.randint(
                    0, self.schedule.timesteps, (real.shape[0],), device=self.device
                )
                noise_flat = torch.randn(
                    real.shape[0], self.blocks.total, device=self.device
                )
                loss = self._column_loss(real, t, noise_flat)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

    def _private_grads(
        self, real: torch.Tensor, t: torch.Tensor, noise_flat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Per-sample-clipped, Gaussian-noised gradient of the *whole* column
        loss w.r.t. every model parameter -- the entire network reads private
        data (there is no free discriminator half here), so the entire
        gradient is privatized."""
        model = self.model
        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}
        spans = [self.blocks.span(j) for j in range(model.n_cols)]
        # Indexed *outside* vmap: `schedule_tensor[t]` is advanced indexing by
        # a batched index tensor, which vmap has no batching rule for. Once
        # resolved to a plain (n,) tensor of per-sample coefficients, feeding
        # it through `in_dims=0` needs only ordinary elementwise arithmetic
        # inside `loss_one`, which vmap handles fine.
        sab = self.schedule.sqrt_alpha_bars[t]
        somab = self.schedule.sqrt_one_minus_alpha_bars[t]

        def loss_one(p, b, x, t_i, sab_i, somab_i, noise_i):
            total = torch.zeros((), device=x.device)
            for j, (start, end) in enumerate(spans):
                x0 = x[start:end]
                eps = noise_i[start:end]
                x_t = sab_i * x0 + somab_i * eps
                parent_cond = x * model.M[:, j]
                pred = functional_call(
                    model,
                    (p, b),
                    (j, parent_cond.unsqueeze(0), x_t.unsqueeze(0), t_i.unsqueeze(0)),
                ).squeeze(0)
                total = total + F.mse_loss(pred, eps)
            return total / len(spans)

        per_sample = vmap(
            grad(loss_one), in_dims=(None, None, 0, 0, 0, 0, 0), randomness="different"
        )(params, buffers, real, t, sab, somab, noise_flat)

        n = real.shape[0]
        sq = torch.zeros(n, device=real.device)
        for g in per_sample.values():
            sq += g.reshape(n, -1).pow(2).sum(dim=1)
        scale = (self.dp_max_grad_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)

        sigma = self.noise_multiplier * self.dp_max_grad_norm
        out: Dict[str, torch.Tensor] = {}
        for name, g in per_sample.items():
            summed = (g * scale.view(-1, *([1] * (g.dim() - 1)))).sum(dim=0)
            noise = torch.normal(0.0, sigma, size=summed.shape, device=summed.device)
            out[name] = (summed + noise) / n
        return out

    def fit_dp(
        self, codes: np.ndarray, epsilon: float, delta: float, seed: int = 0
    ) -> None:
        from ._decaf_dp import find_noise_multiplier, spent_epsilon

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        data = torch.as_tensor(self.blocks.encode(codes), device=self.device)
        n = len(codes)

        steps_per_epoch = max(1, n // self.batch_size)
        total_steps = steps_per_epoch * self.max_epochs
        sample_rate = min(1.0, self.batch_size / n)
        self.noise_multiplier = find_noise_multiplier(
            target_epsilon=epsilon,
            delta=delta,
            sample_rate=sample_rate,
            steps=total_steps,
        )

        step = 0
        for _ in range(self.max_epochs):
            for _ in range(steps_per_epoch):
                # Poisson-style subsampling is what the accountant assumes; a
                # uniform-without-replacement batch is the standard stand-in
                # (same convention as `_decaf_dp.py`/`_decaf_ctgan.py`).
                idx = rng.choice(n, size=self.batch_size, replace=False)
                real = data[torch.as_tensor(idx, device=self.device)]
                t = torch.randint(
                    0, self.schedule.timesteps, (real.shape[0],), device=self.device
                )
                noise_flat = torch.randn(
                    real.shape[0], self.blocks.total, device=self.device
                )

                self.opt.zero_grad(set_to_none=False)
                for name, g in self._private_grads(real, t, noise_flat).items():
                    dict(self.model.named_parameters())[name].grad = g
                self.opt.step()
                step += 1

        self.steps_taken = step
        self.spent_epsilon = spent_epsilon(self.noise_multiplier, sample_rate, step, delta)

    @torch.no_grad()
    def generate(
        self,
        n: int,
        biased_edges: Optional[Dict[int, List[int]]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample `n` rows: topological, per-column T-step reverse diffusion,
        with `biased_edges` shuffling the listed parents' conditioning."""
        model, sched = self.model, self.schedule
        out = torch.zeros(n, self.blocks.total, device=self.device)

        for j in self.gen_order:
            start, end = self.blocks.span(j)
            # Parents are already fixed by the time column j is reached (full
            # topological order), so the conditioning is computed once and
            # reused across all T reverse steps for this column.
            parent_cond = model.masked_parents(out, j, biased_edges)
            x_t = torch.randn(n, end - start, device=self.device)
            for t_step in reversed(range(sched.timesteps)):
                t = torch.full((n,), t_step, device=self.device, dtype=torch.long)
                eps_pred = model(j, parent_cond, x_t, t)
                mean = sched.p_mean(x_t, t, eps_pred)
                if t_step > 0:
                    std = sched.posterior_std(t).unsqueeze(-1)
                    x_t = mean + std * torch.randn_like(x_t)
                else:
                    x_t = mean
            # Write the denoised (soft) block for downstream conditioning;
            # `ColumnBlocks.decode` argmaxes every block into an int code once
            # the full row is generated, same as the CTGAN backbone's
            # continuous-until-the-end convention.
            out = out.clone()
            out[:, start:end] = x_t

        return self.blocks.decode(out.cpu().numpy())
