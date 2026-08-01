"""DP-SGD training for DECAF's causal generator.

DECAF as published has no privacy mechanism: `epsilon` is accepted by
`DECAFMethod` purely for interface parity and logged as NULL. That makes it a
non-private reference point rather than a competitor to MST/PrivBayes/PrivSyn
at a given epsilon. This module supplies the missing half -- DP-GAN-style
training (Xie et al. 2018) of the *discriminator*, which by post-processing
makes the whole generator differentially private -- so DECAF's causal
generator and its generation-time fairness mechanisms can be measured on the
same privacy axis as everything else in the grid.

The privacy argument, and the three deviations from stock DECAF it forces:

1. **Only the discriminator's real-data gradient touches private data.**
   The WGAN critic loss is `mean(D(fake)) - mean(D(real))`. The `fake` term is
   a function of the generator's parameters and fresh noise only, so it costs
   nothing. The `-mean(D(real))` term is privatized with per-sample gradient
   clipping to norm `max_grad_norm` plus Gaussian noise (DP-SGD, Abadi et al.
   2016). The generator only ever sees private data through the
   discriminator's gradients, so its updates are post-processing.

2. **No gradient penalty.** WGAN-GP's penalty is computed on interpolations
   *between real and fake* samples, so it reads private data on a path that
   per-sample clipping cannot cover without a double-backward through the
   per-sample machinery. We therefore fall back to the original WGAN's weight
   clipping (Arjovsky et al. 2017) to enforce the Lipschitz constraint, which
   is what the DP-WGAN literature does for the same reason. This costs some
   sample quality relative to GP and is the main quality/privacy tradeoff here.

3. **No ADS-GAN privacy term.** Stock DECAF adds
   `lambda_privacy * privacy_loss(batch, fake)` to the *generator* loss, which
   reads the real batch directly in a gradient that is never privatized. That
   term is disabled here (`lambda_privacy = 0`); leaving it on would void the
   DP guarantee outright.

Generation itself is unchanged: the same `Generator_causal.sequential` with
the same `biased_edges` shuffling, so the fairness mechanisms behave exactly
as they do for the non-private baseline.

Accounting uses Opacus's RDP accountant (moments accountant with amplification
by subsampling), binary-searching the noise multiplier that spends exactly the
requested `(epsilon, delta)` over the planned number of discriminator steps.
Note this is a *different* accountant from the zCDP path MST/AIM/PrivSyn use
(`_cdp2adp.py`); both produce an `(epsilon, delta)`-DP guarantee, so the
epsilon values are comparable in the usual sense, but the intermediate
`rho` is not.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

from decaf.DECAF import Discriminator, Generator_causal

try:
    from opacus import privacy_analysis as _rdp
except ImportError as exc:  # pragma: no cover - opacus ships with the decaf extra
    raise ImportError(
        "DP-GAN DECAF requires opacus for privacy accounting: pip install opacus"
    ) from exc

# Rényi orders to search over -- the standard Opacus grid. Fine-grained below
# 12 (where the optimum usually sits for small epsilon) and coarse above.
_ORDERS: Tuple[float, ...] = tuple(
    [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 128))
)


def spent_epsilon(
    noise_multiplier: float, sample_rate: float, steps: int, delta: float
) -> float:
    """(epsilon, delta) actually consumed by `steps` subsampled Gaussian
    mechanisms at this noise multiplier."""
    rdp = _rdp.compute_rdp(
        q=sample_rate,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=_ORDERS,
    )
    return float(_rdp.get_privacy_spent(_ORDERS, rdp, delta=delta)[0])


def find_noise_multiplier(
    target_epsilon: float,
    delta: float,
    sample_rate: float,
    steps: int,
    tolerance: float = 0.01,
    max_sigma: float = 1e7,
) -> float:
    """Smallest noise multiplier whose spend stays within `target_epsilon`.

    Binary search: epsilon is monotonically decreasing in sigma, so the
    smallest admissible sigma is the one that just meets the budget -- adding
    more noise than that would waste utility we already paid for.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    low, high = 1e-3, 1.0
    # Expand upward until the budget is met at all; a very tight epsilon on a
    # small dataset can need a genuinely huge sigma.
    while spent_epsilon(high, sample_rate, steps, delta) > target_epsilon:
        low = high
        high *= 2.0
        if high > max_sigma:
            raise ValueError(
                f"cannot reach epsilon={target_epsilon} in {steps} steps at "
                f"sample_rate={sample_rate:.5f} with sigma <= {max_sigma:g}"
            )
    while high - low > tolerance * low:
        mid = (low + high) / 2.0
        if spent_epsilon(mid, sample_rate, steps, delta) > target_epsilon:
            low = mid
        else:
            high = mid
    return high


def _flat_norms(grads: Dict[str, torch.Tensor], batch: int) -> torch.Tensor:
    """L2 norm of each sample's full gradient, across every parameter."""
    sq = torch.zeros(batch, device=next(iter(grads.values())).device)
    for g in grads.values():
        sq += g.reshape(batch, -1).pow(2).sum(dim=1)
    return sq.sqrt()


class DPCausalGANTrainer:
    """WGAN training of DECAF's causal generator under DP-SGD.

    Deliberately a plain torch loop rather than a `pl.LightningModule`: the
    privatized update replaces the optimizer step wholesale, so Lightning's
    manual-optimization path would only add indirection.
    """

    def __init__(
        self,
        input_dim: int,
        dag_seed: List[List[int]],
        h_dim: int = 200,
        lr: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 20,
        n_critic: int = 2,
        max_grad_norm: float = 1.0,
        weight_clip: float = 0.01,
        nonlin_out: Optional[List] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.n_critic = n_critic
        self.max_grad_norm = max_grad_norm
        self.weight_clip = weight_clip
        self.x_dim = input_dim

        self.generator = Generator_causal(
            z_dim=input_dim,
            x_dim=input_dim,
            h_dim=h_dim,
            dag_seed=dag_seed,
            nonlin_out=nonlin_out,
        ).to(self.device)
        self.discriminator = Discriminator(x_dim=input_dim, h_dim=h_dim).to(self.device)
        self.opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
        self.opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)

        self.gen_order = self._gen_order(dag_seed, input_dim)
        self.noise_multiplier: Optional[float] = None
        self.steps_taken = 0

    @staticmethod
    def _gen_order(dag_seed: List[List[int]], x_dim: int) -> List[int]:
        """Topological order over the fixed DAG.

        Mirrors `DECAF.get_gen_order`, but reads `dag_seed` directly instead of
        round-tripping through the mask matrix -- identical result, and it does
        not depend on the model being on any particular device.
        """
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(range(x_dim))
        g.add_edges_from((int(p), int(c)) for p, c in dag_seed)
        return list(nx.algorithms.dag.topological_sort(g))

    def _fake(self, n: int) -> torch.Tensor:
        """Generate `n` synthetic rows.

        The container passed to `sequential` is zeros, not a real batch. With a
        full topological order every column is overwritten before it is read as
        anyone's parent, so stock DECAF's habit of seeding from real rows is
        already a no-op -- but starting from zeros makes it *manifest* that the
        generator's output is not a function of private data, which is the
        whole basis of the post-processing argument.
        """
        container = torch.zeros(n, self.x_dim, device=self.device)
        z = torch.randn(n, self.x_dim, device=self.device)
        return self.generator.sequential(container, z, self.gen_order)

    def _private_real_grads(self, real: torch.Tensor) -> Dict[str, torch.Tensor]:
        """DP-SGD gradient of `-mean(D(real))` w.r.t. the discriminator.

        Per-sample gradients via `torch.func.vmap(grad(...))`, clipped to
        `max_grad_norm`, summed, Gaussian noise added at the clipping-norm
        sensitivity, then averaged.
        """
        disc = self.discriminator
        params = {k: v.detach() for k, v in disc.named_parameters()}
        buffers = {k: v.detach() for k, v in disc.named_buffers()}

        def loss_one(p, b, x):
            return -functional_call(disc, (p, b), (x.unsqueeze(0),)).squeeze()

        # `randomness="different"`: harmless for DECAF's deterministic critic,
        # required the moment one carries dropout (as the PacGAN critic does).
        per_sample = vmap(
            grad(loss_one), in_dims=(None, None, 0), randomness="different"
        )(params, buffers, real)

        n = real.shape[0]
        norms = _flat_norms(per_sample, n)
        scale = (self.max_grad_norm / (norms + 1e-6)).clamp(max=1.0)

        sigma = self.noise_multiplier * self.max_grad_norm
        out: Dict[str, torch.Tensor] = {}
        for name, g in per_sample.items():
            summed = (g * scale.view(-1, *([1] * (g.dim() - 1)))).sum(dim=0)
            noise = torch.normal(0.0, sigma, size=summed.shape, device=summed.device)
            out[name] = (summed + noise) / n
        return out

    def fit(self, values: np.ndarray, epsilon: float, delta: float, seed: int) -> None:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        n = len(values)
        data = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        batches_per_epoch = max(1, n // self.batch_size)
        total_d_steps = batches_per_epoch * self.max_epochs
        sample_rate = min(1.0, self.batch_size / n)

        self.noise_multiplier = find_noise_multiplier(
            target_epsilon=epsilon,
            delta=delta,
            sample_rate=sample_rate,
            steps=total_d_steps,
            # `delta` is per the caller's convention; the accountant needs a
            # positive delta, and pure-epsilon DP is not achievable here.
        )

        step = 0
        for _ in range(self.max_epochs):
            for _ in range(batches_per_epoch):
                # Poisson-style subsampling is what the accountant assumes;
                # sampling a fixed-size batch uniformly at random is the
                # standard practical stand-in.
                idx = rng.choice(n, size=self.batch_size, replace=False)
                real = data[torch.as_tensor(idx, device=self.device)]

                # ---- discriminator (the only private step) ----
                self.opt_d.zero_grad(set_to_none=False)
                fake = self._fake(real.shape[0]).detach()
                # Non-private half: d/dtheta of mean(D(fake)).
                self.discriminator(fake).mean().backward()
                # Private half: DP-SGD gradient of -mean(D(real)).
                for name, g in self._private_real_grads(real).items():
                    param = dict(self.discriminator.named_parameters())[name]
                    param.grad = param.grad + g if param.grad is not None else g
                self.opt_d.step()
                # Lipschitz constraint, standing in for the gradient penalty.
                with torch.no_grad():
                    for p in self.discriminator.parameters():
                        p.clamp_(-self.weight_clip, self.weight_clip)
                step += 1

                # ---- generator (post-processing, free) ----
                if step % self.n_critic == 0:
                    self.opt_g.zero_grad(set_to_none=False)
                    g_loss = -self.discriminator(self._fake(self.batch_size)).mean()
                    g_loss.backward()
                    self.opt_g.step()

        self.steps_taken = step
        self.spent_epsilon = spent_epsilon(
            self.noise_multiplier, sample_rate, step, delta
        )

    @torch.no_grad()
    def generate(self, n: int, biased_edges: Dict[int, List[int]]) -> np.ndarray:
        """Sample `n` rows, applying DECAF's generation-time fairness shuffling.

        Identical call into `Generator_causal.sequential` as the non-private
        baseline, so FTU/DP/CF mean exactly the same thing here.
        """
        container = torch.zeros(n, self.x_dim, device=self.device)
        z = torch.randn(n, self.x_dim, device=self.device)
        out = self.generator.sequential(
            container, z, gen_order=self.gen_order, biased_edges=biased_edges
        )
        return out.detach().cpu().numpy()
