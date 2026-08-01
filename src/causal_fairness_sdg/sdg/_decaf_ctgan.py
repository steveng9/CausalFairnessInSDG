"""CTGAN-style backbone for DECAF's causal generator.

Stock DECAF emits one unbounded scalar per column and treats ordinal category
codes as regression targets. That is a poor fit for categorical tables and is
exactly what broke on Adult in batch `overnight-2026-07-30`: code ranges span
0-1 (`income`) to 0-40 (`native-country`), the wide columns dominate the loss,
and `income` collapsed to a single class (2-way TVD 0.658 vs 0.05-0.13 for the
marginal-based synthesizers). Bounding the heads with a sigmoid rescues it
only partially.

This module replaces the *representation and training procedure* with CTGAN's
(Xu et al., NeurIPS 2019), while keeping DECAF's causal generator structure and
its generation-time fairness mechanism intact:

  - **One-hot column blocks with gumbel-softmax heads.** Each column becomes a
    block of width `domain[col]` and each sub-network emits logits over that
    block, so a category is *sampled*, not regressed. This is the piece that
    should fix Adult.
  - **Conditional vector + training-by-sampling.** A column and category are
    drawn per step (with log-frequency weighting, so rare categories are seen),
    the generator is conditioned on that choice, real samples are drawn from
    the matching stratum, and a cross-entropy term penalizes the generator for
    ignoring the condition. This is CTGAN's fix for mode collapse on imbalanced
    categoricals.
  - **PacGAN discriminator** (`pac` samples concatenated per decision) and
    **WGAN-GP**, as in CTGAN.

CTGAN's mode-specific normalization is deliberately *not* implemented: it
applies to continuous columns, and every column in this project's datasets is
already an ordinal category code.

Causal semantics are preserved at column granularity -- which is what DECAF's
DAG means in the first place, since one node is one attribute. Masking,
topological generation order, and `biased_edges` shuffling all operate on whole
blocks rather than scalars.

One documented interaction: the conditional vector is a *global* input fed to
every sub-network, so when the sampled condition column happens to be a
descendant of the column being generated, it leaks information "backwards"
along the DAG relative to stock DECAF. That is inherent to combining CTGAN's
conditioning with a causal generator; `use_conditional=False` recovers the
unconditioned ablation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnBlocks:
    """Maps columns onto contiguous one-hot spans of the transformed matrix."""

    def __init__(self, columns: Sequence[str], domain: Dict[str, int]):
        self.columns = list(columns)
        self.sizes = [int(domain[c]) for c in self.columns]
        self.starts: List[int] = []
        offset = 0
        for size in self.sizes:
            self.starts.append(offset)
            offset += size
        self.total = offset

    def span(self, col_index: int) -> Tuple[int, int]:
        start = self.starts[col_index]
        return start, start + self.sizes[col_index]

    def clip(self, codes: np.ndarray) -> np.ndarray:
        """Force codes into each column's declared `[0, domain-1]` range.

        Not merely defensive: `data/datasets.py::load_adult` ordinal-codes only
        the *object* columns, so `education-num` keeps its raw UCI 1-16 values
        while its domain is recorded as 16. Clipping reproduces exactly what
        `mbi` already does for MST/AIM/PrivSyn (it folds 16 into the 15 bin and
        leaves bin 0 empty), so this backbone sees the same data they do rather
        than a differently-corrupted version. The underlying encoding bug should
        be fixed in `datasets.py` -- see notes -- but not mid-batch.
        """
        out = np.empty_like(codes, dtype=int)
        for j, size in enumerate(self.sizes):
            out[:, j] = np.clip(codes[:, j].astype(int), 0, size - 1)
        return out

    def encode(self, codes: np.ndarray) -> np.ndarray:
        """(n, n_cols) integer codes -> (n, total) one-hot."""
        codes = self.clip(codes)
        out = np.zeros((len(codes), self.total), dtype="float32")
        for j in range(len(self.sizes)):
            out[np.arange(len(codes)), self.starts[j] + codes[:, j]] = 1.0
        return out

    def decode(self, matrix: np.ndarray) -> np.ndarray:
        """(n, total) block scores -> (n, n_cols) integer codes, by argmax."""
        out = np.zeros((len(matrix), len(self.columns)), dtype=int)
        for j in range(len(self.columns)):
            start, end = self.span(j)
            out[:, j] = matrix[:, start:end].argmax(axis=1)
        return out


class ConditionSampler:
    """CTGAN's training-by-sampling.

    Mirrors `ctgan.data_sampler.DataSampler`, reimplemented for the all-discrete
    case so the block bookkeeping stays in one place and does not depend on
    CTGAN's internal `SpanInfo` structures.

    `log_frequency` is CTGAN's key trick: the column to condition on is picked
    uniformly, but the *category* is picked with probability proportional to
    `log(count)` rather than `count`. Sampling proportional to raw frequency
    would almost never show the model a rare category; uniform sampling would
    distort the marginal it is trying to learn. The log compromise is what lets
    CTGAN fit heavily imbalanced columns such as `native-country`.
    """

    def __init__(self, codes: np.ndarray, blocks: ColumnBlocks, log_frequency: bool = True):
        self.blocks = blocks
        codes = blocks.clip(codes)
        self.n_cols = len(blocks.columns)
        self.category_probs: List[np.ndarray] = []
        self.row_index: List[List[np.ndarray]] = []

        self.marginal_probs: List[np.ndarray] = []
        self.n_rows = len(codes)
        for j, size in enumerate(blocks.sizes):
            counts = np.bincount(codes[:, j].astype(int), minlength=size).astype(float)
            weights = np.log(counts + 1.0) if log_frequency else counts.copy()
            if weights.sum() <= 0:
                weights = np.ones(size)
            self.category_probs.append(weights / weights.sum())
            self.marginal_probs.append(
                counts / counts.sum() if counts.sum() > 0 else np.ones(size) / size
            )
            self.row_index.append(
                [np.flatnonzero(codes[:, j] == k) for k in range(size)]
            )

    def _cond_vector(self, cols: np.ndarray, cats: np.ndarray) -> np.ndarray:
        cond = np.zeros((len(cols), self.blocks.total), dtype="float32")
        cond[np.arange(len(cols)), np.array(self.blocks.starts)[cols] + cats] = 1.0
        return cond

    def sample(self, batch: int, rng: np.random.Generator):
        """Training conditions: category drawn with log-frequency weighting.

        Returns (cond_vector, condition_column, condition_category).
        """
        cols = rng.integers(0, self.n_cols, size=batch)
        cats = np.array(
            [rng.choice(len(self.category_probs[c]), p=self.category_probs[c]) for c in cols]
        )
        return self._cond_vector(cols, cats), cols, cats

    def sample_original(self, batch: int, rng: np.random.Generator) -> np.ndarray:
        """Generation-time conditions: category drawn from the *observed*
        marginal, per CTGAN's `sample_original_condvec`.

        The log-frequency weighting used during training exists to balance
        gradient signal across rare categories; reusing it at generation time
        would push those rare categories into the output far above their true
        rate and skew every marginal we then measure.
        """
        cols = rng.integers(0, self.n_cols, size=batch)
        cats = np.array(
            [rng.choice(len(self.marginal_probs[c]), p=self.marginal_probs[c]) for c in cols]
        )
        return self._cond_vector(cols, cats)

    def sample_matching_rows(self, cols, cats, rng: np.random.Generator) -> np.ndarray:
        """Real row indices whose column `c` equals category `k`, so the critic
        compares like with like. Empty strata (a category present in the domain
        but absent from the data) fall back to a uniform draw."""
        out = np.empty(len(cols), dtype=int)
        for i, (c, k) in enumerate(zip(cols, cats)):
            pool = self.row_index[c][k]
            out[i] = pool[rng.integers(0, len(pool))] if len(pool) else rng.integers(
                0, self.n_rows
            )
        return out


class BlockCausalGenerator(nn.Module):
    """DECAF's `Generator_causal`, lifted from scalars to one-hot column blocks.

    Structurally identical to upstream: one input projection and one output
    head per node, a shared trunk between them, a fixed parent mask, and
    sequential generation in topological order with optional parent shuffling.
    The differences are all consequences of a node now being a block:

      - the mask is expanded from (n_cols, n_cols) to (total, n_cols), so
        masking a parent zeroes its whole one-hot block;
      - each head emits `domain[col]` logits and is sampled with
        gumbel-softmax rather than emitting one unbounded scalar;
      - `biased_edges` permutes whole parent blocks, keeping each shuffled
        row's one-hot encoding internally consistent.
    """

    def __init__(
        self,
        blocks: ColumnBlocks,
        dag_seed: Sequence[Sequence[int]],
        h_dim: int = 200,
        z_dim: int = 32,
        cond_dim: int = 0,
        tau: float = 0.2,
        f_scale: float = 0.1,
    ):
        super().__init__()
        self.blocks = blocks
        self.n_cols = len(blocks.columns)
        self.total = blocks.total
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.tau = tau

        # mask[:, j] selects the *input* dimensions visible to column j, i.e.
        # the one-hot blocks of j's parents.
        mask = torch.zeros(self.total, self.n_cols)
        for parent, child in dag_seed:
            start, end = blocks.span(int(parent))
            mask[start:end, int(child)] = 1.0
        self.register_buffer("M", mask)

        def block_layers(in_feat: int, out_feat: int) -> list:
            return [nn.Linear(in_feat, out_feat), nn.ReLU(inplace=True)]

        self.shared = nn.Sequential(
            *block_layers(h_dim, h_dim), *block_layers(h_dim, h_dim)
        )
        in_dim = self.total + z_dim + cond_dim
        self.fc_i = nn.ModuleList([nn.Linear(in_dim, h_dim) for _ in range(self.n_cols)])
        self.fc_f = nn.ModuleList(
            [nn.Linear(h_dim, size) for size in blocks.sizes]
        )

        for layer in list(self.fc_i) + list(self.fc_f):
            nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale

    def forward(
        self,
        z: torch.Tensor,
        gen_order: Sequence[int],
        cond: Optional[torch.Tensor] = None,
        biased_edges: Optional[Dict[int, List[int]]] = None,
        hard: bool = False,
    ) -> torch.Tensor:
        biased_edges = biased_edges or {}
        n = z.shape[0]
        out = torch.zeros(n, self.total, device=z.device)

        for j in gen_order:
            masked = out * self.M[:, j]
            # Fairness: substitute surrogate values for the listed parents by
            # permuting their rows -- DECAF's mechanism, applied blockwise so a
            # shuffled parent stays a valid one-hot row.
            for parent in biased_edges.get(j, []):
                start, end = self.blocks.span(parent)
                perm = torch.randperm(n, device=z.device)
                masked[:, start:end] = masked[perm, start:end]

            parts = [masked, z]
            if cond is not None:
                parts.append(cond)
            hidden = F.relu(self.fc_i[j](torch.cat(parts, dim=1)))
            logits = self.fc_f[j](self.shared(hidden))
            start, end = self.blocks.span(j)
            # Clone-free in-place writes would break autograd through `out`,
            # so rebuild the row instead.
            sampled = F.gumbel_softmax(logits, tau=self.tau, hard=hard)
            out = out.clone()
            out[:, start:end] = sampled
        return out


class PacDiscriminator(nn.Module):
    """CTGAN's PacGAN critic: judges `pac` rows jointly, which is what stops
    the generator from winning by emitting one plausible row over and over."""

    def __init__(self, input_dim: int, h_dim: int = 256, pac: int = 10):
        super().__init__()
        self.pac = pac
        self.model = nn.Sequential(
            nn.Linear(input_dim * pac, h_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(h_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(-1, x.shape[1] * self.pac))


def topological_order(dag_seed: Sequence[Sequence[int]], n_cols: int) -> List[int]:
    g = nx.DiGraph()
    g.add_nodes_from(range(n_cols))
    g.add_edges_from((int(p), int(c)) for p, c in dag_seed)
    return list(nx.algorithms.dag.topological_sort(g))


class CTGANCausalTrainer:
    """WGAN-GP training of the block causal generator, CTGAN-style."""

    def __init__(
        self,
        blocks: ColumnBlocks,
        dag_seed: Sequence[Sequence[int]],
        h_dim: int = 200,
        z_dim: int = 32,
        lr: float = 2e-4,
        batch_size: int = 500,
        max_epochs: int = 100,
        pac: int = 10,
        lambda_gp: float = 10.0,
        use_conditional: bool = True,
        log_frequency: bool = True,
        device: str = "cpu",
        dp_max_grad_norm: float = 1.0,
        dp_weight_clip: float = 0.01,
        dp_n_critic: int = 2,
    ):
        self.blocks = blocks
        self.device = torch.device(device)
        # Only used by `fit_dp`; measured on the non-CTGAN DP backbone, where
        # weight_clip=0.1 collapsed the generator outright and 0.01 did not.
        self.dp_max_grad_norm = dp_max_grad_norm
        self.dp_weight_clip = dp_weight_clip
        self.dp_n_critic = dp_n_critic
        self.noise_multiplier: Optional[float] = None
        self.steps_taken = 0
        self.batch_size = batch_size - (batch_size % pac)  # PacGAN needs a multiple
        self.max_epochs = max_epochs
        self.pac = pac
        self.lambda_gp = lambda_gp
        self.use_conditional = use_conditional
        self.log_frequency = log_frequency
        self.z_dim = z_dim

        cond_dim = blocks.total if use_conditional else 0
        self.generator = BlockCausalGenerator(
            blocks, dag_seed, h_dim=h_dim, z_dim=z_dim, cond_dim=cond_dim
        ).to(self.device)
        disc_dim = blocks.total + cond_dim
        self.discriminator = PacDiscriminator(disc_dim, h_dim=h_dim, pac=pac).to(
            self.device
        )
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=1e-6
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=1e-6
        )
        self.gen_order = topological_order(dag_seed, len(blocks.columns))
        self.sampler: Optional[ConditionSampler] = None

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        n = real.shape[0] // self.pac
        alpha = torch.rand(n, 1, 1, device=self.device).repeat(1, self.pac, real.shape[1])
        alpha = alpha.view(-1, real.shape[1])
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        scores = self.discriminator(interp)
        grads = torch.autograd.grad(
            outputs=scores,
            inputs=interp,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            only_inputs=True,
        )[0]
        grads = grads.view(n, -1)
        return ((grads.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp

    def _cond_loss(self, fake, cond_cols, cond_cats) -> torch.Tensor:
        """Cross-entropy penalty for generating a different category than the
        one the generator was conditioned on."""
        losses = []
        for j in range(len(self.blocks.columns)):
            mask = cond_cols == j
            if not mask.any():
                continue
            start, end = self.blocks.span(j)
            logits = torch.log(fake[mask, start:end] + 1e-8)
            target = torch.as_tensor(cond_cats[mask], device=self.device, dtype=torch.long)
            losses.append(F.nll_loss(logits, target))
        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    def fit(self, codes: np.ndarray, seed: int = 0) -> None:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        encoded = self.blocks.encode(codes)
        data = torch.as_tensor(encoded, device=self.device)
        self.sampler = ConditionSampler(codes, self.blocks, self.log_frequency)

        n = len(codes)
        steps_per_epoch = max(1, n // self.batch_size)
        for _ in range(self.max_epochs):
            for _ in range(steps_per_epoch):
                cond_t, cond_cols, cond_cats, real = self._draw(data, n, rng)

                # ---- critic ----
                fake = self.generator(
                    self._z(self.batch_size), self.gen_order, cond=cond_t
                )
                real_in = real if cond_t is None else torch.cat([real, cond_t], dim=1)
                fake_in = fake if cond_t is None else torch.cat([fake, cond_t], dim=1)
                d_loss = (
                    self.discriminator(fake_in.detach()).mean()
                    - self.discriminator(real_in).mean()
                    + self._gradient_penalty(real_in, fake_in.detach())
                )
                self.opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                self.opt_d.step()

                # ---- generator ----
                fake = self.generator(
                    self._z(self.batch_size), self.gen_order, cond=cond_t
                )
                fake_in = fake if cond_t is None else torch.cat([fake, cond_t], dim=1)
                g_loss = -self.discriminator(fake_in).mean()
                if self.use_conditional:
                    g_loss = g_loss + self._cond_loss(fake, cond_cols, cond_cats)
                self.opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                self.opt_g.step()

    def fit_dp(
        self, codes: np.ndarray, epsilon: float, delta: float, seed: int = 0
    ) -> None:
        """Train under DP-SGD instead of plain WGAN-GP.

        Same privacy argument as `_decaf_dp.DPCausalGANTrainer` -- only the
        critic's real-data gradient is privatized, everything else is
        post-processing -- with two extra deviations that CTGAN's specific
        machinery forces:

        1. **No gradient penalty**, replaced by weight clipping, because the
           penalty is computed on real/fake interpolations (as in `_decaf_dp`).
        2. **No conditional vectors / training-by-sampling.** Both the
           log-frequency category weights and the "draw a real row matching
           this condition" step read private counts on a path that is not
           privatized, and the conditioned batch is no longer a uniform
           subsample, which would also invalidate the accountant's
           amplification-by-subsampling assumption. Disabling conditioning is
           the rigorous option; the better one -- spending a slice of the
           budget on noisy 1-way marginals and conditioning on *those* -- is
           left as a documented improvement rather than done hastily here.

        Consequence: `decaf_dpctgan` keeps CTGAN's one-hot/gumbel-softmax
        representation (the part that fixed Adult's fidelity) but loses its
        imbalanced-category handling.
        """
        from ._decaf_dp import find_noise_multiplier, spent_epsilon

        if self.use_conditional:
            raise ValueError(
                "fit_dp requires use_conditional=False -- see the docstring; "
                "conditional sampling reads unprivatized private counts."
            )

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        data = torch.as_tensor(self.blocks.encode(codes), device=self.device)
        n = len(codes)

        steps_per_epoch = max(1, n // self.batch_size)
        total_steps = steps_per_epoch * self.max_epochs
        sample_rate = min(1.0, self.batch_size / n)
        self.noise_multiplier = find_noise_multiplier(
            target_epsilon=epsilon, delta=delta,
            sample_rate=sample_rate, steps=total_steps,
        )

        step = 0
        for _ in range(self.max_epochs):
            for _ in range(steps_per_epoch):
                idx = rng.choice(n, size=self.batch_size, replace=False)
                real = data[torch.as_tensor(idx, device=self.device)]

                # ---- critic: private ----
                self.opt_d.zero_grad(set_to_none=False)
                fake = self.generator(
                    self._z(self.batch_size), self.gen_order
                ).detach()
                self.discriminator(fake).mean().backward()
                for name, g in self._private_real_grads(real).items():
                    param = dict(self.discriminator.named_parameters())[name]
                    param.grad = param.grad + g if param.grad is not None else g
                self.opt_d.step()
                with torch.no_grad():
                    for p in self.discriminator.parameters():
                        p.clamp_(-self.dp_weight_clip, self.dp_weight_clip)
                step += 1

                # ---- generator: post-processing ----
                if step % self.dp_n_critic == 0:
                    self.opt_g.zero_grad(set_to_none=False)
                    fake = self.generator(self._z(self.batch_size), self.gen_order)
                    (-self.discriminator(fake).mean()).backward()
                    self.opt_g.step()

        self.steps_taken = step
        self.spent_epsilon = spent_epsilon(
            self.noise_multiplier, sample_rate, step, delta
        )

    def _private_real_grads(self, real: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-sample clipped + noised gradient of `-mean(D(real))`."""
        from torch.func import functional_call, grad, vmap

        disc = self.discriminator
        params = {k: v.detach() for k, v in disc.named_parameters()}
        buffers = {k: v.detach() for k, v in disc.named_buffers()}

        def loss_one(p, b, x):
            return -functional_call(disc, (p, b), (x.unsqueeze(0),)).squeeze()

        # `randomness="different"` because the PacGAN critic contains dropout:
        # each sample must draw its own mask, which is what an ordinary batched
        # forward pass does anyway. vmap's default mode errors out rather than
        # silently sharing one mask across the batch.
        per_sample = vmap(
            grad(loss_one), in_dims=(None, None, 0), randomness="different"
        )(params, buffers, real)

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

    def _z(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.z_dim, device=self.device)

    def _draw(self, data, n, rng):
        if not self.use_conditional:
            idx = rng.integers(0, n, size=self.batch_size)
            return None, None, None, data[torch.as_tensor(idx, device=self.device)]
        cond, cols, cats = self.sampler.sample(self.batch_size, rng)
        idx = self.sampler.sample_matching_rows(cols, cats, rng)
        return (
            torch.as_tensor(cond, device=self.device),
            cols,
            cats,
            data[torch.as_tensor(idx, device=self.device)],
        )

    @torch.no_grad()
    def generate(
        self, n: int, biased_edges: Optional[Dict[int, List[int]]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample `n` rows as integer codes, with DECAF's fairness shuffling."""
        rng = rng or np.random.default_rng(0)
        chunks = []
        remaining = n
        while remaining > 0:
            size = min(self.batch_size, remaining)
            cond = None
            if self.use_conditional:
                cond = torch.as_tensor(
                    self.sampler.sample_original(size, rng), device=self.device
                )
            out = self.generator(
                self._z(size), self.gen_order, cond=cond,
                biased_edges=biased_edges, hard=True,
            )
            chunks.append(out.cpu().numpy())
            remaining -= size
        return self.blocks.decode(np.concatenate(chunks)[:n])
