# DECAF with DP-GAN and CTGAN backbones

Built 2026-07-30. Three new `SDGMethod`s that keep DECAF's causal generator and its
generation-time fairness mechanism, and swap only *how the generator is trained or
parameterized*.

| method | private? | epsilon in DB | what changed |
|---|---|---|---|
| `decaf` (existing) | no | NULL | published baseline, untouched |
| **`decaf_dpgan`** | **yes** | swept | stock causal generator, critic trained with DP-SGD |
| **`decaf_ctgan`** | no | NULL | CTGAN representation + training procedure |
| **`decaf_dpctgan`** | **yes** | swept | both |

Files: `sdg/_decaf_dp.py`, `sdg/_decaf_ctgan.py`, `sdg/decaf_variants.py`,
`tests/test_decaf_variants.py` (15 tests). `sdg/decaf.py` is **deliberately untouched**
so the published-DECAF rows in `experiments.db` stay reproducible.

Why these two: `decaf_dpgan` gives DECAF a real epsilon, so it finally enters the same
privacy sweep as MST/PrivBayes/PrivSyn instead of sitting outside as a non-private
reference. `decaf_ctgan` targets the fidelity failure documented in
`2026-07-30-overnight-batch.md` §9.1 (Adult 2-way TVD 0.658).

---

## 1. What is preserved

Both backbones keep DECAF's actual contribution intact:

- **The causal generator**: one sub-network per attribute, masked so it sees only its
  parents in the ground-truth DAG, generated in topological order.
- **The fairness mechanism**: `biased_edges` surrogate-value substitution, applied at
  *generation* time by shuffling the listed parent columns. FTU/DP/CF therefore mean
  exactly what they mean for the baseline, and the train cache still holds — one
  training serves all 12 (role config × mechanism) cells per seed.

`decaf_ctgan` lifts the masking from scalars to one-hot **blocks**, which is what the
DAG meant in the first place (one node = one attribute). Shuffling permutes whole
blocks, so a surrogate parent stays a valid one-hot row.

---

## 2. DP-GAN: the privacy argument

WGAN critic loss is `mean(D(fake)) - mean(D(real))`.

- `mean(D(fake))` is a function of generator parameters and fresh noise → **free**.
- `-mean(D(real))` reads private data → privatized with per-sample gradient clipping to
  `max_grad_norm` plus Gaussian noise (DP-SGD, Abadi et al. 2016). Per-sample gradients
  come from `torch.func.vmap(grad(...))`.
- The generator only ever sees private data *through* the critic's gradients → its
  updates are **post-processing**, and cost nothing.

Accounting: Opacus RDP accountant (moments accountant with amplification by
subsampling), binary-searching the smallest noise multiplier that spends exactly the
requested (ε, δ) over the planned number of critic steps. Note this is a **different
accountant** from the zCDP path MST/AIM/PrivSyn use (`_cdp2adp.py`); both yield an
(ε, δ)-DP guarantee so the epsilons are comparable in the usual sense, but the
intermediate ρ is not. Realized spend, noise multiplier and step count are logged as
metrics (`noise_multiplier`, `dp_steps`, `spent_epsilon`) so the calibration behind a
row is recoverable from the DB.

### Three deviations DP forces on stock DECAF

1. **No gradient penalty.** WGAN-GP's penalty is computed on interpolations *between
   real and fake* samples, so it reads private data on a path per-sample clipping can't
   cover without a double-backward through the per-sample machinery. Replaced with the
   original WGAN's **weight clipping**, which is what the DP-WGAN literature does for the
   same reason. Measured: `weight_clip=0.01` works, `0.1` collapses the generator
   outright (7 constant columns, TVD1 0.39 vs 0.09).
2. **No ADS-GAN privacy term.** Stock DECAF adds
   `lambda_privacy * privacy_loss(batch, fake)` to the *generator* loss, reading the real
   batch directly in a gradient that is never privatized. Leaving it on would void the
   guarantee. Disabled.
3. **Fake samples seeded from zeros, not a real batch.** With a full topological order
   every column is overwritten before it is read as anyone's parent, so DECAF's habit of
   seeding `sequential()` from real rows is already a no-op — but starting from zeros
   makes it *manifest* that generator output isn't a function of private data, which is
   the whole basis of the post-processing argument.

### And one more for `decaf_dpctgan`

**Conditional vectors / training-by-sampling are disabled** under DP. Both the
log-frequency category weights and the "draw a real row matching this condition" step
read private counts on an unprivatized path, and a conditioned batch is no longer a
uniform subsample — which would also invalidate the accountant's amplification
assumption. `fit_dp` raises if `use_conditional=True` rather than silently emitting an
unsound guarantee.

PacGAN is also disabled (`pac=1`): packing `pac` records into one critic decision makes
a "sample" a group of records, so the sensitivity analysis no longer matches a
per-record neighbouring-datasets definition.

So `decaf_dpctgan` keeps CTGAN's representation (the part that fixes fidelity) but loses
its imbalanced-category handling. **The better fix — spend a slice of the budget on noisy
1-way marginals and condition on those — is the obvious next improvement** and was left
undone rather than done hastily.

---

## 3. CTGAN backbone: what was and wasn't implemented

Implemented, following Xu et al. (NeurIPS 2019):

- one-hot column blocks with **gumbel-softmax** heads (τ=0.2) — categories are *sampled*,
  not regressed. This is the piece that fixes Adult.
- **conditional vector + training-by-sampling** with log-frequency category weighting, so
  rare categories are actually seen; cross-entropy penalty for ignoring the condition.
  At *generation* time the condition is drawn from the observed marginal instead
  (CTGAN's `sample_original_condvec`) — reusing the log weighting there would push rare
  categories far above their true rate and skew every marginal we then measure.
- **PacGAN** critic (pac=10) and **WGAN-GP**.

Not implemented: **mode-specific normalization**. It applies to continuous columns, and
every column in these datasets is already an ordinal category code.

Documented interaction: the conditional vector is a *global* input fed to every
sub-network, so when the sampled condition column happens to be a descendant of the
column being generated, information flows "backwards" along the DAG relative to stock
DECAF. Inherent to combining CTGAN's conditioning with a causal generator;
`use_conditional=False` recovers the unconditioned ablation.

---

## 4. Measured settings (why they're per dataset)

Tuned at ε=1000 for the DP variants, so the epoch count reflects what the *optimizer*
needs; the privacy cost of those steps is then paid honestly at every ε.

**`decaf_ctgan`, COMPAS** — fidelity degrades past ~20 epochs, so it is early-stopped:

| epochs | TVD-1 | TVD-2 | downstream acc |
|---|---|---|---|
| **20** | **0.042** | **0.078** | 0.551 |
| 30 | 0.050 | 0.083 | 0.571 |
| 60 | 0.091 | 0.154 | 0.543 |

At 20 epochs it **beats the published DECAF baseline's fidelity** on COMPAS
(TVD-2 0.078 vs 0.084) while giving up downstream accuracy (0.551 vs 0.659). A real
tradeoff, not a strict improvement.

**`decaf_ctgan`, Adult** — the headline result. Stock DECAF: TVD-1 0.395 / TVD-2 0.658.
CTGAN backbone at 30 epochs: **TVD-1 0.090 / TVD-2 0.196**, accuracy 0.698 (vs baseline
0.695). That is a **3.4× improvement in 2-way TVD at equal accuracy** — the Adult
fidelity problem is fixed.

**`decaf_dpgan`, COMPAS** (at ε=1000, negligible noise):

| epochs | TVD-1 | TVD-2 | acc | note |
|---|---|---|---|---|
| **100** | **0.130** | **0.206** | 0.625 | chosen |
| 200 | 0.160 | 0.239 | 0.642 | more steps = more noise under DP, worse fidelity |

Epoch counts are **held fixed across ε on purpose**. Spending fewer steps at a tight
budget would produce better numbers at ε=1, but it confounds "what does privacy cost
this architecture" with "how well did we tune each cell".

### The honest consequence

**DP-GAN is not viable at ε=1 on COMPAS.** At 100 epochs the noise multiplier reaches
σ≈11.9 and the generator collapses to a single class with 8 constant columns
(TVD-1 0.528). At 200 epochs, σ≈16.8 and it is worse still (TVD-1 0.783). This is a
result about DP-GANs on small tables — consistent with the literature that marginal-based
DP synthesizers dominate DP-GANs at tight budgets — not a configuration to paper over.
Expect `decaf_dpgan` and `decaf_dpctgan` rows at ε=1 to come back `partial`.

---

## 5. Bug found while building this

`data/datasets.py::load_adult` mis-encodes `education-num`. See
`2026-07-30-overnight-batch.md` §9.3. The new backbones reproduce mbi's clipping exactly
(`ColumnBlocks.clip`) so they see the same data as every other method rather than a
differently-corrupted version.

---

## 6. Running them

Opt-in via `--methods`, so resuming an existing batch never grows its grid:

```bash
BATCH=gan-backbones-2026-07-30 ./scripts/launch_experiments.sh \
  --methods decaf_dpgan,decaf_ctgan,decaf_dpctgan --seeds 0,1,2,3,4
```

840 cells (2 datasets × 3 role configs × 4 mechanisms × 5 seeds ×
[3 ε for the two DP variants, 1 for `decaf_ctgan`]).
