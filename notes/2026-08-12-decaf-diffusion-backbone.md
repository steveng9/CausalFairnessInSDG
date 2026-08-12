# DECAF with a diffusion backbone

Built 2026-08-12. Two new `SDGMethod`s that keep DECAF's causal generator structure and
its generation-time fairness mechanism, and swap the GAN sub-network for a per-attribute
conditional DDPM.

| method | private? | epsilon in DB | what changed |
|---|---|---|---|
| `decaf` (existing) | no | NULL | published baseline, untouched |
| `decaf_dpgan`/`decaf_ctgan`/`decaf_dpctgan` (existing) | mixed | see `2026-07-30-decaf-gan-backbones.md` | GAN backbone variants |
| **`decaf_diffusion`** | no | NULL | per-attribute conditional DDPM, plain supervised training |
| **`decaf_dpdiffusion`** | **yes** | swept | the same DDPM trained under DP-SGD |

Files: `sdg/_decaf_diffusion.py`, additions to `sdg/decaf_variants.py`,
`tests/test_decaf_diffusion.py` (12 tests). `sdg/decaf.py` is untouched, same reasoning as
the GAN backbones note.

---

## 0. Why this exists — the literature gap

Prompted by asking: has anyone put DECAF's causal generator + edge-removal fairness
mechanism on a diffusion backbone? A survey (2026-08-12, see the README's "Diffusion
backbones" section) found:

- **CausalDiffTab** (arXiv 2506.14206) and **TabSCM** (arXiv 2604.22337) already generate
  tabular data column-by-column over a causal DAG with per-variable conditional diffusion
  models — but for generation fidelity, not fairness. No edge-removal, no surrogate
  substitution, no protected attribute at all.
- **TableDiffusion / DP-TLDM / DP-FinDiff** already train diffusion models for tabular
  data under DP-SGD — but none are causal-DAG-structured; they denoise the whole row (or
  a latent encoding of it) jointly, so there's no per-edge object to cut for fairness.
- **PreFair** combines causal fairness with DP, but on MST, not diffusion, and with one
  coarse fairness notion.

Net: causal-DAG diffusion (for fidelity) exists; DP diffusion (without causal structure)
exists; nobody has put DECAF's specific per-attribute masked generator plus FTU/DP/CF
edge-removal on a diffusion backbone, DP-SGD-trained or not. That's the gap this module
fills, in the same "swap the backbone, keep the mechanism" spirit as the GAN variants.

---

## 1. What is preserved, what is different

Preserved, exactly as in the GAN backbones:

- **The causal generator's masking**: `BlockCausalDenoiser.M` is the same
  parents-only-visible mask as `BlockCausalGenerator.M` in `_decaf_ctgan.py`, over the
  same one-hot column blocks (`ColumnBlocks`, reused directly — no new representation).
- **The fairness mechanism**: `biased_edges` surrogate-value substitution, applied at
  *generation* time by row-permuting the listed parent's block
  (`BlockCausalDenoiser.masked_parents`). FTU/DP/CF mean exactly what they mean for every
  other DECAF variant, and the train cache still holds — one training run serves all
  4 mechanisms x 3 role configs per (dataset, seed, epsilon).

Different, because diffusion training and GAN training are different processes:

- **Training is supervised, not adversarial.** DECAF's GAN generator produces a complete
  fake row from noise and is judged (indirectly) against real rows by a discriminator.
  The diffusion generator instead trains each column's denoiser by teacher forcing on
  real data — noise a real column's true one-hot block, condition on that same real row's
  true (masked) parent values, predict the noise. This is the standard SCM chain-rule
  factorization `p(x) = prod_j p(x_j | parents(x_j))`, trained by direct regression. No
  discriminator, no adversarial instability, no weight-clipping-vs-gradient-penalty
  tradeoff.
- **Generation costs T forward passes per column, not one.** Reverse diffusion is a loop:
  for each column in topological order, run `timesteps` reverse steps conditioned on the
  (already-fixed, possibly-shuffled) parent blocks, then move to the next column. Cheap
  here because each per-column network is a small MLP and `timesteps` defaults to 50 —
  but it is a real cost the GAN backbones don't pay.
- **Categorical representation is a simplification.** Diffusion runs on the *continuous*
  one-hot block (Gaussian noise added/removed on real-valued vectors), decoded to an
  integer code by argmax only at the very end (`ColumnBlocks.decode`) — same
  continuous-until-the-end convention `_decaf_ctgan.py` already uses. This is the
  "analog bits" / simplex-diffusion simplification, not TabDDPM's proper multinomial
  diffusion process for categoricals. A full categorical-diffusion upgrade (matching
  TabSCM's per-column model choice more closely) is a documented possible improvement,
  not done here.

---

## 2. The DP argument — genuinely simpler than DP-GAN

`_decaf_dp.py`'s DP-GAN needs three deviations from stock DECAF because only *half* the
model (the discriminator, via its real-data gradient) reads private data, and the
generator's freedom from that constraint has to be argued as post-processing:
no gradient penalty (weight clipping instead), no ADS-GAN privacy term, zero-seeded fake
generation to make the post-processing argument manifest.

None of that applies here. `decaf_dpdiffusion` has **one network and one supervised
loss** — the whole per-sample gradient of the noise-prediction MSE is clipped and noised
every step (`DDPMCausalTrainer._private_grads`, per-sample gradients via
`torch.func.vmap(grad(...))`, same primitives as `_decaf_dp.py`). There is no
discriminator to keep private-data-free, so there is no post-processing split to reason
about, no weight-clipping substitute, no disabled auxiliary loss term. Same accountant
(Opacus RDP, moments accountant with subsampling amplification) and the same
`find_noise_multiplier`/`spent_epsilon` helpers from `_decaf_dp.py`, reused directly.

One implementation wrinkle worth recording: the diffusion schedule's per-timestep
coefficients (`sqrt_alpha_bars[t]`, `sqrt_one_minus_alpha_bars[t]`) must be indexed
*outside* the `vmap`'d per-sample gradient function. Indexing a plain tensor with a
`vmap`-batched index tensor (advanced/fancy indexing) has no registered batching rule and
fails at trace time; the fix is to resolve `schedule_tensor[t]` to a plain `(n,)` tensor
of per-sample scalars first, then feed those through `vmap`'s `in_dims=0` — inside the
per-sample function they're just ordinary scalars, only ordinary arithmetic (which vmap
handles fine) is left to do.

---

## 3. Measured settings

**None yet.** Unlike the GAN backbones' tuning tables, the `settings` dicts in
`decaf_variants.py` for `decaf_diffusion`/`decaf_dpdiffusion` (100 epochs, timesteps=50,
per-dataset batch size) are an untuned starting point copied across every dataset, not a
measured choice. Run an epoch/timestep sweep the same way `epoch_sweep_all` did for the
GAN backbones (`2026-07-30-decaf-gan-backbones.md` §4) before quoting fidelity or
downstream-accuracy numbers from either method. Given the DP argument above, the
epsilon=1-collapse story documented for `decaf_dpgan` on COMPAS/Adult is exactly the
empirical question worth checking first — plain supervised DP-SGD training is a much
better-behaved optimization problem than DP-WGAN, so `decaf_dpdiffusion` may not collapse
where `decaf_dpgan` does at tight budgets. That comparison is the headline result this
backbone is for.

Verified only end-to-end correctness on a toy 4-column dataset (see
`tests/test_decaf_diffusion.py`): both methods produce in-domain synthetic data, the
train cache is reused across all four fairness mechanisms, FTU/DP/CF select different
edge sets, and the DP variant's realized epsilon spend never exceeds the requested budget
and grows as epsilon tightens.

---

## 4. Running them

Opt-in via `--methods`, so resuming an existing batch never grows its grid:

```bash
BATCH=diffusion-backbone-2026-08-12 ./scripts/launch_experiments.sh \
  --methods decaf_diffusion,decaf_dpdiffusion --seeds 0,1,2,3,4
```

Both methods are also folded into `_is_gan`/`_apply_device` in `scripts/run_experiments.py`
(renamed in a comment, not in code, to note it now covers diffusion too) so `--device
cuda` reaches their torch trainers the same way it does the GAN backbones.

## 5. Next steps

1. Epoch/timestep sweep on COMPAS and Adult (same protocol as the GAN backbones' 3-seed
   ranked sweep) before trusting any fidelity number.
2. Compare `decaf_dpdiffusion` against `decaf_dpgan` at epsilon=1 specifically — the
   question this backbone exists to answer.
3. If the simplex-diffusion categorical representation turns out to be the bottleneck
   (unlikely to matter much for binary/low-cardinality columns like COMPAS's, more likely
   for Adult's wide `native-country`), implement proper multinomial diffusion for
   categoricals per TabDDPM/TabSCM instead of the current Gaussian-on-one-hot
   simplification.
