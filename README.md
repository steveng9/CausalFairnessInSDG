# Causal Fairness in SDG

Applying causal-graph fairness mechanisms (à la DECAF) to graphical/marginal-based
differentially private synthetic data generators (MST, PrivBayes, AIM, PrivSyn), plus a
faithful DECAF baseline for comparison.

## Citation

If you use this code or build on this work, please cite (author names are withheld
for anonymous review and will be added on acceptance):

```bibtex
@misc{anonymous2026portable,
  title        = {Portable Causal Fairness Across Synthetic Data Generator Families},
  author       = {Anonymous},
  year         = {2026},
  note         = {NeurIPS 2026 AI for Good Workshop (non-archival), under review},
  howpublished = {\url{https://anonymous.4open.science/r/CausalFairnessInSDG-55EC/}}
}
```

## The idea

[DECAF](https://arxiv.org/abs/2110.12884) (van Breugel et al., NeurIPS 2021) generates
fair synthetic tabular data from a GAN whose per-variable generators are wired to match
a known causal DAG. At inference time, it achieves fairness by **removing edges** into
the outcome variable (or its Markov boundary) and substituting a "surrogate" parent
value in their place — no retraining required. It formalizes three algorithmic fairness
definitions this way: Fairness Through Unawareness (FTU), Demographic Parity (DP), and
Conditional Fairness (CF, which subsumes both).

Graphical/marginal-based DP synthesizers (MST, PrivBayes, AIM, MWEM-PGM) already build a
Bayesian-network-style graphical model over the attributes as their core mechanism —
selecting a spanning tree / junction tree of attribute pairs by mutual information,
then sampling from the resulting model. That graph is structurally the same kind of
object DECAF manipulates for fairness. This project asks: can we import DECAF's
edge-removal fairness mechanisms directly into the graph-selection step of these
DP synthesizers?

## Prior art: PreFair

[PreFair](https://www.vldb.org/pvldb/vol16/p1573-pujol.pdf) (Pujol, Gilad,
Machanavajjhala — VLDB 2023) already does part of this, and is the closest existing
work. What it does:

- Defines **justifiable fairness** for a Marginals-MST: no directed path from a
  protected attribute to an outcome attribute may avoid passing through an admissible
  attribute. This is structurally similar to DECAF's CF condition (Proposition 1),
  but for an *undirected* tree whose direction is only imposed by sampling order —
  there's no assumed causal DAG, no per-variable generator network, and no
  "surrogate value" substitution. Fairness is instead enforced by never selecting the
  offending edge into the model in the first place.
- Modifies the **graph-selection step**, not a post-hoc trained model: unfair edges are
  excluded from candidacy before the (private) MST is built, rather than being cut out
  of an already-built graph. This is the "budget isn't wasted measuring edges that get
  thrown away" benefit from the original notes in `initial_ideas.txt` — PreFair already
  captures this specific insight for MST.
- Provides two algorithms: an exponential-time optimal `Exponential-PreFair` and a
  linear-time `Greedy-PreFair` (which restricts outcome-attribute neighbors to
  admissible/other-outcome attributes, then runs a private Kruskal's).
- Proves the general fair-MST problem is NP-hard, gives RDP proofs for both algorithms.

**What PreFair does *not* do (the gap this project targets):**

1. **Only one fairness definition.** Justifiable fairness is a single, coarse,
   independence-style condition. It does not give you DECAF's finer-grained menu —
   in particular a narrow **FTU**-style cut (remove only the direct protected→outcome
   edge, leave indirect paths alone) or a **DP**-style cut (remove edges from the
   outcome's whole Markov boundary that aren't d-separated from the protected
   attribute) as distinct, selectable options with different fairness/utility
   trade-offs. PreFair's own related-work section (Sec. 8) flags "additional fairness
   definitions" as future work.
2. **Only MST is actually built and evaluated.** Section 7 of the paper ("Extensions of
   PreFair") explicitly states the propositions "apply to any Bayes network... includ
   ing PrivBayes and MWEM-PGM" but this is asserted, not implemented or tested. No
   experiments exist on PrivBayes, AIM, or MWEM-PGM.
3. **No surrogate-substitution mechanism.** PreFair enforces fairness by *never
   modeling* the correlation (hard independence once conditioned on admissible
   attributes). DECAF's surrogate `do`-value substitution is a softer mechanism that
   preserves more of the removed edge's marginal information (e.g., substituting a
   fixed or sampled value rather than blank independence) — untried in the marginal-SDG
   setting.
4. **No exploration of whether fairness constraints reduce DP noise.** The
   `initial_ideas.txt` hypothesis — that removing a dependency pre-emptively might mean
   that edge/marginal doesn't need to be privately measured at all, freeing up privacy
   budget — is adjacent to PreFair's budget-reallocation benefit but not directly
   tested as a "spend less noise because you removed the edge" effect.

A search of work from the last 3 years (below) turned up nothing that fills gaps 1–3.
**This is the open space for the paper.**

## Related work (search conducted 2026-07-29, last ~3 years)

- **PreFair** (VLDB 2023) — see above; closest prior work.
- **FairCauseSyn** (arXiv 2506.19082, June 2025) — causally fair *LLM-augmented*
  synthetic data generation. Different generative backbone (LLM, not graphical/marginal
  DP), uses total/direct/indirect/spurious effect decomposition. Worth citing for the
  fairness-taxonomy angle but not a graphical-SDG competitor.
- **FLIP** — "Achieving Hilbert-Schmidt Independence Under Rényi DP for Fair and Private
  Data Generation" (arXiv 2508.21815, Aug 2025) — transformer VAE + latent diffusion,
  HSIC/CKA-based latent disentanglement under RDP. Not graphical/marginal-based.
- **PrAda-GAN** (arXiv 2511.07997, Nov 2025) — hybrid GAN with a sparsity-regularized
  Bayes-network structure for privacy-utility tradeoff. Architecturally adjacent
  (Bayes-net + generator, similar in spirit to DECAF's per-variable generators) but not
  fairness-focused and not marginal-based DP.
- **AIM-Fair** (CVPR 2025) — uses AIM-generated synthetic data to selectively fine-tune
  a *downstream classifier* for fairness. Not about making the SDG mechanism itself
  fair; orthogonal use of AIM.
- **"Where to Intervene? Benchmarking Fairness-Aware Learning on DP Synthetic Tabular
  Data"** (arXiv 2607.07471) — benchmarks fairness-aware *downstream* learning on top of
  existing DP synthesizers (incl. AIM, MWEM-PGM). Useful as an evaluation benchmark /
  baseline reference, not a competing method.
- **Quantitative Auditing of AI Fairness with DP Synthetic Data** (arXiv 2504.21634) —
  uses DP synthetic data to *audit* fairness of other models, not to generate fair data.

### Diffusion backbones (search conducted 2026-08-12)

Relevant to the `decaf_diffusion`/`decaf_dpdiffusion` backbone work (see
`notes/2026-08-12-decaf-diffusion-backbone.md`): does anyone already put a DECAF-style
causal generator, or its edge-removal fairness mechanism, on a diffusion backbone?

- **CausalDiffTab** (arXiv 2506.14206, June 2025) — mixed-type tabular diffusion
  conditioned on a causal DAG, generating column-by-column in topological order much
  like DECAF's masked sub-networks. Targets generation **fidelity/causal-fit**, not
  fairness — no edge-removal, no surrogate substitution, no protected attribute at all.
- **TabSCM** (arXiv 2604.22337, 2026) — same causal-DAG-first decomposition: conditional
  diffusion for continuous parents-conditioned variables, gradient-boosted trees for
  categorical ones, KDE/frequency for roots. ~583x faster to train than joint diffusion
  baselines because each per-variable model is small. Also fidelity/validity-focused
  (privacy evaluated only via DCR, not a DP mechanism); no fairness notion.
- **TableDiffusion** (github.com/gianlucatruda/TableDiffusion) and **DP-TLDM**
  (arXiv 2403.07842) and **DP-FinDiff** — DP-SGD-trained diffusion models for tabular
  data. None are causal-DAG-structured; they denoise the whole row jointly (or a latent
  encoding of it), so there is no per-edge object to cut for fairness.
- **FLIP** (arXiv 2508.21815, already listed above) — transformer VAE + latent diffusion
  with HSIC/CKA independence penalties under RDP. The closest existing
  fairness+DP+diffusion combination, but fairness is an HSIC independence penalty on
  learned latents, not DECAF's causal edge-removal/surrogate-substitution mechanism, and
  there's no explicit DAG.
- **Privacy-Preserving Fair Synthetic Tabular Data** (arXiv 2503.02968) — combines
  privacy and fairness constraints, explicitly framed as unexplored territory by its own
  authors, but the backbone is WGAN-GP (not diffusion) and it doesn't use a causal DAG
  or DECAF's mechanism.

Net (diffusion): the causal-DAG-conditioned-diffusion idea itself is not novel
(CausalDiffTab, TabSCM already do it, for fidelity). DP diffusion for tabular data is
not novel (TableDiffusion, DP-TLDM, DP-FinDiff). But **DECAF's specific per-attribute
masked generator plus FTU/DP/CF edge-removal and surrogate substitution, implemented on
a diffusion backbone — DP-SGD-trained or not — has not been published.** It also isn't
a drop-in swap of a stock joint TabDDPM: fairness surgery needs a per-edge input to cut,
which only the CausalDiffTab/TabSCM-style per-attribute conditional decomposition
provides, not a model that denoises every column at once.

Net (overall): no paper in either search window builds and evaluates DECAF-style,
multi-definition causal fairness edge-surgery on PrivBayes, AIM, or MWEM-PGM, nor on a
diffusion generator. PreFair remains the only implemented graphical/marginal fair-SDG
method, and only for MST with one fairness definition.

## Implemented SDG targets

| Method | Notes | Where |
|---|---|---|
| MST (Marginals-MST) | McKenna et al. 2021; what PreFair targets | `sdg/mst.py`, vendor-adapted from `private-pgm`/reprosyn |
| PrivBayes | Zhang et al. 2014 Bayes-net synthesizer | `sdg/privbayes.py` + `sdg/_privbayes_vendor.py`, vendor-adapted from `DataResponsibly/DataSynthesizer` |
| AIM | McKenna et al. 2022; current SOTA marginal-based DP synthesizer | `sdg/aim.py` + `sdg/_aim_vendor.py`, adapted from `ryan112358/private-pgm/mechanisms/aim.py` (pairwise workload only) |
| PrivSyn | Zhang et al. 2021 | `sdg/privsyn.py` + `sdg/_privsyn_gum.py`, built from the paper's description (no importable official implementation found) |
| DECAF | Kyono et al. 2021 (the paper this whole project ports fairness mechanisms *from*) | `sdg/decaf.py`, wraps the real `decaf-synthetic-data` PyPI package directly (optional `decaf` extra) — a non-private baseline, not a DP synthesizer |
| PrivTree | Zhang et al. 2016 | not implemented — partitions the joint value domain hierarchically rather than building an attribute graph, so it doesn't fit the edge-removal fairness abstraction without a separate design pass |

All four DP methods (MST/PrivBayes/AIM/PrivSyn) share the same modular `FairnessMechanism`
hooks (`fairness/base.py`): a static `filter_candidates` pass plus an incremental
`allow_edge` check during structure search, with multi-attribute candidates (PrivBayes'
parent sets) reusing the single-edge interface via `edges_allowed`. DECAF is architecturally
different — it needs a full ground-truth causal DAG as input rather than discovering one
privately (see `data/causal_graphs.py`) and applies fairness via `select_biased_edges`
(shuffling parent contributions at generation time) instead of restricting what gets
measured.

## Environment

Development happens in the conda env named `sdg` (Python 3.10), which already has
`causal-fairness-sdg` installed editable plus `private-pgm` and (if the `decaf` extra is
installed) `decaf-synthetic-data`/`torch`/`pytorch-lightning`:

```
source <conda-base>/bin/activate sdg
pip install -e '.[dev,decaf]'   # decaf extra optional; pulls in torch + pytorch-lightning
pytest tests/ -m "not slow"     # excludes DECAF's real-GAN-training tests
```

## Status

Core SDG methods (MST, PrivBayes, AIM, PrivSyn, DECAF) and all four fairness mechanisms
(FTU/DP/CF/DF) are implemented and tested on Adult and COMPAS; `scripts/run_smoke_test.py`
runs the full grid end-to-end. See `initial_ideas.txt` for the original brainstorm this
project grew from. Beyond the original GAN backbone, DECAF now also has DP-GAN and CTGAN
backbones (`notes/2026-07-30-decaf-gan-backbones.md`) and a diffusion backbone is in
progress (`notes/2026-08-12-decaf-diffusion-backbone.md`), so the DECAF side of the grid
is becoming a backbone sweep (GAN / DP-GAN / CTGAN / diffusion / DP-diffusion) in its own
right, in addition to the graphical-method side (MST/PrivBayes/AIM/PrivSyn). Not yet done:
PrivTree, final dataset/metric/settings decisions for the paper's actual experiments, and
the DP-budget-savings research question from the original notes.
