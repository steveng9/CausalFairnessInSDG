# Causal Fairness in SDG

Applying causal-graph fairness mechanisms (à la DECAF) to graphical/marginal-based
differentially private synthetic data generators (MST, PrivBayes, AIM, MWEM-PGM, ...).

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

Net: no paper in this window builds and evaluates DECAF-style, multi-definition causal
fairness edge-surgery on PrivBayes, AIM, or MWEM-PGM. PreFair remains the only
implemented graphical/marginal fair-SDG method, and only for MST with one fairness
definition.

## Candidate graphical/marginal SDG targets

| Method | Notes | Local code available |
|---|---|---|
| MST (Marginals-MST) | McKenna et al. 2021; what PreFair targets | `../SyntheticData_MIA/reprosyn-main/src/reprosyn/methods/mbi/mst.py` |
| PrivBayes | Zhang et al. 2014 Bayes-net synthesizer | `../SyntheticData_MIA/reprosyn-main/src/reprosyn/methods/mbi/privbayes.py` (+ Cython `privBayesSelect`) |
| AIM | McKenna et al. 2022; current SOTA marginal-based DP synthesizer | not present locally — pull from [ryan112358/private-pgm](https://github.com/ryan112358/private-pgm) or the AIM repo |
| MWEM-PGM | McKenna et al. 2019; PreFair explicitly names it as extensible | check `private-pgm` |
| PrivSyn / PrivTree | Zhang et al. 2021 / 2016 | not present locally |

Given AIM is the current SOTA marginal method and PreFair explicitly calls it out as an
intended-but-untested extension target, it (alongside PrivBayes) is a strong second
target beyond MST — don't feel constrained to reuse the `reprosyn` MST/PrivBayes
implementations if `private-pgm`'s versions (which AIM is built on) prove easier to
extend consistently across methods.

## Status

Early-stage. No experiments yet. See `initial_ideas.txt` for the original brainstorm
this project grew from.
