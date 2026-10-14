# Response to review comments — 2026-09-01

Record of every reviewer comment in the 2026-09-01 draft
(`_AI4Good_Neurips_26_Workshop__Causal_Fairness_in_SDG.zip`) and how it was
handled. The `\martine{}` / `\s{}` macros for resolved items have been removed
from the source; **the untouched originals are preserved at
`manuscript/unzipped_latest/`** if any of these need to be re-litigated.

**Counts.** 55 comment macros: **35 from Martine (M1–M35 below)** and 20 from
Steven, folded into the Martine comment each clarifies except where noted.

**Status:** 32 of 35 resolved · 3 open, all in the same §4.1 (Fidelity and utility) paragraph,
which is blocked on the `rf-2026-09-01` sweep.

Four numbers in the text are still `TKTK` placeholders, defined in one block at
the top of `main.tex` and filled from `python manuscript/make_figs_v2.py`.

---

## What changed at the level of the whole paper

Three changes were made once and resolve several comments each:

1. **Numbered citations.** `\usepackage[numbers,sort&compress]{natbib}`, loaded
   *before* `neurips_2026` pulls natbib in bare. Saves roughly a line per
   related-work paragraph. (M17)
2. **Bibliography rebuilt** to one house style: a paper that appeared in a
   peer-reviewed venue is cited to that venue with volume/number/pages and the
   arXiv id dropped; a paper that has only ever been a preprint is cited as
   `arXiv preprint arXiv:NNNN.NNNNN` and nothing else. All 28 entries
   re-verified against the primary source; **none were fabricated**. Corrected:
   MST → *J. Privacy and Confidentiality* 11(3); AIM → *PVLDB* 15(11):2599–2612;
   DECAF → NeurIPS 34:22221–22233; FairCauseSyn → IEEE EMBC 2025; "Where to
   Intervene" → *PETS* 2026. Added: Census TopDown (Abowd et al. 2022), NIST
   2018 challenge.
3. **`$\dagger$` marks every differentially private generator** in Table 1,
   Figure 2 and the text, defined once in §3. (M25, and Steven's suggestion at
   #19)

---

## §2 Background and related work

**M1 — "Need to cite MST paper already here, with the first mention of MST."**
Resolved. `\citep{mckenna2021mst}` moved to the first mention, in the DECAF
paragraph.

**M2 — "What counts as 'training' in DECAF? When you change the graph, don't
you need to re-estimate the conditional probability distributions?"**
(with Steven #3: *"I believe cuts in DECAF really just ignore those conditional
probabilities … Need to check on this."*)

Resolved, and Steven's guess was checked against the source rather than
assumed. In `decaf/DECAF.py:138-142`, cutting edge $j \rightarrow i$ **randomly
permutes column $j$'s values across the batch** before they reach $X_i$'s
sub-network. So the surrogate is a draw from $X_j$'s own empirical marginal —
not uniform — and the sub-network's weights are untouched. A new paragraph in
§2 states this precisely: nothing is re-estimated; the trained conditional is
*evaluated* at a parent value drawn independently of its siblings, a *do*-style
intervention at sampling time. It also draws the consequence Martine's question
implies — that is why the cut is cheap, **and** why it can cost fidelity, since
the sub-network is asked about parent combinations it never saw in training.

**M4 — "Remove 'unusually', it sounds colloquial."** Resolved.

**M5 — "Insert citations about DP data releases … could also cite NIST."**
Resolved. §2 now cites the 2020 US Census release under DP
(`abowd2022topdown`), the NIST 2018 challenge (`nist2018challenge`), MST as its
winning entry, and `vos2024`.

**M6 / M8 — "Should you compare against the other fair-SDG methods? … and can
you explain why not?"** (with Steven #7, #9)

Resolved. *Related approaches* and *The gap* are merged into one paragraph, and
it now states the reason explicitly and briefly: only the downstream mitigators
are measured in the units we measure, so `angelozzi2026where` is the comparison
we run (§4.3); the embedding- and LLM-based generators report neither a parity
gap on a held-out real population nor a marginal-fidelity score, so putting them
on our axes would mean reimplementing them rather than comparing to them. It
adds why that is also the *right* comparison: downstream mitigation is the
alternative a publisher actually has today.

**M10 — "Was the gap worth filling? What justifies this paper?"**
(with Steven #11: *"foreshadow it here more explicitly"*)

Resolved. The gap paragraph now states the practical consequence in both
directions rather than only the scientific question: if the mechanism is
confined to DECAF, a publisher who wants a fair release must accept a
non-private GAN that our own results show nobody would deploy; if it is a
property of the factorisation, they keep the fairness lever while choosing a
synthesizer on privacy and quality.

**M12 — "The Zhangs are 2 different people, McKenna is twice the same person."**
Resolved. The citation pile is gone; each method is now named with its authors
attached — "MST and AIM of McKenna et al., PrivBayes of Zhang et al., and
PrivSyn of Zhang et al."

**M14 — "'the current tabular quality leader' — not according to your results in
Figure 2a-b."** (with Steven #15)
Resolved. Softened to "which recent work has shown to lead on several
tabular-quality benchmarks", which no longer asserts anything our own figure
contradicts.

---

## §3 Porting the mechanisms

**M16, M21, M24 — active voice; "we enforce the fairness cut."**
(with Steven #22: *"look for opportunities throughout to follow this pattern"*)
Resolved throughout §3: "We implement … and port that interface", "we enforce
the cut on paths", "we check the constraint incrementally", "we specify each
dataset's DAG by hand". The passive constructions Martine highlighted are gone.

**M17 — "Readers can't see which papers these methods come from; consider
numbered citations in square brackets."** Resolved on both counts — each method
now carries its citation at first mention in §3, and see *whole paper* item 1.

**M18 — "Are AIM's results reported anywhere? If not, omit it."**
(with Steven #20: *"just keep AIM in the list with a one-line footnote"*)
Resolved as Steven directed. The long footnote about seeds, trials and
smoke-tests is deleted; what remains is one line: "AIM appears only in the
comparison of §4.3, not in the headline sweep."

**M23 — "Need to say first what TVD is."** Resolved. Defined at first use in the
new *Fidelity, utility, and matched pairs* paragraph: half the summed absolute
difference between the two distributions, 0 = exact agreement, 1 = no shared
mass.

**M25 — "Which generators are with DP? Is it the 3 marginals-based and the ones
with DP in the name?"** Resolved, and yes — that reading was right. §3 now says
so explicitly and introduces the `$\dagger$` marker.

**M26 / M27 — "'deliberately strict' is unclear … not enough info to reproduce
the experiments."** (with Steven #28: *"a small table would be helpful"*)
Resolved with **a new Table 2**, giving all six role splits — protected $S$,
admissible $A$, and outcome $Y$ — for both datasets, with the vague prose
("deliberately strict", "two variants") deleted.

**M29 — "Do you report the 1-way anywhere?"** (with Steven #30, the long comment
asking for more fidelity metrics and a radar chart)

Resolved. This drove the largest piece of new work:

- The evaluation now computes **1-way TVD, 2-way TVD, mean Jensen–Shannon
  divergence, a range-normalized 1-way Wasserstein distance, Cramér's-V
  correlation difference**, and **AUC + macro-F1 + accuracy for three
  downstream classifiers**, plus a nearest-neighbour privacy pair.
- **New Figure 3, a single radar with one polygon per family** and six axes, as
  Steven specified (one figure, three polygons, ε=10 for the DP generators,
  averaged over datasets/splits/seeds).
- Synthetic tables are now **saved to disk** (`--synth-dir`). They were not
  saved before, which is why adding a fidelity metric required regenerating the
  sweep rather than re-reading it.

One deviation from Steven's spec, deliberately: **there is no privacy axis on
the radar.** Distance-to-closest-record is degenerate on these tables — a *real*
training record has exact duplicates inside the training set (COMPAS has 8
low-cardinality columns; the real-vs-real DCR ratio is 0.0), so the metric
measures the discretization, not the release. It is recorded per run but not
plotted. The sixth axis is **"utility kept under the cut"** instead, which is
both defensible and answers M53 directly.

**M31 — "I am not sure what conditional parity / rate balance are. Are their
values reported anywhere?"** (with Steven #32: *"make a dedicated definition of
the fairness metric … not buried in a paragraph about other metrics"*)

Resolved as Steven asked, with a **dedicated *The fairness metric* paragraph**
in §3 carrying a displayed formula for the demographic-parity gap, a worked
example (50% of men vs 17% of women → gap 0.33), and an explicit statement of
what the metric does *not* say — that it asks only whether two groups receive
the positive prediction equally often, not whether either prediction is correct.
That last sentence is what makes M43 answerable. The other three metrics are
mentioned in one clause as robustness checks rather than described as if they
were reported.

---

## §4 Results

**M33 — "Hard to tell the blue and green lines apart in Fig. 2c; add a
label/legend."** Resolved — each line in panel (c) is now labelled inline with
its family and dataset (A = Adult, C = COMPAS), and the caption says so.

**Steven #34 (no Martine comment) — "could we also add the porting graph back."**
**OPEN.** `figs/porting.pdf` is shipped in the zip but not yet placed in the
text. Deferred pending space once §4.1 is rewritten.

**M41 — "'agreeing to 0.002' — do you mean 0.055?"** Resolved. Reworded to
"agreeing to within 0.002 of each other", which is what was meant: the three
family means are 0.056 / 0.055 / 0.055.

### §4.4 Privacy and fairness

**M43 — "Eugene's paper is about accuracy on minority classes. The demographic
parity gap is something quite different. I don't think his findings transfer."**
(with Steven #42, #44)

Resolved, and the claim was wrong as written — Martine is right. §4.4 is
rewritten to state the expectation and then say precisely why it does not carry
over: DP costs the most *accuracy* on the groups that appear least often, but
the parity gap asks whether two groups receive a positive prediction equally
often regardless of correctness, and is measured on real held-out people rather
than on the noisy release. "Losing accuracy on a small group and treating that
group unequally are different failures, and differential privacy causes only the
first." The same reframing is applied in the Discussion (M55).

**M45 — "How much is that widening? Do you report it? How do you explain it?"**
Resolved, and checking it weakened the original claim, so the claim was
weakened. The widening is **0.008** on Adult marginals (0.155 at ε=1 vs 0.147 at
ε=1000) and 0.018 on COMPAS — and it **runs backwards** on COMPAS diffusion
(0.272 vs 0.336). The text now gives all these numbers and says the effect is
small and not universal, instead of asserting it as a mechanism.

**M46 — "I see the ε=10 effect for COMPAS but not for Adult?"**
(with Steven #47: *"3 of the 4 lines … worded poorly"*)

Resolved — Martine is right and the original text was wrong. Measured: the dip
at ε=10 appears on **both COMPAS lines and neither Adult line**. The claim "in
both families and both datasets" is corrected to name COMPAS explicitly. It also
turned out **not** to be unexplained, as the draft had conceded: the baseline
parity gap is itself lowest at ε=10 (0.309 marginals, 0.244 diffusion), so there
is simply less disparity present for the cut to remove. That explanation is now
in the text and the "one deviation we cannot yet explain" sentence is gone.

### §4.1 Fidelity and utility — **OPEN, blocked on the sweep**

These four are the remaining work. The *fix* is built and running; the prose is
not yet rewritten because it needs the final numbers.

**Steven #35 (no Martine comment) — "doesn't this quality metric contradict the
recent literature that diffusion has best tabular fidelity? Be more precise."**
Resolved. The ranking is now stated as a ranking *among the causal backbones,
on this metric*, followed by an explicit disclaimer that it is a 2-way TVD
result over discretized tables under a fixed causal factorisation and not a
general fidelity claim. The apparent contradiction is named and explained: the
literature's diffusion advantage (TabDDPM, CausalDiffTab) is measured on
unconstrained generators with continuous columns, and our backbones trade that
freedom away for the per-column edge the cut needs. Consistent with the M14
fix already made in §2.

**M36 — "Why do you call it a 'port'? Is it because you use the DAG from DECAF
again in the diffusion model?"** Resolved, and moved to where the term is
introduced rather than defended in the results. Martine's guess is right but
incomplete, so §3 now names all three things that are reused — DECAF's DAG, its
one-network-per-column factorisation, and its topological generation order —
and the one thing replaced: each column's generator becomes a denoiser trained
by supervised noise prediction rather than adversarially. A sentence then says
"this is what we mean by *porting*". Separately, the paper had been using
"port" in two senses — as a verb for moving the mechanism, and as a noun for
the model. The noun is now "diffusion backbone" throughout, which was the
actual source of the ambiguity.

**M37 — "What are the accuracies for train on real–test on real?"**
**OPEN in the text, but computed.** `scripts/compute_real_baselines.py` now
writes them to `report/data/real_baselines.csv`: Adult accuracy 0.818 (MLP) /
0.813 (RF) against a 0.747 majority rate, AUC 0.867; COMPAS accuracy 0.663, AUC
0.698. Figure 2b already draws them as reference lines.

**M38 — "So the accuracy becomes 0.808? Say once explicitly that it is 80%."**
**OPEN.** Will be stated in absolute terms on first mention.

**M39 — "DECAF, DP-CTGAN and DP-Diff are also below baseline; why call out only
MST and DP-GAN?"** (with Steven #40: *"redo the accuracy experiments with RF …
seems very bizarre … no reasonable explanation"*)

**OPEN in the text; diagnosed, and the fix is running.** Three findings:

1. **Martine is right that the call-out was arbitrary.** DECAF itself is
   0.039 *below* the majority baseline on Adult, under every classifier and
   every ε.
2. **RF does not fix it.** Switching the downstream model to a random forest
   makes Adult accuracy slightly *worse* (MST 0.730 vs 0.747; DP-GAN 0.623 vs
   0.690). Steven's hypothesis was tested and did not hold.
3. **The metric is the problem, not the classifier.** Adult's majority class is
   0.747 and a real-data classifier reaches only 0.818 — the entire learnable
   band is seven accuracy points wide, so good and useless generators are
   compressed together and several land below the baseline for reasons
   unrelated to quality.

The fix is to report **AUC** (floor 0.5) and **macro-F1** (floor 0.41) against
the train-on-real reference. Early results: MST/Adult AUC 0.734, PrivBayes 0.845,
diffusion 0.817, against a real-data 0.856 — nothing below the floor, and the
spread is wide. Restricting to ε=10 separately rescues DP-Diffusion (−0.122 →
+0.024 lift). Figure 2b has been rebuilt on this basis.

*A related point worth keeping:* RF **is** the better choice for the fairness
metric even though it is worse on accuracy. On Adult the MLP scores macro-F1
0.431 — it is largely predicting the majority class — while RF scores 0.605. A
classifier that rarely predicts the positive class has an artificially small
parity gap, so the new sweep uses `--classifier rf`.

---

## Tables

**M48 — "What are the n=135 etc. values?"** (with Steven #49) Resolved —
removed.

**M50 — "Can you include a column for the final accuracy, like the 'end' column
for the parity gap?"** Resolved. Table 1 is rebuilt as `gap (Δ)` and `acc (Δ)`
per dataset, so the absolute value and the paired change sit together, and each
family header now gives its no-mechanism accuracy as well as its gap.

**M51 (Table 3) — "What is the point of releasing synthetic data if the
downstream model has the same accuracy or is worse than the baseline?"**
Resolved in the caption, and this is the same underlying issue as M39. The
caption now notes that accuracy is the benchmark's own metric, that it
understates what the release carries (Adult's majority class alone scores 0.758
and a real-data classifier reaches only 0.82), and states the actual purpose:
not to beat a model trained on real data — it cannot — but to let others build
one at all without the real records leaving the building.

---

## Workshop file (AI4Good intro and discussion)

**M52 — "Include citations, e.g. the US–UK PETs Prize Challenge and V-CHAMPS."**
Resolved. The opening claim now carries the Census TopDown release, the NIST
challenge, V-CHAMPS and `vos2024`.

**M53 — "Is my impression correct that the fairness mechanism destroys utility,
so there is not much point in releasing synthetic data generated with it?"**
(with Steven #54: *"this was not my impression … which generator would we use?
These are the strong conclusions that will help an AI4Good reader see our
impact."*)

Resolved, and the impression was not correct — which means the paper was failing
to say so. Two new Discussion paragraphs, placed second and third so they land
before the contingent material:

- **"The cut is affordable."** CF costs under two accuracy points in five of six
  family × dataset cells and leaves downstream AUC at `\CFAUCKEPT` of its
  unconstrained value. A publisher is not choosing between a fair table and a
  useful one. *(That macro is still `TKTK` pending the sweep.)*
- **"So which generator should a publisher use?"** A direct answer, as Steven
  asked: **marginals-based by default** (truest on every fidelity measure, DP by
  construction, and the cut is free in the privacy accounting); **causal
  diffusion when the fairness lever has to bite** (close on fidelity, retains
  the most downstream signal, removes far more of the disparity); **avoid the
  causal GAN** (trails on both, and its DP variant fails outright). "The lever
  DECAF introduced is worth having; DECAF is not the place to get it."

**M55 — "If by 'textbook' you mean Eugene's paper, reconsider whether that
explanation applies to your fairness definition."** Resolved — the word
"textbook" is gone and the framing matches the corrected §4.4 (M43).

---

## Open items, in priority order

| # | Item | Blocker |
|---|------|---------|
| M39 | Rewrite §4.1 utility discussion on AUC / macro-F1 | final sweep numbers |
| M37, M38 | State train-on-real reference and absolute accuracies in §4.1 | same paragraph |
| S#34 | Consider restoring the porting figure | space, after §4.1 |
| — | Fill 4 `TKTK` macros in `main.tex` | final sweep numbers |
| — | Compile check | no LaTeX toolchain on this machine |

**Nothing in this draft has been compiled.** Structural checks were run in place
of a build: braces and environments balance, every `\cite` key resolves against
`refs.bib`, every `\ref` has a label, and no bib entry is orphaned. The most
likely breakage is the new role-split table, whose long attribute lists are set
in wrapping `p{}` columns to keep them inside `\textwidth`.
