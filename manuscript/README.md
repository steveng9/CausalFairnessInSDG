# Manuscript

Two NeurIPS 2026 workshop submissions from one source tree.

| file | what it is |
|---|---|
| `main.tex` | master document; everything shared between the two submissions |
| `workshop-ai4good.tex` | AI4Good title / venue / abstract / intro / discussion |
| `workshop-privacy.tex` | PriLOM title / venue / abstract / intro / discussion |
| `table1.tex` | Table 1, matched-pairs effects by family x mechanism |
| `refs.bib` | 22 entries, each verified against the primary source |
| `make_figs.py` | regenerates `figs/results.pdf` from `../report/data/all_runs.csv` |

## Switching submissions

One line at the top of `main.tex`:

```latex
\newcommand{\WKFILE}{workshop-ai4good}
%\newcommand{\WKFILE}{workshop-privacy}
```

Each workshop file defines exactly five macros (`\wkTitle`, `\wkVenue`,
`\wkAbstract`, `\wkIntro`, `\wkDiscussion`) and nothing else.

## Anonymity

Handled by the package option, so the real author block can stay in the source
permanently. `dblblindworkshop` prints "Anonymous Author(s)" and ignores
`\author` entirely; swap to `sglblindworkshop` for a single-blind venue or the
camera-ready.

AI4Good is confirmed **double-blind**, 2-9 pages excluding references,
non-archival. PriLOM's limit was not published at the time of writing --
check before submitting.

## Building

No system LaTeX on this box. A self-contained `tectonic` binary works (download the static musl release from
the tectonic GitHub releases page):

```bash
tectonic -X compile main.tex --outdir build
```

Current lengths (body, i.e. excluding references): AI4Good ~4.9 pages,
PriLOM ~5.9 pages.

## Figures

`figs/results.pdf` is generated: `python make_figs.py`. Three panels, one float:
(a) 2-way TVD, (b) accuracy over the majority-class baseline, (c) CF effect
against epsilon. Colour means generator family in all three panels.

**Figure 1 is a placeholder** awaiting the author's causal-pruning schematic
(`main.tex`, the `\fbox{\parbox...}` block). Target <= 0.85in tall. Note the
demographic-parity mechanism is called **DemPar** throughout, not DP; DP means
differential privacy only.

## Naming conventions used throughout

- **DP** = differential privacy. **DemPar** = demographic parity (DECAF's "DP").
- **marginals-based** (plural), never "marginal-based".
- No em dashes anywhere in the body text.
