# Manuscript

One paper, two workshop submissions. 4 pages, NeurIPS 2026 workshop template.

## Files

| File | What it is |
|---|---|
| `main.tex` | **The master.** Everything shared between both submissions. |
| `workshop-ai4good.tex` | AI4Good variant: title, venue, abstract, intro, discussion. |
| `workshop-privacy.tex` | "Privacy in the Era of Large Opaque Models" variant, same five macros. |
| `table1.tex` | Table 1 (matched-pairs fairness results). Shared. |
| `refs.bib` | Bibliography. Every entry checked against the primary source 2026-08-22. |
| `figs/*.pdf` | Figures, produced by `make_figs.py`. |
| `make_figs.py` | Regenerates every figure from `../report/data/all_runs.csv`. |

## Switching workshops

Top of `main.tex`, comment/uncomment one line:

```latex
\newcommand{\WKFILE}{workshop-ai4good}
%\newcommand{\WKFILE}{workshop-privacy}
```

Each workshop file defines exactly five macros — `\wkTitle`, `\wkVenue`,
`\wkAbstract`, `\wkIntro`, `\wkDiscussion` — and nothing else. To retune a
submission, edit only that file.

Both are set up as `[dblblindworkshop]`. If either turns out to be single-blind,
swap the package option in `main.tex` and fill in the `\author` block.

## Building

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

(There is no TeX installation on this machine — the .tex has not been compiled.)

## Figures

```
python make_figs.py     # writes figs/fidelity.pdf and figs/fairness.pdf
```

Reads `../report/data/all_runs.csv`, filters `superseded == False` (2,520 active
runs), and prints the Table 1 numbers to stdout so the table can be re-checked
against the data.

Figure 1 is a **placeholder** — the causal-pruning schematic goes there.
