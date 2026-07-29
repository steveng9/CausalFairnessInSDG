"""Fairness metrics for a downstream classifier's predictions, mirroring
PreFair Table 2 (DP, TPRB, TNRB, and their admissible-attribute-conditioned
variants CDP/CTPRB/CTNRB).

Which dataset these get computed *against* -- the real/original distribution
vs. the synthetic data's own distribution -- is DECAF's Distributional
Fairness (DF) axis (`fairness.df.EvalReference`). These functions are
reference-distribution-agnostic: the caller decides by choosing which
dataframe of (protected, prediction, label, admissible) columns to pass in.

DP/TPRB/TNRB are defined here for a single *binary* protected attribute, per
PreFair's own formulation. For multi-category protected attributes (e.g.
Adult's `race`, `native-country`), binarize into a disadvantaged-vs-rest
indicator first -- see `binarize_protected` -- following DECAF/PreFair's own
practice of picking one disadvantaged group for these metrics.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def binarize_protected(series: pd.Series, disadvantaged_value) -> pd.Series:
    """1 for the chosen disadvantaged group, 0 for everyone else."""
    return (series == disadvantaged_value).astype(int)


def _rate_diff(
    df: pd.DataFrame,
    protected_col: str,
    mask: pd.Series,
    outcome_col: str,
    target_value: int,
) -> float:
    d = df[mask]
    grp1 = d[d[protected_col] == 1][outcome_col]
    grp0 = d[d[protected_col] == 0][outcome_col]
    if len(grp1) == 0 or len(grp0) == 0:
        return float("nan")
    p1 = float((grp1 == target_value).mean())
    p0 = float((grp0 == target_value).mean())
    return p1 - p0


def demographic_parity_gap(df: pd.DataFrame, protected_col: str, predicted_col: str) -> float:
    """DP: Pr(O=1|S=1) - Pr(O=1|S=0)."""
    return _rate_diff(df, protected_col, pd.Series(True, index=df.index), predicted_col, 1)


def true_positive_rate_balance(
    df: pd.DataFrame, protected_col: str, predicted_col: str, label_col: str
) -> float:
    """TPRB: Pr(O=1|S=1,Y=1) - Pr(O=1|S=0,Y=1)."""
    return _rate_diff(df, protected_col, df[label_col] == 1, predicted_col, 1)


def true_negative_rate_balance(
    df: pd.DataFrame, protected_col: str, predicted_col: str, label_col: str
) -> float:
    """TNRB: Pr(O=0|S=1,Y=0) - Pr(O=0|S=0,Y=0)."""
    return _rate_diff(df, protected_col, df[label_col] == 0, predicted_col, 0)


def _conditional_average(
    df: pd.DataFrame,
    protected_col: str,
    predicted_col: str,
    admissible_cols: List[str],
    metric: str,
    label_col: str = None,
) -> float:
    """E_A[metric | A=a], averaged over admissible-attribute value
    combinations, weighted by group size (CDP/CTPRB/CTNRB)."""
    gaps, weights = [], []
    for _, g in df.groupby(admissible_cols):
        if metric == "dp":
            gap = demographic_parity_gap(g, protected_col, predicted_col)
        elif metric == "tprb":
            gap = true_positive_rate_balance(g, protected_col, predicted_col, label_col)
        elif metric == "tnrb":
            gap = true_negative_rate_balance(g, protected_col, predicted_col, label_col)
        else:
            raise ValueError(f"Unknown metric {metric!r}")
        if not np.isnan(gap):
            gaps.append(gap)
            weights.append(len(g))
    if not gaps:
        return float("nan")
    return float(np.average(gaps, weights=weights))


def compute_fairness_metrics(
    df: pd.DataFrame,
    protected_attrs: Iterable[str],
    predicted_col: str,
    label_col: str,
    admissible_attrs: Iterable[str],
) -> Dict[str, float]:
    """DP/TPRB/TNRB and their conditional (C-prefixed) variants for each
    protected attribute in `protected_attrs`. Each protected column in `df`
    must already be binary (0/1) -- see `binarize_protected`.
    """
    admissible_attrs = list(admissible_attrs)
    metrics: Dict[str, float] = {}
    for s in protected_attrs:
        metrics[f"dp_gap__{s}"] = demographic_parity_gap(df, s, predicted_col)
        metrics[f"tprb__{s}"] = true_positive_rate_balance(df, s, predicted_col, label_col)
        metrics[f"tnrb__{s}"] = true_negative_rate_balance(df, s, predicted_col, label_col)
        if admissible_attrs:
            metrics[f"cdp_gap__{s}"] = _conditional_average(
                df, s, predicted_col, admissible_attrs, "dp"
            )
            metrics[f"ctprb__{s}"] = _conditional_average(
                df, s, predicted_col, admissible_attrs, "tprb", label_col=label_col
            )
            metrics[f"ctnrb__{s}"] = _conditional_average(
                df, s, predicted_col, admissible_attrs, "tnrb", label_col=label_col
            )
    return metrics
