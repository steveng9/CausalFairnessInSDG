"""DECAF's causal generator with no generator network at all.

The third backbone swap, after `_decaf_ctgan.py` and `_decaf_diffusion.py`.
Those two keep DECAF's structure -- one conditional per attribute, restricted
to its DAG parents, sampled in topological order, with the fairness mechanism
acting on the parents' *conditioning* at generation time (`biased_edges`
surrogate-value substitution) -- and swap how each conditional is trained.
Here the conditional is not trained: it is read off a conditional-distribution
estimator, `p(x_j | parents(x_j))`, fitted on the real rows. With a tabular
foundation model as that estimator (`decaf_tabpfn.py`) the real table is the
model's context and nothing is optimised.

This module is the part that does not care which estimator that is. It owns
the DECAF semantics -- generation order, root marginals, the edge cut, the
sampling -- and asks the estimator for one thing: class probabilities for a
batch of parent configurations. `tests/test_decaf_incontext.py` drives it with
a counting estimator and never imports a foundation model.

Why probabilities and not samples: every column here is a category code, so
`p(x_j | parents)` is a finite table indexed by the parent configuration, and
the estimator only has to be asked about each *distinct* configuration once.
A column whose parents span few configurations is tabulated in full when the
sampler is fitted (`table_cap`); after that, generating under any fairness
mechanism and any role split is a table lookup. A column whose parents span
too many (in practice the outcome, with 10-15 parents) keeps its fitted
estimator and is asked only for configurations it has not seen, the answers
being remembered across generations. Either way a configuration always gets
the same probabilities, so the runs under `none`, FTU, DP and CF differ only
by the cut and not by estimator noise.

The cut is DECAF's, unchanged: when sampling column j, each parent listed in
`biased_edges[j]` has its values row-permuted before the conditional is read,
which keeps that parent's marginal and severs its dependence on the row. That
can produce parent configurations the real data never contained; the
estimator is asked about them like any other.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from ._decaf_ctgan import topological_order


class FittedConditional(Protocol):
    """One column's `p(child | parents)`, ready to be queried."""

    def predict_proba(self, parents: np.ndarray) -> np.ndarray:
        """`(m, n_parents)` parent codes -> `(m, n_classes)` probabilities,
        one column per child code `0..n_classes-1`."""

    def release(self) -> None:
        """Free whatever the fit is holding (GPU memory, caches)."""


class ConditionalEstimator(Protocol):
    def fit(
        self, parents: np.ndarray, child: np.ndarray, n_classes: int
    ) -> FittedConditional:
        """Fit `p(child | parents)` on real rows."""


def _normalised(probs: np.ndarray, n_classes: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != n_classes:
        raise ValueError(
            f"estimator returned probabilities of shape {probs.shape}, "
            f"expected (m, {n_classes})"
        )
    if not np.isfinite(probs).all():
        raise ValueError("estimator returned non-finite probabilities")
    probs = np.clip(probs, 0.0, None)
    totals = probs.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("estimator returned an all-zero probability row")
    return probs / totals


class _Column:
    """`p(column | parents)` addressed by parent configuration.

    A configuration is one integer, the parents' codes read as a mixed-radix
    number (`np.ravel_multi_index`). Probabilities for configurations asked
    about so far are kept sorted by that integer; a tabulated column simply
    starts with all of them.
    """

    def __init__(self, parents: List[int], radix: List[int], n_classes: int):
        self.parents = parents
        self.radix = tuple(radix)
        self.n_classes = n_classes
        self.keys = np.empty(0, dtype=np.int64)
        self.probs = np.empty((0, n_classes), dtype=np.float64)
        self.fitted: Optional[FittedConditional] = None
        self.constant: Optional[int] = None
        self.queried = 0

    @property
    def n_configs(self) -> int:
        return int(np.prod([int(r) for r in self.radix], dtype=object))

    def remember(self, keys: np.ndarray) -> None:
        """Ask the estimator about `keys` (distinct, none known) and store."""
        configs = np.stack(np.unravel_index(keys, self.radix), axis=1)
        probs = _normalised(self.fitted.predict_proba(configs), self.n_classes)
        merged = np.concatenate([self.keys, keys])
        order = np.argsort(merged, kind="stable")
        self.keys = merged[order]
        self.probs = np.concatenate([self.probs, probs])[order]
        self.queried += len(keys)

    def lookup(self, keys: np.ndarray) -> np.ndarray:
        """Probabilities for each of `keys` (distinct), querying as needed."""
        if self.constant is not None:
            out = np.zeros((len(keys), self.n_classes))
            out[:, self.constant] = 1.0
            return out
        where = np.searchsorted(self.keys, keys)
        known = np.zeros(len(keys), dtype=bool)
        inside = where < len(self.keys)
        known[inside] = self.keys[where[inside]] == keys[inside]
        if not known.all():
            if self.fitted is None:
                raise RuntimeError(
                    "parent configuration outside a fully tabulated column"
                )
            self.remember(keys[~known])
            where = np.searchsorted(self.keys, keys)
        return self.probs[where]


class InContextCausalSampler:
    """Fit once on the real table, then sample under any set of cut edges.

    `cards[j]` is column j's number of codes; `dag_seed` is DECAF's edge list
    of `[parent, child]` column indices. `table_cap` is the largest parent
    configuration space tabulated in full at fit time.
    """

    def __init__(
        self,
        cards: Sequence[int],
        dag_seed: Sequence[Sequence[int]],
        estimator: ConditionalEstimator,
        table_cap: int = 100_000,
    ):
        self.cards = [int(c) for c in cards]
        self.estimator = estimator
        self.table_cap = int(table_cap)
        self.gen_order = topological_order(dag_seed, len(self.cards))
        self.parents: Dict[int, List[int]] = {j: [] for j in range(len(self.cards))}
        for parent, child in dag_seed:
            self.parents[int(child)].append(int(parent))
        for child in self.parents:
            self.parents[child] = sorted(set(self.parents[child]))
        self.root_marginals: Dict[int, np.ndarray] = {}
        self.columns: Dict[int, _Column] = {}
        self.fit_seconds = 0.0

    def fit(self, codes: np.ndarray) -> "InContextCausalSampler":
        started = time.time()
        codes = np.asarray(codes, dtype=np.int64)
        for j in self.gen_order:
            parents, k = self.parents[j], self.cards[j]
            child = codes[:, j]
            if child.min() < 0 or child.max() >= k:
                raise ValueError(f"column {j}: codes outside 0..{k - 1}")
            if not parents:
                self.root_marginals[j] = np.bincount(child, minlength=k) / len(child)
                continue
            column = _Column(parents, [self.cards[p] for p in parents], k)
            self.columns[j] = column
            if len(np.unique(child)) == 1:
                # Nothing to estimate, and a classifier cannot be fitted to it.
                column.constant = int(child[0])
                continue
            column.fitted = self.estimator.fit(codes[:, parents], child, k)
            if column.n_configs <= self.table_cap:
                column.remember(np.arange(column.n_configs, dtype=np.int64))
                column.fitted.release()
                column.fitted = None
        self.fit_seconds = time.time() - started
        return self

    def generate(
        self,
        n: int,
        biased_edges: Optional[Dict[int, List[int]]] = None,
        rng: Optional[np.random.Generator] = None,
        cut_strength: float = 1.0,
    ) -> np.ndarray:
        """Sample `n` rows in topological order, row-permuting each parent
        listed in `biased_edges[child]` before the child's conditional is read.

        `cut_strength` below 1 applies the cut to that share of the rows only
        (chosen independently per child, the same rows for all of the child's
        cut parents), so the child is drawn from the mixture
        `(1 - s) p(child | parents) + s p_cut(child | parents)`. 1 is DECAF's
        cut and draws exactly the random numbers it always did; 0 is no cut.
        """
        if not 0.0 <= cut_strength <= 1.0:
            raise ValueError(f"cut_strength {cut_strength} outside [0, 1]")
        rng = rng if rng is not None else np.random.default_rng()
        biased_edges = biased_edges or {}
        out = np.zeros((n, len(self.cards)), dtype=np.int64)
        for j in self.gen_order:
            if j in self.root_marginals:
                out[:, j] = rng.choice(self.cards[j], size=n, p=self.root_marginals[j])
                continue
            column = self.columns[j]
            conditioning = out[:, column.parents].copy()
            cut = [p for p in biased_edges.get(j, []) if p in column.parents]
            partial = cut_strength < 1.0 and bool(cut)
            rows = rng.random(n) < cut_strength if partial else None
            for parent in cut:
                at = column.parents.index(parent)
                shuffled = conditioning[rng.permutation(n), at]
                if partial:
                    conditioning[rows, at] = shuffled[rows]
                else:
                    conditioning[:, at] = shuffled
            keys = np.ravel_multi_index(conditioning.T, column.radix)
            distinct, inverse = np.unique(keys, return_inverse=True)
            cumulative = np.cumsum(column.lookup(distinct), axis=1)[inverse]
            draws = rng.random(n)[:, None] * cumulative[:, -1:]
            out[:, j] = np.minimum((draws >= cumulative).sum(axis=1), self.cards[j] - 1)
        return out

    def stats(self) -> Dict[str, float]:
        """What the fit and the generations so far asked of the estimator."""
        columns = self.columns.values()
        return {
            "incontext_fit_seconds": round(self.fit_seconds, 1),
            "incontext_root_columns": len(self.root_marginals),
            "incontext_tabulated_columns": sum(
                c.fitted is None and c.constant is None for c in columns
            ),
            "incontext_lazy_columns": sum(c.fitted is not None for c in columns),
            "incontext_constant_columns": sum(c.constant is not None for c in columns),
            "incontext_configs_queried": int(sum(c.queried for c in columns)),
        }

    def release(self) -> None:
        for column in self.columns.values():
            if column.fitted is not None:
                column.fitted.release()
                column.fitted = None
