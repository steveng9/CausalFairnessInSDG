from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Tuple

import networkx as nx

Edge = Tuple[str, str]


@dataclass(frozen=True)
class AttributeRoles:
    """The three disjoint attribute sets every causal fairness definition is
    stated in terms of: protected (sensitive, unactionable), admissible
    (allowed to influence the outcome despite correlating with a protected
    attribute), and outcome (the prediction target(s))."""

    protected: FrozenSet[str]
    admissible: FrozenSet[str]
    outcome: FrozenSet[str]

    @classmethod
    def create(
        cls,
        protected: Iterable[str],
        admissible: Iterable[str],
        outcome: Iterable[str],
    ) -> "AttributeRoles":
        protected_s = frozenset(protected)
        admissible_s = frozenset(admissible)
        outcome_s = frozenset(outcome)
        overlap = (
            (protected_s & outcome_s)
            | (protected_s & admissible_s)
            | (admissible_s & outcome_s)
        )
        if overlap:
            raise ValueError(
                f"Attribute role sets must be disjoint; overlap: {sorted(overlap)}"
            )
        return cls(protected=protected_s, admissible=admissible_s, outcome=outcome_s)


class FairnessMechanism(ABC):
    """Common interface for causal-graph fairness mechanisms applied to the
    structure-selection step of a graphical/marginal-based SDG method (MST,
    PrivBayes, ...).

    Two independent hook points, chosen to match where PreFair's algorithms
    intervene in MST's Kruskal's-style construction:

      - `filter_candidates`: a one-time static filter of the full candidate
        edge list, applied once before any scoring happens. Cheap; sufficient
        for mechanisms whose condition doesn't depend on the partially-built
        structure (FTU).
      - `allow_edge`: an incremental predicate evaluated for each candidate
        edge during greedy tree construction, given the graph built so far.
        Needed by mechanisms whose condition depends on what's already been
        connected (CF's path-blocking, DP's component-separation).

    Both default to no-ops, so a mechanism only overrides what it needs. A
    5th definition is a new subclass plus one line in `registry.py`.
    """

    name: str = "base"

    def filter_candidates(
        self, candidates: List[Edge], roles: AttributeRoles
    ) -> List[Edge]:
        return candidates

    def allow_edge(self, edge: Edge, graph: nx.Graph, roles: AttributeRoles) -> bool:
        return True
