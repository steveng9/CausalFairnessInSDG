from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Tuple

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

    def select_biased_edges(
        self, dag: nx.DiGraph, roles: AttributeRoles
    ) -> Dict[str, List[str]]:
        """For SDG methods that start from a *whole, fixed* causal DAG (DECAF)
        rather than incrementally building one (MST/PrivBayes/AIM), decide
        which edges bias the outcome and should be shuffled ("surrogate value
        substitution") at generation time.

        Returns `{child: [parents whose contribution should be shuffled]}`.
        Default: no-op (nothing biased), matching `NoFairness`.
        """
        return {}


def edges_allowed(
    edges: List[Edge], graph: nx.Graph, roles: AttributeRoles, mechanism: "FairnessMechanism"
) -> bool:
    """True iff every pairwise edge in `edges` survives both of `mechanism`'s
    hooks against the graph built so far. Lets multi-parent (PrivBayes) and
    multi-attribute-clique (AIM) candidates reuse the existing single-edge
    `FairnessMechanism` interface unchanged, rather than each mechanism
    needing its own multi-edge-aware logic."""
    survivors = mechanism.filter_candidates(edges, roles)
    if len(survivors) != len(edges):
        return False
    return all(mechanism.allow_edge(e, graph, roles) for e in edges)


def biased_edges_from_paths(
    dag: nx.DiGraph,
    roles: AttributeRoles,
    path_is_blocked,
) -> Dict[str, List[str]]:
    """Shared `select_biased_edges` implementation for FTU/DP/CF: enumerate
    every simple directed path from a protected attribute to an outcome
    attribute, keep the ones `path_is_blocked(interior_nodes)` says are
    *not* blocked, and cut each such path by shuffling the edge nearest the
    outcome node (matches DECAF's own convention -- see e.g. its toy example
    `bias_dict = {6: [3]}`, which cuts the edge immediately upstream of the
    affected node rather than the one nearest the protected attribute).

    `path_is_blocked(interior)` receives the path's interior nodes (protected
    and outcome endpoints excluded) and returns whether that path should be
    left alone. FTU/DP/CF differ only in this predicate.
    """
    biased: Dict[str, set] = {}
    for p in roles.protected:
        if p not in dag:
            continue
        for o in roles.outcome:
            if o not in dag or p == o:
                continue
            for path in nx.all_simple_paths(dag, p, o):
                interior = path[1:-1]
                if path_is_blocked(interior):
                    continue
                biased.setdefault(o, set()).add(path[-2])
    return {child: sorted(parents) for child, parents in biased.items()}
