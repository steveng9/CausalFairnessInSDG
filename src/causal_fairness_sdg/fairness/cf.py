from typing import Dict, FrozenSet, List

import networkx as nx

from .base import AttributeRoles, Edge, FairnessMechanism, biased_edges_from_paths


def path_blocked(graph: nx.Graph, a: str, b: str, admissible: FrozenSet[str]) -> bool:
    """True iff the unique path between `a` and `b` in the forest `graph`
    passes through at least one admissible attribute. Assumes `graph` is
    acyclic (a forest) and `a`/`b` are connected within it, so the path is
    unique -- this is always true for MST's tree-under-construction."""
    path = nx.shortest_path(graph, a, b)
    interior = path[1:-1]
    return any(node in admissible for node in interior)


class CFFairness(FairnessMechanism):
    """Conditional Fairness (DECAF Def. 3 / Corollary 1), equivalent to
    PreFair's "justifiable fairness": every path between a protected
    attribute and an outcome attribute must pass through at least one
    admissible attribute. CF subsumes both FTU (admissible = everything but
    the protected attribute) and DP (admissible = empty set) in DECAF's
    original DAG framing; on a tree we keep it as a distinct, tunable-via-R
    mechanism.

    Implemented as an incremental predicate: reject a candidate edge if
    merging the two components it would connect creates any protected<->
    outcome pair whose path is not blocked by an admissible attribute.
    """

    name = "cf"

    def allow_edge(self, edge: Edge, graph: nx.Graph, roles: AttributeRoles) -> bool:
        a, b = edge
        component = set(nx.node_connected_component(graph, a)) | set(
            nx.node_connected_component(graph, b)
        )
        protected_in = component & roles.protected
        outcome_in = component & roles.outcome
        if not protected_in or not outcome_in:
            return True

        merged = graph.copy()
        merged.add_edge(a, b)
        return all(
            path_blocked(merged, p, o, roles.admissible)
            for p in protected_in
            for o in outcome_in
        )

    def select_biased_edges(
        self, dag: nx.DiGraph, roles: AttributeRoles
    ) -> Dict[str, List[str]]:
        # A path is blocked (left alone) iff some interior node is admissible.
        return biased_edges_from_paths(
            dag,
            roles,
            path_is_blocked=lambda interior: any(n in roles.admissible for n in interior),
        )
