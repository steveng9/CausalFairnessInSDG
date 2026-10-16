from typing import Dict, FrozenSet, List

import networkx as nx

from .base import AttributeRoles, Edge, FairnessMechanism


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
        """DECAF Corollary 1: every non-admissible member of the outcome's
        Markov boundary must be d-separated from every protected attribute
        given the admissible set R, in the real-data graph `dag`. Members that
        are not get the edge joining them to the boundary cut.

        A directed-path rule ("a path through an admissible node is blocked")
        is not enough on a DAG: conditioning on an admissible *collider*
        (A -> R <- B -> Y) opens a path, so B carries information about A
        among people with the same R. Paths are judged by d-separation instead.
        """
        protected = [p for p in roles.protected if p in dag]
        admissible = set(roles.admissible) & set(dag.nodes)
        biased: Dict[str, set] = {}

        def tied_to_protected(node: str) -> bool:
            if node in roles.protected:
                return True
            given = admissible - {node}
            return any(not _d_separated(dag, {node}, {p}, given) for p in protected)

        for o in roles.outcome:
            if o not in dag or not protected:
                continue
            for parent in dag.predecessors(o):
                if parent not in admissible and tied_to_protected(parent):
                    biased.setdefault(o, set()).add(parent)
            for child in dag.successors(o):
                if child not in admissible and tied_to_protected(child):
                    biased.setdefault(child, set()).add(o)
                for spouse in dag.predecessors(child):
                    if spouse != o and spouse not in admissible and tied_to_protected(spouse):
                        biased.setdefault(child, set()).add(spouse)
        return {child: sorted(parents) for child, parents in biased.items()}


def _d_separated(dag: nx.DiGraph, x: set, y: set, z: set) -> bool:
    # networkx renamed `d_separated` to `is_d_separator` in 3.3.
    test = getattr(nx, "is_d_separator", None) or nx.d_separated
    return test(dag, x, y, z)
