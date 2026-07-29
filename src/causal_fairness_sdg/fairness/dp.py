import networkx as nx

from .base import AttributeRoles, Edge, FairnessMechanism


class DPFairness(FairnessMechanism):
    """Demographic Parity / unconditional independence (DECAF Def. 2 /
    Corollary 3, i.e. Conditional Fairness with R = empty set).

    On a general causal DAG, DP removes only the specific edges from the
    outcome's Markov boundary that aren't d-separated from the protected
    attribute given nothing. On a *connected* tree this doesn't translate
    directly: once directed by sampling order a spanning tree is a polytree
    with no colliders, so any two connected nodes are dependent absent
    conditioning -- meaning true unconditional independence (R = empty set)
    between a protected and an outcome attribute is only achievable by never
    letting them end up in the same connected component at all.

    So DP-for-trees is implemented as: reject any candidate edge that would
    merge a component containing a protected attribute with a component
    containing an outcome attribute. The practical effect is that MST no
    longer produces a single spanning tree but a spanning *forest*, with all
    protected-side and outcome-side attributes kept in disjoint components.
    This is a real consequence of the definition, not an approximation --
    and matches DECAF's own characterization of DP as the strictest of the
    three algorithmic definitions (see README.md for the full derivation).
    """

    name = "dp"

    def allow_edge(self, edge: Edge, graph: nx.Graph, roles: AttributeRoles) -> bool:
        a, b = edge
        component = set(nx.node_connected_component(graph, a)) | set(
            nx.node_connected_component(graph, b)
        )
        return not (component & roles.protected and component & roles.outcome)
