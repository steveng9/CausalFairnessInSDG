from typing import List

from .base import AttributeRoles, Edge, FairnessMechanism


def _is_direct_protected_outcome_edge(edge: Edge, roles: AttributeRoles) -> bool:
    a, b = edge
    return (a in roles.protected and b in roles.outcome) or (
        a in roles.outcome and b in roles.protected
    )


class FTUFairness(FairnessMechanism):
    """Fairness Through Unawareness (DECAF Def. 1 / Corollary 2, narrow form).

    Forbids only the *direct* edge between a protected attribute and an
    outcome attribute. Indirect paths through other attributes (including
    non-admissible ones) are left untouched -- this is the narrowest, most
    literal translation of DECAF's FTU edge-removal rule onto a tree
    structure, and the cheapest to enforce: a one-time static filter of the
    candidate list, no need to inspect the partially-built structure.
    """

    name = "ftu"

    def filter_candidates(
        self, candidates: List[Edge], roles: AttributeRoles
    ) -> List[Edge]:
        return [
            e for e in candidates if not _is_direct_protected_outcome_edge(e, roles)
        ]
