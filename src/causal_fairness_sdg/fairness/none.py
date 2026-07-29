from .base import FairnessMechanism


class NoFairness(FairnessMechanism):
    """Baseline: no fairness constraint. Identical behavior to vanilla MST/PrivBayes."""

    name = "none"
