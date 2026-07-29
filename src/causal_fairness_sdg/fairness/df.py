from enum import Enum


class EvalReference(Enum):
    """DECAF's Distributional Fairness (Def. 4) is *not* a graph-mutation rule
    like FTU/DP/CF -- it's a meta-property about which reference distribution
    a downstream optimal predictor's fairness gets evaluated against, once
    trained on the synthetic data P'(X):

      - ORIGINAL: evaluate the trained predictor's fairness against the real,
        possibly-biased distribution P(X). This is the practically useful
        case DECAF actually implements and evaluates in its experiments --
        publish fair-looking synthetic data, but require it stays fair when a
        model trained on it meets the real world.
      - SYNTHETIC: evaluate against the synthetic data's own distribution
        P'(X). DECAF calls this the "uninteresting" case (Sec 4.2): trivially
        satisfiable by e.g. randomizing the protected attribute, and gives no
        guarantee about real-world deployment.

    Because DF is orthogonal to FTU/DP/CF -- any of the three (or none) can be
    combined with either reference distribution -- it is intentionally *not*
    a `FairnessMechanism` subclass and does not live in `registry.py`'s
    graph-mechanism list. It is logged as its own axis (`eval_reference` in
    the experiments database) and consumed by `eval.fairness_metrics` when
    computing fairness gaps against held-out data.
    """

    ORIGINAL = "original"
    SYNTHETIC = "synthetic"
