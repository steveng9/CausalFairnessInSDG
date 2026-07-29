from typing import Dict, Type

from .base import FairnessMechanism
from .cf import CFFairness
from .dp import DPFairness
from .ftu import FTUFairness
from .none import NoFairness

# Adding a 5th graph-based fairness mechanism: implement a FairnessMechanism
# subclass (see ftu.py/cf.py/dp.py for the pattern) and add one entry here.
FAIRNESS_MECHANISMS: Dict[str, Type[FairnessMechanism]] = {
    cls.name: cls for cls in (NoFairness, FTUFairness, DPFairness, CFFairness)
}


def get_fairness_mechanism(name: str) -> FairnessMechanism:
    try:
        return FAIRNESS_MECHANISMS[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown fairness mechanism {name!r}; available: {sorted(FAIRNESS_MECHANISMS)}"
        ) from exc
