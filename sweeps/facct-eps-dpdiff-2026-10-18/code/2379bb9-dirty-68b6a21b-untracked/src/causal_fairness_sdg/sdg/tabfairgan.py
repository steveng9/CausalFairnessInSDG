"""TabFairGAN (Rajabi & Garibay, MAKE 2022) as a comparison generator.

The fair-synthetic-data baseline with no causal graph: a WGAN-GP whose
generator, after `epochs - fair_epochs` epochs of ordinary training, is
trained for `fair_epochs` more on the critic loss plus `lamda` times the
demographic-parity gap `P(y=1|s=under) - P(y=1|s=priv)` of its own batch. It
is the generator behind Eitan et al.'s "fair synthetic data does not
necessarily lead to fair models", and it tells us whether our findings come
from cutting edges in a graph or from any parity constraint imposed inside
the table.

The model is the published package, unmodified (`pip install tabfairgan`,
0.1.0, MIT licence; https://github.com/amirarsalan90/TabFairGAN). This
module only maps our tables and roles onto its interface:

  Mechanisms   `none` trains without the fairness phase (`fair_epochs` = 0).
               `dp` is its demographic-parity loss. It has nothing that
               corresponds to FTU or CF, so the grid gives it only these two
               (`mechanisms` below).
  S            TabFairGAN takes one binary protected attribute. For a role
               split protecting several, it is the first one of
               `PROTECTED_PRIORITY[dataset]` that the split protects: the
               attribute the dataset's single-attribute split is about, and
               race on COMPAS as in the paper. It is recorded on every row
               (`tabfairgan_s`).
               A many-valued S is binarised majority-versus-rest on the
               training rows, as every gap metric here binarises it. The
               generator then sees only the two groups, so each "rest" row it
               generates is given a value drawn from the training rows'
               non-majority distribution of S. For a binary S this mapping
               is the identity.
  Y            the outcome, with Y_desire = 1 and S_under the group with the
               lower real rate of y = 1. Its loss is signed (it raises the
               under-group's rate), so this pairing is what closes the gap,
               whichever label is the favourable one.
  Columns      every column of our tables is an integer-coded categorical
               and is given to it as one (string dtype, one-hot). The package
               quantile-scales at least one numeric column and fails without
               one, so a constant column is added for training and dropped
               from the release; it carries no information.

Settings, from the paper's Table III (batch 256, n_crit 4, Adam 2e-4 and the
WGAN-GP weight 10 are fixed in the package): COMPAS 40 + 30 epochs, lambda
2.2; Adult 170 + 30, lambda 0.5. Every other dataset takes Adult's, the
paper's setting for the larger census-style table. Their COMPAS is a
different extract (risk decile, 16k rows) from ours.

**Not private.** `epsilon`/`delta` are accepted for interface parity and
logged as NULL.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..fairness.base import AttributeRoles, FairnessMechanism
from .base import SDGMethod, SDGResult

#: Which protected attribute TabFairGAN's single S is, per dataset: the first
#: of these the role split protects.
PROTECTED_PRIORITY: Dict[str, List[str]] = {
    "adult": ["sex", "race", "native-country"],
    "compas": ["race", "sex", "age_cat"],
    "snake": ["female", "wbhaom", "citistat"],
    "sbo": ["SEX1", "RACE1", "ETH1", "VET1", "BORNUS1"],
    "acsincome_ca": ["SEX", "RAC1P", "HISP", "WAOB"],
    "acsincome_tx": ["SEX", "RAC1P", "HISP", "WAOB"],
    "acspubcov_ca": ["SEX", "RAC1P", "HISP", "DIS", "NATIVITY"],
    "acspubcov_tx": ["SEX", "RAC1P", "HISP", "DIS", "NATIVITY"],
    "cdc_diabetes": ["Sex", "Income", "Education"],
    "hmda_ca": ["sex", "race", "ethnicity", "age"],
    "hmda_tx": ["sex", "race", "ethnicity", "age"],
}

PAPER_ADULT = {"epochs": 200, "fair_epochs": 30, "lamda": 0.5, "batch_size": 256}
SETTINGS: Dict[str, Dict[str, Any]] = {
    "compas": {"epochs": 70, "fair_epochs": 30, "lamda": 2.2, "batch_size": 256},
}

CONST_COL = "_tabfairgan_const"
MAJORITY, REST = "majority", "rest"


class TabFairGANMethod(SDGMethod):
    name = "tabfairgan"
    is_private = False
    mechanisms = ("none", "dp")

    def __init__(self) -> None:
        self.overrides: Dict[str, Any] = {}
        self._cache: Dict[Tuple, Any] = {}

    @staticmethod
    def _require_backend() -> None:
        import tabfairgan  # noqa: F401 - raises ImportError when absent

    def _settings_for(self, dataset_name: Optional[str]) -> Dict[str, Any]:
        cfg = dict(SETTINGS.get(dataset_name or "", PAPER_ADULT))
        cfg.update(self.overrides)
        return cfg

    @staticmethod
    def protected_attribute(dataset_name: str, roles: AttributeRoles) -> str:
        for attr in PROTECTED_PRIORITY.get(dataset_name, []):
            if attr in roles.protected:
                return attr
        return sorted(roles.protected)[0]

    def fit_generate(
        self,
        data: pd.DataFrame,
        domain: Dict[str, int],
        roles: AttributeRoles,
        fairness_mechanism: FairnessMechanism,
        epsilon: float,
        delta: float,
        n_synth: int,
        seed: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ) -> SDGResult:
        import torch
        from tabfairgan import TFG

        mech = fairness_mechanism.name
        if mech not in self.mechanisms:
            raise ValueError(f"tabfairgan has no {mech!r} mechanism; only {self.mechanisms}")
        seed = 0 if seed is None else int(seed)
        cfg = self._settings_for(dataset_name)
        device = cfg.get("device", "cpu")
        outcome = next(iter(roles.outcome))
        s_attr = self.protected_attribute(dataset_name, roles)

        table = data.astype(str)
        majority = data[s_attr].mode().iloc[0]
        is_major = data[s_attr] == majority
        rest_values = data.loc[~is_major, s_attr].to_numpy()
        table[s_attr] = np.where(is_major, MAJORITY, REST)
        table[CONST_COL] = 0.0
        y1 = data[outcome].astype(str) == "1"
        under = MAJORITY if y1[is_major].mean() < y1[~is_major].mean() else REST

        fairness = None
        if mech == "dp":
            fairness = {"fair_epochs": int(cfg["fair_epochs"]), "lamda": float(cfg["lamda"]),
                        "S": s_attr, "Y": outcome, "S_under": under, "Y_desire": "1"}

        digest = hashlib.sha1(pd.util.hash_pandas_object(data, index=False).values).hexdigest()
        key = (digest, seed, mech, s_attr if mech == "dp" else None,
               tuple(sorted((k, str(v)) for k, v in cfg.items())))
        if key not in self._cache:
            if self._cache and next(iter(self._cache))[0] != digest:
                self._cache.clear()  # keep only the current data's models
            random.seed(seed)
            np.random.seed(seed)  # the package's own 90/10 split draws from it
            torch.manual_seed(seed)
            model = TFG(table, epochs=int(cfg["epochs"]), batch_size=int(cfg["batch_size"]),
                        device=device, fairness_config=fairness)
            model.train()
            self._cache[key] = model
        model = self._cache[key]

        torch.manual_seed(seed)
        fake = model.generate_fake_df(num_rows=n_synth).drop(columns=[CONST_COL])
        rng = np.random.default_rng(seed)
        s_gen = fake[s_attr].to_numpy()
        is_rest = s_gen == REST
        synth = fake.copy()
        synth[s_attr] = majority
        synth.loc[is_rest, s_attr] = rng.choice(rest_values, size=int(is_rest.sum()))
        for col in data.columns:
            if col != s_attr:
                synth[col] = synth[col].astype(float).round().astype(data[col].dtype)
        synth[s_attr] = synth[s_attr].astype(data[s_attr].dtype)
        synth = synth[list(data.columns)]

        extra = {"tabfairgan_s": s_attr, "tabfairgan_s_under": under,
                 **{f"tabfairgan_{k}": v for k, v in cfg.items() if isinstance(v, str)}}
        return SDGResult(synthetic_data=synth, graph_edges=[], extra=extra)
