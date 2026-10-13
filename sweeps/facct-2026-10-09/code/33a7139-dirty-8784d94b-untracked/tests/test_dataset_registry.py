"""A dataset is registered in several places -- loader, causal graph, role
splits, per-method training settings. These tests name whichever is missing,
so adding a dataset is "make this file pass" rather than a grep."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from causal_fairness_sdg.data import acs
from causal_fairness_sdg.data._encoding import validate_encoding
from causal_fairness_sdg.data.causal_graphs import CAUSAL_GRAPHS, as_digraph
from causal_fairness_sdg.data.datasets import DATASETS
from causal_fairness_sdg.experiments.runner import SDG_METHODS
from causal_fairness_sdg.fairness import AttributeRoles

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_experiments import DECAF_SETTINGS, ROLE_CONFIGS  # noqa: E402

#: Adult's `education-num` runs 1..16 (see `_encoding._recode`); kept as is
#: because every published Adult number was produced under it.
KNOWN_CONTRACT_BREAKS = {"adult"}


def _load(name):
    try:
        return DATASETS[name]()
    except FileNotFoundError as exc:  # SNAKE/SBO live outside the repo
        pytest.skip(str(exc))
    except Exception as exc:  # noqa: BLE001 - e.g. no network for a first download
        pytest.skip(f"{name} unavailable: {exc}")


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_registered_everywhere(name):
    assert name in CAUSAL_GRAPHS, "add a DAG in data/causal_graphs.py"
    assert name in ROLE_CONFIGS, "add role splits in scripts/run_experiments.py"
    assert name in DECAF_SETTINGS, "add DECAF settings in scripts/run_experiments.py"
    for method_name, method in SDG_METHODS.items():
        settings = getattr(method, "settings", None)
        if settings:
            assert name in settings, f"add {method_name} settings in sdg/decaf_variants.py"


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_loader_honours_the_encoding_contract(name):
    df, domain, roles = _load(name)
    if name not in KNOWN_CONTRACT_BREAKS:
        validate_encoding(df, domain, roles)
    as_digraph(CAUSAL_GRAPHS[name], list(df.columns))
    outcome = next(iter(roles.outcome))
    for _label, protected, admissible in ROLE_CONFIGS[name]:
        split = AttributeRoles.create(protected, admissible, [outcome])
        assert (split.protected | split.admissible) <= set(df.columns)


def test_acs_code_maps_refuse_unknown_codes():
    with pytest.raises(ValueError, match="no mapping"):
        acs._map_codes(pd.Series([1, 2, 3], name="SEX"), acs.SEX_LEVELS)


def test_acs_occupation_groups_cover_their_ranges():
    codes = pd.Series([10, 440, 500, 9830], name="OCCP")
    assert list(acs._occupation_group(codes)) == ["MGR", "MGR", "BUS", "MIL"]
    with pytest.raises(ValueError, match="outside every major group"):
        acs._occupation_group(pd.Series([450], name="OCCP"))


def test_acs_codes_stay_contiguous_when_a_level_is_absent():
    labelled = pd.DataFrame({
        "MAR": acs._map_codes(pd.Series([1, 3, 5, 5], name="MAR"), acs.MAR_LEVELS),
    })
    codes, codebook = acs._to_codes(labelled)
    assert sorted(codes["MAR"].unique()) == [0, 1, 2]
    assert codebook["MAR"] == ["Married", "Divorced", "Never married"]


def test_acs_bins_are_closed_on_the_documented_edges():
    ages = acs._cut(pd.Series([17, 24, 25, 64, 65, 94], name="AGEP"), acs.AGE_BINS)
    assert list(ages) == ["17-24", "17-24", "25-34", "55-64", "65+", "65+"]
    hours = acs._cut(pd.Series([1, 39, 40, 41, 99], name="WKHP"), acs.HOURS_BINS)
    assert list(hours) == ["1-19", "35-39", "40", "41-49", "60+"]
