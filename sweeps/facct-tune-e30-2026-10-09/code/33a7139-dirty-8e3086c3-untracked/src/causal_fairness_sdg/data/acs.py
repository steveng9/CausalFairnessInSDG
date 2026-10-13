"""ACS PUMS prediction tasks from Ding et al. 2021 ("Retiring Adult"), built
on the `folktables` package so the row filter and the target are the
published ones rather than a re-implementation.

`ACSIncome` is the designated successor to UCI Adult: same question (does this
person earn more than $50k), but a documented sampling frame, one state and
one year at a time. Registered per state (`acsincome_ca`, `acsincome_tx`) so
the same causal graph can be evaluated on two populations.

Where this departs from folktables' feature list, and why:

  - `POBP` (219 birthplaces) -> `WAOB`, the Census Bureau's own 8-way "world
    area of birth" recode of it, read from the same PUMS record.
  - `OCCP` (529 occupations) -> the 25 major occupation groups the PUMS data
    dictionary prefixes every occupation label with (MGR, BUS, FIN, ...).
  - `SCHL` (24 levels) and `RELP` (18) are collapsed to 10 and 7 levels;
    `AGEP` and `WKHP` are cut at fixed, public edges. Fixed edges rather than
    the quantile bins Adult uses, so no part of the encoding is fitted to the
    private data.
  - `HISP` is *added*, as a Hispanic-origin indicator. folktables leaves it
    out, but a third of the California sample is Hispanic and the ACS records
    that separately from race -- without it the largest minority group in
    both states is invisible to every fairness metric.
  - The three American Indian / Alaska Native codes of `RAC1P` are merged
    (one of them has 13 rows in California).

`ACSPublicCoverage` (`acspubcov_ca`, `acspubcov_tx`) asks whether a low-income
adult under 65 has public health insurance: a different outcome on a different
population, with eligibility rules (income, disability, age, pregnancy,
citizenship) that make "admissible" a statutory question rather than a
judgement call. California expanded Medicaid and Texas did not, so the two
states differ in policy and not only in population. Departures from
folktables: `ST` is dropped (constant within a state), `HISP` is added and
`RAC1P`/`SCHL` are collapsed as above, `AGEP`/`PINCP` are cut at fixed edges,
and "not applicable" is kept as an explicit level wherever the survey skips a
question (`ESP`, `MIL`, `ESR`, `FER`) instead of folktables' -1.

Building a table parses a ~270 MB PUMS file, so the encoded result is cached
under `data/processed/` next to a JSON codebook (code -> label for every
column, plus provenance). `run_single` calls the loader once per run; only the
first call per state pays for the build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..fairness.base import AttributeRoles
from ._encoding import (
    BIG_TABLE_ROWS,
    BIG_TABLE_SEED,
    PROCESSED_DIR,
    RAW_DIR,
    LoadedDataset,
    _cut,
    _levels,
    _map_codes,
    _sha256,
    _subsample,
    _to_codes,
    _write_atomically,
    validate_encoding,
)

ACS_RAW_DIR = RAW_DIR / "acs"
ACS_YEAR = "2018"
INCOME_OUTCOME = "high_income"
PUBCOV_OUTCOME = "public_coverage"
NOT_APPLICABLE = "N/A"

AGE_BINS = ([16, 24, 34, 44, 54, 64, np.inf],
            ["17-24", "25-34", "35-44", "45-54", "55-64", "65+"])
HOURS_BINS = ([0, 19, 34, 39, 40, 49, 59, np.inf],
              ["1-19", "20-34", "35-39", "40", "41-49", "50-59", "60+"])
# ACSPublicCoverage keeps people aged 15-64 with income up to $30k.
PUBCOV_AGE_BINS = ([14, 17, 24, 34, 44, 54, 64],
                   ["15-17", "18-24", "25-34", "35-44", "45-54", "55-64"])
PUBCOV_INCOME_BINS = (
    [-np.inf, 0, 5000, 10000, 15000, 20000, 25000, 30000],
    ["<=0", "1-5k", "5-10k", "10-15k", "15-20k", "20-25k", "25-30k"],
)

SEX_LEVELS = {1: "Male", 2: "Female"}
MAR_LEVELS = {
    1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never married",
}
COW_LEVELS = {
    1: "Private for-profit", 2: "Private not-for-profit", 3: "Local government",
    4: "State government", 5: "Federal government",
    6: "Self-employed, not incorporated", 7: "Self-employed, incorporated",
    8: "Unpaid family worker",
}
WAOB_LEVELS = {
    1: "US state", 2: "PR and US Island Areas", 3: "Latin America", 4: "Asia",
    5: "Europe", 6: "Africa", 7: "Northern America", 8: "Oceania and at Sea",
}
RAC1P_LEVELS = {
    1: "White", 2: "Black", 3: "AIAN", 4: "AIAN", 5: "AIAN", 6: "Asian",
    7: "NHPI", 8: "Some other race", 9: "Two or more races",
}
# Ordered by attainment, so the integer codes are ordinal.
SCHL_LEVELS = {
    **{code: "Grade 8 or less" for code in range(1, 12)},
    **{code: "Grade 9-12, no diploma" for code in range(12, 16)},
    16: "High school diploma", 17: "GED",
    18: "Some college", 19: "Some college",
    20: "Associate's", 21: "Bachelor's", 22: "Master's",
    23: "Professional degree", 24: "Doctorate",
}
RELP_LEVELS = {
    0: "Reference person", 1: "Spouse", 13: "Unmarried partner",
    2: "Child", 3: "Child", 4: "Child", 14: "Child",
    **{code: "Other relative" for code in range(5, 11)},
    11: "Nonrelative", 12: "Nonrelative", 15: "Nonrelative",
    16: "Group quarters", 17: "Group quarters",
}
YES_NO = {1: "Yes", 2: "No"}
NATIVITY_LEVELS = {1: "Native", 2: "Foreign born"}
CIT_LEVELS = {
    1: "Born in the US", 2: "Born in PR or US Island Areas",
    3: "Born abroad of American parent(s)", 4: "Naturalized", 5: "Not a citizen",
}
MIG_LEVELS = {1: "Same house", 2: "Moved from abroad", 3: "Moved within the US"}
ANC_LEVELS = {1: "Single", 2: "Multiple", 3: "Unclassified", 4: "Not reported"}
# The four codes below are asked of a sub-population only; the rest are N/A.
MIL_LEVELS = {  # N/A: under 17
    1: "On active duty", 2: "Veteran", 3: "Reserves/Guard training only",
    4: "Never served",
}
ESR_LEVELS = {  # N/A: under 16
    1: "Employed", 2: "Employed", 3: "Unemployed",
    4: "Armed forces", 5: "Armed forces", 6: "Not in labor force",
}
FER_LEVELS = {  # N/A: not a woman aged 15-50
    1: "Gave birth in the last year", 2: "No birth in the last year",
}
ESP_LEVELS = {  # N/A: not an own child living with a parent
    1: "All parents in labor force", 5: "All parents in labor force",
    7: "All parents in labor force",
    2: "One of two parents in labor force", 3: "One of two parents in labor force",
    4: "No parent in labor force", 6: "No parent in labor force",
    8: "No parent in labor force",
}

#: 2018 Census occupation codes by major group, inclusive. Checked against the
#: data dictionary's own label prefixes on every build (`_check_occupations`).
OCCP_GROUPS: List[Tuple[str, int, int]] = [
    ("MGR", 10, 440), ("BUS", 500, 750), ("FIN", 800, 960), ("CMM", 1005, 1240),
    ("ENG", 1305, 1560), ("SCI", 1600, 1980), ("CMS", 2001, 2060),
    ("LGL", 2100, 2180), ("EDU", 2205, 2555), ("ENT", 2600, 2920),
    ("MED", 3000, 3550), ("HLS", 3601, 3655), ("PRT", 3700, 3960),
    ("EAT", 4000, 4160), ("CLN", 4200, 4255), ("PRS", 4330, 4655),
    ("SAL", 4700, 4965), ("OFF", 5000, 5940), ("FFF", 6005, 6130),
    ("CON", 6200, 6765), ("EXT", 6800, 6950), ("RPR", 7000, 7640),
    ("PRD", 7700, 8990), ("TRN", 9005, 9760), ("MIL", 9800, 9830),
]


def _map_codes_or_na(codes: pd.Series, mapping: Dict[int, str]) -> pd.Series:
    """`_map_codes` for a question the survey skips for some people: a missing
    answer becomes the explicit first level `NOT_APPLICABLE`."""
    answered = codes.notna()
    mapped = _map_codes(codes[answered], mapping)
    levels = [NOT_APPLICABLE, *_levels(mapping)]
    out = pd.Series(NOT_APPLICABLE, index=codes.index, dtype=object)
    out[answered] = mapped.astype(object)
    return pd.Series(
        pd.Categorical(out, categories=levels, ordered=False), index=codes.index
    )


def _hispanic(codes: pd.Series) -> pd.Categorical:
    return pd.Categorical(
        np.where(codes.astype("int64") == 1, "Not Hispanic", "Hispanic"),
        categories=["Not Hispanic", "Hispanic"],
    )


def _occupation_group(codes: pd.Series) -> pd.Series:
    as_int = codes.astype("int64")
    edges = pd.IntervalIndex.from_tuples(
        [(lo, hi) for _, lo, hi in OCCP_GROUPS], closed="both"
    )
    position = edges.get_indexer(as_int)
    if (position < 0).any():
        raise ValueError(
            "OCCP: codes outside every major group: "
            f"{sorted(as_int[position < 0].unique())[:10]}"
        )
    names = [name for name, _, _ in OCCP_GROUPS]
    return pd.Series(
        pd.Categorical.from_codes(position, categories=names, ordered=False),
        index=codes.index,
    )


def _check_occupations(definitions: pd.DataFrame) -> None:
    """Every occupation the data dictionary labels `XXX-...` must fall in the
    `OCCP_GROUPS` range for `XXX`. (Two labels in the 2018 dictionary are
    missing their prefix, which is why the ranges and not the labels are the
    source of truth.)"""
    rows = definitions[(definitions[0] == "VAL") & (definitions[1] == "OCCP")]
    labelled = rows[rows[6].astype(str).str.match(r"^[A-Z]{3}-")]
    codes = pd.Series(labelled[4].astype(int).to_numpy(), name="OCCP")
    declared = labelled[6].astype(str).str[:3].to_numpy()
    assigned = _occupation_group(codes).astype(str).to_numpy()
    wrong = codes[declared != assigned]
    if len(wrong):
        raise ValueError(f"OCCP_GROUPS disagrees with the data dictionary at {list(wrong)}")


def _data_source():
    from folktables import ACSDataSource

    return ACSDataSource(
        survey_year=ACS_YEAR, horizon="1-Year", survey="person",
        root_dir=str(ACS_RAW_DIR),
    )


def _encode_income(raw: pd.DataFrame) -> pd.DataFrame:
    """The ACSIncome population and target, on labelled categorical columns."""
    import folktables

    people = folktables.adult_filter(raw)
    target = folktables.ACSIncome.target_transform(people[folktables.ACSIncome.target])
    return pd.DataFrame({
        "AGEP": _cut(people["AGEP"], AGE_BINS),
        "SEX": _map_codes(people["SEX"], SEX_LEVELS),
        "RAC1P": _map_codes(people["RAC1P"], RAC1P_LEVELS),
        "HISP": _hispanic(people["HISP"]),
        "WAOB": _map_codes(people["WAOB"], WAOB_LEVELS),
        "SCHL": _map_codes(people["SCHL"], SCHL_LEVELS),
        "MAR": _map_codes(people["MAR"], MAR_LEVELS),
        "RELP": _map_codes(people["RELP"], RELP_LEVELS),
        "OCCP": _occupation_group(people["OCCP"]),
        "COW": _map_codes(people["COW"], COW_LEVELS),
        "WKHP": _cut(people["WKHP"], HOURS_BINS),
        INCOME_OUTCOME: pd.Categorical(
            np.where(target, ">50K", "<=50K"), categories=["<=50K", ">50K"]
        ),
    })


def _encode_public_coverage(raw: pd.DataFrame) -> pd.DataFrame:
    """The ACSPublicCoverage population and target, on labelled columns."""
    import folktables

    task = folktables.ACSPublicCoverage
    people = folktables.acs.public_coverage_filter(raw)
    target = task.target_transform(people[task.target])
    return pd.DataFrame({
        "AGEP": _cut(people["AGEP"], PUBCOV_AGE_BINS),
        "SEX": _map_codes(people["SEX"], SEX_LEVELS),
        "RAC1P": _map_codes(people["RAC1P"], RAC1P_LEVELS),
        "HISP": _hispanic(people["HISP"]),
        "NATIVITY": _map_codes(people["NATIVITY"], NATIVITY_LEVELS),
        "CIT": _map_codes(people["CIT"], CIT_LEVELS),
        "ANC": _map_codes(people["ANC"], ANC_LEVELS),
        "SCHL": _map_codes(people["SCHL"], SCHL_LEVELS),
        "MAR": _map_codes(people["MAR"], MAR_LEVELS),
        "ESP": _map_codes_or_na(people["ESP"], ESP_LEVELS),
        "MIG": _map_codes(people["MIG"], MIG_LEVELS),
        "MIL": _map_codes_or_na(people["MIL"], MIL_LEVELS),
        "FER": _map_codes_or_na(people["FER"], FER_LEVELS),
        "DEAR": _map_codes(people["DEAR"], YES_NO),
        "DEYE": _map_codes(people["DEYE"], YES_NO),
        "DREM": _map_codes(people["DREM"], YES_NO),
        "DIS": _map_codes(people["DIS"], YES_NO),
        "ESR": _map_codes_or_na(people["ESR"], ESR_LEVELS),
        "PINCP": _cut(people["PINCP"], PUBCOV_INCOME_BINS),
        PUBCOV_OUTCOME: pd.Categorical(
            np.where(target, "Covered", "Not covered"),
            categories=["Not covered", "Covered"],
        ),
    })


@dataclass(frozen=True)
class AcsTask:
    """One folktables prediction task as this project encodes it."""

    #: Dataset-name and cache-file prefix: `<name>_<state>`.
    name: str
    folktables_name: str
    encode: Callable[[pd.DataFrame], pd.DataFrame]
    outcome: str
    roles: AttributeRoles
    #: Rows surviving the folktables filter in the 2018 1-Year sample, per
    #: state. A truncated or wrong-year download fails on this rather than
    #: producing a plausible-looking smaller table.
    expected_rows: Dict[str, int]
    #: Bump when the task's encoding changes, so a stale cached table is
    #: rebuilt instead of silently reused.
    version: int = 1


# Roles follow the Adult/SNAKE `prefair`-style split: ascribed characteristics
# protected, the labour-market chain admissible.
ACS_INCOME = AcsTask(
    name="acsincome",
    folktables_name="ACSIncome",
    encode=_encode_income,
    outcome=INCOME_OUTCOME,
    roles=AttributeRoles.create(
        protected={"SEX", "RAC1P", "HISP", "WAOB"},
        admissible={"SCHL", "OCCP", "COW", "WKHP"},
        outcome={INCOME_OUTCOME},
    ),
    expected_rows={"CA": 195_665, "TX": 135_924},
)

# Admissible = the categories public programmes are written in terms of:
# income, employment, disability, age, and a recent birth.
ACS_PUBLIC_COVERAGE = AcsTask(
    name="acspubcov",
    folktables_name="ACSPublicCoverage",
    encode=_encode_public_coverage,
    outcome=PUBCOV_OUTCOME,
    roles=AttributeRoles.create(
        protected={"SEX", "RAC1P", "HISP", "NATIVITY"},
        admissible={"PINCP", "ESR", "DIS", "AGEP", "FER"},
        outcome={PUBCOV_OUTCOME},
    ),
    expected_rows={"CA": 138_554, "TX": 98_928},
)


def _cache_paths(task: AcsTask, state: str) -> Tuple[Path, Path]:
    stem = (
        f"{task.name}_{state.lower()}_{ACS_YEAR}_n{BIG_TABLE_ROWS}"
        f"_s{BIG_TABLE_SEED}_v{task.version}"
    )
    return PROCESSED_DIR / f"{stem}.parquet", PROCESSED_DIR / f"{stem}.json"


def build_acs(task: AcsTask, state: str) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Download (if needed), filter, encode and subsample one state's table
    for `task`, and write it to the processed cache."""
    import folktables

    source = _data_source()
    raw = source.get_data(states=[state], download=True)
    _check_occupations(source.get_definitions(download=True))

    labelled = task.encode(raw)
    expected = task.expected_rows.get(state)
    if expected is not None and len(labelled) != expected:
        raise ValueError(
            f"{task.folktables_name} {state} {ACS_YEAR}: {len(labelled)} rows after "
            f"the folktables filter, expected {expected}"
        )
    if labelled.isna().any().any():
        raise ValueError(
            f"missing values after encoding: {labelled.isna().sum().to_dict()}"
        )

    df, codebook = _to_codes(_subsample(labelled))
    table_path, codebook_path = _cache_paths(task, state)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = ACS_RAW_DIR / ACS_YEAR / "1-Year" / f"psam_p{_state_fips(state)}.csv"
    positive = labelled[task.outcome].cat.categories[1]
    provenance = {
        "task": task.folktables_name,
        "state": state,
        "survey": f"ACS PUMS {ACS_YEAR} 1-Year person file",
        "folktables_version": getattr(folktables, "__version__", "unknown"),
        "encoding_version": task.version,
        "rows_after_filter": int(len(labelled)),
        "rows_kept": int(len(df)),
        "subsample_seed": BIG_TABLE_SEED,
        "raw_file": raw_file.name,
        "raw_file_sha256": _sha256(raw_file),
        "positive_label": str(positive),
        "positive_rate_full": float((labelled[task.outcome] == positive).mean()),
        "positive_rate_kept": float(df[task.outcome].mean()),
        "levels": codebook,
    }
    _write_atomically(table_path, lambda p: df.to_parquet(p, index=False))
    _write_atomically(
        codebook_path, lambda p: p.write_text(json.dumps(provenance, indent=2))
    )
    return df, codebook


def _state_fips(state: str) -> str:
    from folktables.load_acs import _STATE_CODES

    return _STATE_CODES[state]


def load_acs(task: AcsTask, state: str) -> LoadedDataset:
    """One state's table for `task`: 50k rows, built on first use."""
    table_path, _ = _cache_paths(task, state)
    df = pd.read_parquet(table_path) if table_path.exists() else build_acs(task, state)[0]
    domain = {col: int(df[col].nunique()) for col in df.columns}
    validate_encoding(df, domain, task.roles)
    return df, domain, task.roles


#: Which states each task is registered for. Adding one is an entry here plus
#: the dataset name in the DAG / role / training-settings registries -- the
#: registry test (`tests/test_dataset_registry.py`) names whichever is missing.
ACS_REGISTRY: Dict[str, Tuple[AcsTask, str]] = {
    f"{task.name}_{state.lower()}": (task, state)
    for task, states in (
        (ACS_INCOME, ("CA", "TX")),
        (ACS_PUBLIC_COVERAGE, ("CA", "TX")),
    )
    for state in states
}

ACS_DATASETS: Dict[str, Callable[[], LoadedDataset]] = {
    name: partial(load_acs, task, state) for name, (task, state) in ACS_REGISTRY.items()
}


def acs_codebook(dataset: str) -> Dict[str, object]:
    """Provenance and code -> label lists for a registered ACS dataset name."""
    return json.loads(_cache_paths(*ACS_REGISTRY[dataset])[1].read_text())
