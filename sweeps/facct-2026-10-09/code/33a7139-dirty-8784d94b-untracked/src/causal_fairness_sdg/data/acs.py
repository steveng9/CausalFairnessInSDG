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

Building a table parses a ~270 MB PUMS file, so the encoded result is cached
under `data/processed/` next to a JSON codebook (code -> label for every
column, plus provenance). `run_single` calls the loader once per run; only the
first call per state pays for the build.
"""

from __future__ import annotations

import hashlib
import json
import os
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
    _subsample,
    validate_encoding,
)

ACS_RAW_DIR = RAW_DIR / "acs"
ACS_YEAR = "2018"
#: Bump when any mapping below changes, so a stale cached table is rebuilt
#: instead of silently reused.
ENCODING_VERSION = 1

#: Rows surviving folktables' `adult_filter`, as reported for the 2018 1-Year
#: sample. A truncated or wrong-year download fails here rather than producing
#: a plausible-looking smaller table.
EXPECTED_INCOME_ROWS = {"CA": 195_665, "TX": 135_924}

INCOME_OUTCOME = "high_income"

AGE_BINS = ([16, 24, 34, 44, 54, 64, np.inf],
            ["17-24", "25-34", "35-44", "45-54", "55-64", "65+"])
HOURS_BINS = ([0, 19, 34, 39, 40, 49, 59, np.inf],
              ["1-19", "20-34", "35-39", "40", "41-49", "50-59", "60+"])

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


def _levels(mapping: Dict[int, str]) -> List[str]:
    """Distinct labels of `mapping` in first-appearance order."""
    return list(dict.fromkeys(mapping.values()))


def _map_codes(codes: pd.Series, mapping: Dict[int, str]) -> pd.Series:
    """Relabel PUMS `codes`, refusing any value the mapping does not cover --
    an unmapped code would otherwise become NaN and be dropped without trace."""
    as_int = codes.astype("int64")
    unmapped = sorted(set(as_int.unique()) - set(mapping))
    if unmapped:
        raise ValueError(f"{codes.name}: PUMS codes with no mapping: {unmapped}")
    return pd.Series(
        pd.Categorical(as_int.map(mapping), categories=_levels(mapping), ordered=True),
        index=codes.index,
    )


def _cut(values: pd.Series, bins: Tuple[Sequence[float], Sequence[str]]) -> pd.Series:
    edges, labels = bins
    out = pd.cut(values, bins=edges, labels=labels, right=True, ordered=True)
    if out.isna().any():
        raise ValueError(
            f"{values.name}: values outside ({edges[0]}, {edges[-1]}]: "
            f"{sorted(values[out.isna()].unique())[:10]}"
        )
    return out


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
        "HISP": pd.Categorical(
            np.where(people["HISP"].astype("int64") == 1, "Not Hispanic", "Hispanic"),
            categories=["Not Hispanic", "Hispanic"],
        ),
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


def _to_codes(labelled: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Integer-code `labelled` on the levels actually present, in declared
    order, and return the code -> label lists alongside. Levels are dropped
    *after* subsampling so the codes stay contiguous whatever the sample
    happens to contain."""
    codes, codebook = {}, {}
    for col in labelled.columns:
        observed = labelled[col].cat.remove_unused_categories()
        codes[col] = observed.cat.codes.astype("int64")
        codebook[col] = [str(level) for level in observed.cat.categories]
    return pd.DataFrame(codes), codebook


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_paths(task: str, state: str) -> Tuple[Path, Path]:
    stem = (
        f"{task}_{state.lower()}_{ACS_YEAR}_n{BIG_TABLE_ROWS}"
        f"_s{BIG_TABLE_SEED}_v{ENCODING_VERSION}"
    )
    return PROCESSED_DIR / f"{stem}.parquet", PROCESSED_DIR / f"{stem}.json"


def _write_atomically(path: Path, write: Callable[[Path], None]) -> None:
    # Concurrent shards may all find the cache missing; a rename makes the
    # last writer win with a complete file instead of interleaving.
    scratch = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write(scratch)
    os.replace(scratch, path)


def build_acs_income(state: str) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Download (if needed), filter, encode and subsample one state's
    ACSIncome table, and write it to the processed cache."""
    import folktables

    source = _data_source()
    raw = source.get_data(states=[state], download=True)
    _check_occupations(source.get_definitions(download=True))

    labelled = _encode_income(raw)
    expected = EXPECTED_INCOME_ROWS.get(state)
    if expected is not None and len(labelled) != expected:
        raise ValueError(
            f"ACSIncome {state} {ACS_YEAR}: {len(labelled)} rows after the "
            f"folktables filter, expected {expected}"
        )
    if labelled.isna().any().any():
        raise ValueError(
            f"missing values after encoding: {labelled.isna().sum().to_dict()}"
        )

    df, codebook = _to_codes(_subsample(labelled))
    table_path, codebook_path = _cache_paths("acsincome", state)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted((ACS_RAW_DIR / ACS_YEAR / "1-Year").glob("psam_p*.csv"))
    provenance = {
        "task": "ACSIncome",
        "state": state,
        "survey": f"ACS PUMS {ACS_YEAR} 1-Year person file",
        "folktables_version": getattr(folktables, "__version__", "unknown"),
        "encoding_version": ENCODING_VERSION,
        "rows_after_filter": int(len(labelled)),
        "rows_kept": int(len(df)),
        "subsample_seed": BIG_TABLE_SEED,
        "raw_files_sha256": {p.name: _sha256(p) for p in raw_files},
        "positive_rate_full": float((labelled[INCOME_OUTCOME] == ">50K").mean()),
        "positive_rate_kept": float(df[INCOME_OUTCOME].mean()),
        "levels": codebook,
    }
    _write_atomically(table_path, lambda p: df.to_parquet(p, index=False))
    _write_atomically(
        codebook_path, lambda p: p.write_text(json.dumps(provenance, indent=2))
    )
    return df, codebook


def load_acs_income(state: str) -> LoadedDataset:
    """ACSIncome for one state, 11 attributes + outcome, 50k rows.

    Roles follow the Adult/SNAKE `prefair`-style split: ascribed
    characteristics protected, the labour-market chain admissible.
    """
    table_path, _ = _cache_paths("acsincome", state)
    df = pd.read_parquet(table_path) if table_path.exists() else build_acs_income(state)[0]

    domain = {col: int(df[col].nunique()) for col in df.columns}
    roles = AttributeRoles.create(
        protected={"SEX", "RAC1P", "HISP", "WAOB"},
        admissible={"SCHL", "OCCP", "COW", "WKHP"},
        outcome={INCOME_OUTCOME},
    )
    validate_encoding(df, domain, roles)
    return df, domain, roles


def acs_codebook(dataset: str) -> Dict[str, object]:
    """Provenance and code -> label lists for a registered ACS dataset name."""
    task, state = dataset.split("_")
    return json.loads(_cache_paths(task, state.upper())[1].read_text())


#: States registered for ACSIncome. Adding one is this tuple plus the dataset
#: name in the DAG / role / training-settings registries -- the registry test
#: (`tests/test_dataset_registry.py`) names whichever of those is missing.
ACS_INCOME_STATES = ("CA", "TX")

ACS_DATASETS: Dict[str, Callable[[], LoadedDataset]] = {
    f"acsincome_{state.lower()}": partial(load_acs_income, state)
    for state in ACS_INCOME_STATES
}
