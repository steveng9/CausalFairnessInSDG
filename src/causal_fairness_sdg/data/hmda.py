"""HMDA mortgage applications (Home Mortgage Disclosure Act, 2018 public
loan/application register, from the CFPB/FFIEC data browser): was this
application approved?

The lending dataset of the benchmark. Unlike the credit tables usually used
for fairness work (German Credit has 1,000 rows; Lending Club publishes no
protected attribute), HMDA is the record lenders are required by law to
file, it exists *because of* discrimination in lending, and it carries the
applicant's race, ethnicity, sex and age next to the underwriting variables
added in 2018 (debt-to-income, loan-to-value, loan term, the automated
underwriting system used). Registered per state (`hmda_ca`, `hmda_tx`), the
same two states and the same year as the ACS tables.

Population: applications for a first-lien, closed-end, non-commercial loan on
a one-unit, site-built, owner-occupied home, to buy it or to refinance it,
on which the lender made a credit decision (originated, approved but not
accepted, or denied). Withdrawn and incomplete files and purchased loans
never reach a decision and are not downloaded.

Outcome: `approved` -- originated, or approved and not accepted by the
applicant -- against denied. The positive class is the favourable one.

What is dropped, and what that costs:

  - Applications without the first applicant's sex, race, ethnicity or age
    (about 20% in both states: not provided in mail, internet or telephone
    applications, or not applicable). Their denial rate is higher (23%
    against 17% in California), so the table understates denial overall.
    They are dropped rather than kept as a level because every fairness
    metric here needs a group to assign the row to.
  - Applications with no reported income, loan-to-value, debt-to-income or
    loan term, or filed by an institution exempt from reporting them (about
    4% each, largely overlapping), and those with a non-positive income.
    These too are denied more often: an application abandoned at
    underwriting may never get a ratio computed.
  - Tracts with no income ratio (under 1%).

What the public file does not have: a credit score. Lenders report which
scoring model they used, not the score. Every analysis of denial disparities
on public HMDA carries this caveat, and so does this table -- a
race-to-approval dependence that survives conditioning on the columns here
is not, by itself, evidence of disparate treatment.

Everything is cut at fixed public edges (the regulatory thresholds where
there is one: 80% and 97% loan-to-value, 43% debt-to-income, the CRA tract
income classes), never at quantiles of the data. Race is the first race the
first applicant reported, with Asian and Pacific Islander subgroups folded
into their parent group.

Building a table parses a 300-470 MB download, so the encoded result is
cached under `data/processed/` beside a JSON codebook, as for the ACS tables.
"""

from __future__ import annotations

import json
import urllib.request
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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
    _map_codes,
    _sha256,
    _subsample,
    _to_codes,
    _write_atomically,
    validate_encoding,
)

HMDA_RAW_DIR = RAW_DIR / "hmda"
HMDA_YEAR = "2018"
HMDA_URL = (
    "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
    "?states={state}&years={year}&actions_taken=1,2,3"
)
HMDA_OUTCOME = "approved"
#: Bump when anything below changes what a row or a code means.
HMDA_ENCODING_VERSION = 1
#: Rows left by the filters in `_labelled`, pinned so a changed download or a
#: changed filter is noticed.
HMDA_EXPECTED_ROWS: Dict[str, int] = {"CA": 508_821, "TX": 382_657}

SITE_BUILT_SINGLE_FAMILY = "Single Family (1-4 Units):Site-Built"
NOT_REPORTED = ("Exempt", "NA", "U", "1111", "8888", "9999")

SEX_LEVELS = {1: "Male", 2: "Female"}
# First digit of `applicant_race-1`: 21-27 are Asian subgroups, 41-44 Pacific
# Islander subgroups.
RACE_LEVELS = {
    5: "White", 3: "Black", 2: "Asian", 1: "American Indian / Alaska Native",
    4: "Pacific Islander",
}
ETHNICITY_LEVELS = {2: "Not Hispanic", 1: "Hispanic"}  # 11-14: Hispanic subgroups
AGE_LEVELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
LOAN_TYPE_LEVELS = {1: "Conventional", 2: "FHA", 3: "VA", 4: "USDA"}
LOAN_PURPOSE_LEVELS = {1: "Purchase", 31: "Refinance", 32: "Cash-out refinance"}
AUS_LEVELS = {
    1: "Desktop Underwriter", 2: "Loan Prospector", 3: "FHA TOTAL Scorecard",
    4: "USDA GUS", 5: "Other", 6: "None",
}
APPROVED_LEVELS = {3: "Denied", 1: "Approved", 2: "Approved"}

INCOME_BINS = ([0, 30, 50, 75, 100, 150, 250, np.inf],
               ["<=30k", "30-50k", "50-75k", "75-100k", "100-150k", "150-250k", ">250k"])
LOAN_AMOUNT_BINS = (
    [0, 100_000, 150_000, 200_000, 300_000, 400_000, 600_000, np.inf],
    ["<=100k", "100-150k", "150-200k", "200-300k", "300-400k", "400-600k", ">600k"],
)
LTV_BINS = ([0, 60, 70, 80, 90, 95, 97, 100, np.inf],
            ["<=60", "60-70", "70-80", "80-90", "90-95", "95-97", "97-100", ">100"])
LOAN_TERM_BINS = ([0, 179, 180, 359, np.inf],
                  ["Under 15 years", "15 years", "16-29 years", "30 years or more"])
TRACT_MINORITY_BINS = ([-1, 10, 20, 50, 80, 100],
                       ["<=10%", "10-20%", "20-50%", "50-80%", ">80%"])
# CRA classes of tract median family income relative to the metro area's.
TRACT_INCOME_BINS = ([0, 49, 79, 119, np.inf],
                     ["Low (<50%)", "Moderate (50-79%)", "Middle (80-119%)", "Upper (120%+)"])
#: The file reports exact percentages only from 36 to 49 and bands elsewhere.
DTI_LEVELS = ["<20%", "20%-<30%", "30%-<36%", "36%-40%", "41%-43%", "44%-49%",
              "50%-60%", ">60%"]

RAW_COLUMNS = [
    "state_code", "action_taken", "loan_purpose", "lien_status", "occupancy_type",
    "derived_dwelling_category", "total_units", "reverse_mortgage",
    "open-end_line_of_credit", "business_or_commercial_purpose",
    "applicant_sex", "applicant_race-1", "applicant_ethnicity-1", "applicant_age",
    "co-applicant_sex", "income", "loan_amount", "loan_to_value_ratio",
    "debt_to_income_ratio", "loan_type", "loan_term", "conforming_loan_limit",
    "aus-1", "tract_minority_population_percent", "tract_to_msa_income_percentage",
]


def _raw_path(state: str) -> Path:
    return HMDA_RAW_DIR / f"hmda_{HMDA_YEAR}_{state}_actions123.csv"


def _download(state: str) -> Path:
    path = _raw_path(state)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_suffix(".csv.part")
        urllib.request.urlretrieve(HMDA_URL.format(state=state, year=HMDA_YEAR), scratch)
        scratch.replace(path)
    return path


def _debt_to_income(values: pd.Series) -> pd.Series:
    exact = pd.to_numeric(values, errors="coerce")
    banded = values.where(exact.isna())
    banded = banded.mask(exact.between(36, 40), "36%-40%")
    banded = banded.mask(exact.between(41, 43), "41%-43%")
    banded = banded.mask(exact.between(44, 49), "44%-49%")
    unknown = sorted(set(banded.unique()) - set(DTI_LEVELS))
    if unknown:
        raise ValueError(f"debt_to_income_ratio: values with no band: {unknown}")
    return pd.Series(
        pd.Categorical(banded, categories=DTI_LEVELS, ordered=True), index=values.index
    )


def _leading_digit(codes: pd.Series) -> pd.Series:
    return codes.str[0].astype("int64").rename(codes.name)


def _labelled(state: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """The filtered, labelled table for `state`, and how many rows each
    filter step left."""
    raw = pd.read_csv(_download(state), usecols=RAW_COLUMNS, dtype=str)
    if set(raw["state_code"].unique()) != {state}:
        raise ValueError(f"{_raw_path(state)}: states {sorted(raw['state_code'].unique())}")
    steps = {"downloaded (decided applications)": len(raw)}

    raw = raw[
        raw["loan_purpose"].isin([str(code) for code in LOAN_PURPOSE_LEVELS])
        & (raw["lien_status"] == "1")
        & (raw["occupancy_type"] == "1")
        & (raw["derived_dwelling_category"] == SITE_BUILT_SINGLE_FAMILY)
        & (raw["total_units"] == "1")
        & (raw["reverse_mortgage"] == "2")
        & (raw["open-end_line_of_credit"] == "2")
        & (raw["business_or_commercial_purpose"] == "2")
    ]
    steps["first-lien owner-occupied one-unit purchase or refinance"] = len(raw)

    race, ethnicity = raw["applicant_race-1"], raw["applicant_ethnicity-1"]
    raw = raw[
        raw["applicant_sex"].isin(["1", "2"])
        & race.str[0].isin([str(code) for code in RACE_LEVELS]) & (race.str.len() <= 2)
        & ethnicity.str[0].isin(["1", "2"]) & (ethnicity.str.len() <= 2)
        & raw["applicant_age"].isin(AGE_LEVELS)
    ]
    steps["first applicant's sex, race, ethnicity and age reported"] = len(raw)

    reported = pd.Series(True, index=raw.index)
    for col in ("income", "loan_to_value_ratio", "debt_to_income_ratio", "loan_term",
                "conforming_loan_limit", "aus-1", "co-applicant_sex"):
        reported &= raw[col].notna() & ~raw[col].isin(NOT_REPORTED)
    reported &= pd.to_numeric(raw["income"], errors="coerce") > 0
    reported &= pd.to_numeric(raw["tract_to_msa_income_percentage"], errors="coerce") > 0
    raw = raw[reported]
    steps["underwriting and tract variables reported"] = len(raw)

    def number(col: str) -> pd.Series:
        return pd.to_numeric(raw[col], errors="raise")

    def coded(col: str) -> pd.Series:
        return raw[col].astype("int64")

    yes_no = ["No", "Yes"]
    labelled = pd.DataFrame({
        "sex": _map_codes(coded("applicant_sex"), SEX_LEVELS),
        "race": _map_codes(_leading_digit(raw["applicant_race-1"]), RACE_LEVELS),
        "ethnicity": _map_codes(
            _leading_digit(raw["applicant_ethnicity-1"]), ETHNICITY_LEVELS
        ),
        "age": pd.Categorical(raw["applicant_age"], categories=AGE_LEVELS, ordered=True),
        # 5 = "no co-applicant"; anything else reported is a co-applicant.
        "coapplicant": pd.Categorical(
            np.where(raw["co-applicant_sex"] == "5", "No", "Yes"),
            categories=yes_no, ordered=True,
        ),
        "income": _cut(number("income"), INCOME_BINS),
        "tract_minority": _cut(
            number("tract_minority_population_percent"), TRACT_MINORITY_BINS
        ),
        "tract_income": _cut(number("tract_to_msa_income_percentage"), TRACT_INCOME_BINS),
        "loan_purpose": _map_codes(coded("loan_purpose"), LOAN_PURPOSE_LEVELS),
        "loan_amount": _cut(number("loan_amount"), LOAN_AMOUNT_BINS),
        "jumbo": pd.Categorical(
            np.where(raw["conforming_loan_limit"] == "NC", "Yes", "No"),
            categories=yes_no, ordered=True,
        ),
        "loan_type": _map_codes(coded("loan_type"), LOAN_TYPE_LEVELS),
        "loan_term": _cut(number("loan_term"), LOAN_TERM_BINS),
        "ltv": _cut(number("loan_to_value_ratio"), LTV_BINS),
        "dti": _debt_to_income(raw["debt_to_income_ratio"]),
        "aus": _map_codes(coded("aus-1"), AUS_LEVELS),
        HMDA_OUTCOME: _map_codes(coded("action_taken"), APPROVED_LEVELS),
    }, index=raw.index).reset_index(drop=True)
    if labelled.isna().any().any():
        raise ValueError(f"missing values after encoding: {labelled.isna().sum().to_dict()}")
    return labelled, steps


HMDA_ROLES = AttributeRoles.create(
    protected={"sex", "race", "ethnicity", "age"},
    admissible={"income", "dti", "ltv", "loan_amount", "loan_term"},
    outcome={HMDA_OUTCOME},
)


def _cache_paths(state: str) -> Tuple[Path, Path]:
    stem = (
        f"hmda_{state.lower()}_{HMDA_YEAR}_n{BIG_TABLE_ROWS}"
        f"_s{BIG_TABLE_SEED}_v{HMDA_ENCODING_VERSION}"
    )
    return PROCESSED_DIR / f"{stem}.parquet", PROCESSED_DIR / f"{stem}.json"


def build_hmda(state: str) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Download (if needed), filter, encode and subsample one state's table,
    and write it to the processed cache."""
    labelled, steps = _labelled(state)
    expected = HMDA_EXPECTED_ROWS.get(state)
    if expected is not None and len(labelled) != expected:
        raise ValueError(
            f"HMDA {state} {HMDA_YEAR}: {len(labelled)} rows after filtering, "
            f"expected {expected}"
        )
    df, codebook = _to_codes(_subsample(labelled))
    table_path, codebook_path = _cache_paths(state)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = _raw_path(state)
    provenance = {
        "source": f"HMDA {HMDA_YEAR} loan/application register, FFIEC data browser",
        "url": HMDA_URL.format(state=state, year=HMDA_YEAR),
        "state": state,
        "encoding_version": HMDA_ENCODING_VERSION,
        "rows_after_each_filter": steps,
        "rows_kept": int(len(df)),
        "subsample_seed": BIG_TABLE_SEED,
        "raw_file": raw_file.name,
        "raw_file_sha256": _sha256(raw_file),
        "positive_label": "Approved",
        "positive_rate_full": float((labelled[HMDA_OUTCOME] == "Approved").mean()),
        "positive_rate_kept": float(df[HMDA_OUTCOME].mean()),
        "levels": codebook,
    }
    _write_atomically(table_path, lambda p: df.to_parquet(p, index=False))
    _write_atomically(
        codebook_path, lambda p: p.write_text(json.dumps(provenance, indent=2))
    )
    return df, codebook


def load_hmda(state: str) -> LoadedDataset:
    """One state's mortgage applications: 16 attributes + outcome, 50k rows,
    built on first use.

    Default roles: the applicant's sex, race, ethnicity and age protected (all
    four are prohibited bases under the Equal Credit Opportunity Act); income,
    the two underwriting ratios and the size and length of the loan
    admissible. The tract variables are never admissible: where the property
    is, is the redlining channel.
    """
    table_path, _ = _cache_paths(state)
    df = pd.read_parquet(table_path) if table_path.exists() else build_hmda(state)[0]
    domain = {col: int(df[col].nunique()) for col in df.columns}
    validate_encoding(df, domain, HMDA_ROLES)
    return df, domain, HMDA_ROLES


HMDA_STATES = ("CA", "TX")
HMDA_REGISTRY: Dict[str, str] = {f"hmda_{state.lower()}": state for state in HMDA_STATES}
HMDA_DATASETS: Dict[str, Callable[[], LoadedDataset]] = {
    name: partial(load_hmda, state) for name, state in HMDA_REGISTRY.items()
}


def hmda_codebook(dataset: str) -> Dict[str, object]:
    """Provenance and code -> label lists for a registered HMDA dataset name."""
    return json.loads(_cache_paths(HMDA_REGISTRY[dataset])[1].read_text())


__all__: Tuple[str, ...] = (
    "HMDA_DATASETS", "HMDA_REGISTRY", "build_hmda", "hmda_codebook", "load_hmda",
)
