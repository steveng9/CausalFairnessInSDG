"""CDC Diabetes Health Indicators (UCI id 891): 253,680 respondents to the
2015 Behavioral Risk Factor Surveillance System, 21 health and demographic
indicators, and whether the respondent has (pre)diabetes.

Here for its shape as much as its domain: 15 of the 22 columns are yes/no,
the outcome is rare (14% positive, against 36-58% on the other tables), and
there is no race column -- the sensitive attributes are sex and socioeconomic
status (income, education).

The file is already integer-coded by its publishers. Only three columns are
re-binned, at fixed clinical cut-points rather than quantiles: `BMI` to the
WHO classes, and the two "days unwell in the past 30" counts to
none / 1-13 / 14+ (14 is the CDC's "frequent distress" threshold). The lowest
two education levels are merged (the lowest has 174 respondents).
"""

from __future__ import annotations

import urllib.request
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..fairness.base import AttributeRoles
from ._encoding import (
    RAW_DIR,
    LoadedDataset,
    _cut,
    _map_codes,
    _subsample,
    _to_codes,
    validate_encoding,
)

CDC_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
CDC_PATH = RAW_DIR / "cdc_diabetes" / "data.csv"
CDC_ROWS = 253_680
CDC_OUTCOME = "Diabetes_binary"

NO_YES = {0: "No", 1: "Yes"}
BINARY_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk",
]
SEX_LEVELS = {0: "Female", 1: "Male"}
AGE_LEVELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79",
    13: "80+",
}
EDUCATION_LEVELS = {
    1: "Elementary or less", 2: "Elementary or less", 3: "Some high school",
    4: "High school graduate", 5: "Some college", 6: "College graduate",
}
INCOME_LEVELS = {
    1: "<10k", 2: "10-15k", 3: "15-20k", 4: "20-25k", 5: "25-35k",
    6: "35-50k", 7: "50-75k", 8: "75k+",
}
GENHLTH_LEVELS = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
BMI_BINS = ([0, 18, 24, 29, 34, 39, np.inf],
            ["<18.5", "18.5-24.9", "25-29.9", "30-34.9", "35-39.9", "40+"])
DAYS_UNWELL_BINS = ([-1, 0, 13, 30], ["0", "1-13", "14-30"])


def _labelled() -> pd.DataFrame:
    if not CDC_PATH.exists():
        CDC_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CDC_URL, CDC_PATH)
    raw = pd.read_csv(CDC_PATH)
    if len(raw) != CDC_ROWS or raw.isna().any().any():
        raise ValueError(
            f"{CDC_PATH}: {len(raw)} rows / {int(raw.isna().sum().sum())} missing "
            f"values, expected {CDC_ROWS} / 0"
        )
    columns = {
        "Sex": _map_codes(raw["Sex"], SEX_LEVELS),
        "Age": _map_codes(raw["Age"], AGE_LEVELS),
        "Education": _map_codes(raw["Education"], EDUCATION_LEVELS),
        "Income": _map_codes(raw["Income"], INCOME_LEVELS),
        "BMI": _cut(raw["BMI"], BMI_BINS),
        "GenHlth": _map_codes(raw["GenHlth"], GENHLTH_LEVELS),
        "MentHlth": _cut(raw["MentHlth"], DAYS_UNWELL_BINS),
        "PhysHlth": _cut(raw["PhysHlth"], DAYS_UNWELL_BINS),
        **{col: _map_codes(raw[col], NO_YES) for col in BINARY_COLUMNS},
        CDC_OUTCOME: _map_codes(raw[CDC_OUTCOME], NO_YES),
    }
    return pd.DataFrame(columns)


def cdc_diabetes_codebook() -> Dict[str, List[str]]:
    """Code -> label list for every column of the table `load_cdc_diabetes`
    returns."""
    return _to_codes(_subsample(_labelled()))[1]


def load_cdc_diabetes() -> LoadedDataset:
    """CDC Diabetes Health Indicators, 21 attributes + outcome, 50k rows.

    Default roles: sex and socioeconomic status protected, the clinical risk
    factors a screening guideline would name (and age) admissible.
    """
    df, _codebook = _to_codes(_subsample(_labelled()))
    domain = {col: int(df[col].nunique()) for col in df.columns}
    roles = AttributeRoles.create(
        protected={"Sex", "Income", "Education"},
        admissible={"Age", "BMI", "HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke"},
        outcome={CDC_OUTCOME},
    )
    validate_encoding(df, domain, roles)
    return df, domain, roles


CDC_DATASETS = {"cdc_diabetes": load_cdc_diabetes}

__all__: Tuple[str, ...] = ("CDC_DATASETS", "load_cdc_diabetes", "cdc_diabetes_codebook")
