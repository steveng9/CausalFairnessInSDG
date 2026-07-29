"""Dataset loaders. Protected/admissible/outcome role splits are reused
directly from PreFair (Pujol, Gilad, Machanavajjhala, VLDB'23) Table 1, so
results are comparable to that paper's numbers without re-deriving roles.

Raw files are cached under data/raw/ (gitignored) on first use.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable, Dict, Tuple

import pandas as pd

from ..fairness.base import AttributeRoles

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
    "compas-scores-two-years.csv"
)

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

LoadedDataset = Tuple[pd.DataFrame, Dict[str, int], AttributeRoles]


def _download(url: str, cache_path: Path) -> Path:
    if not cache_path.exists():
        urllib.request.urlretrieve(url, cache_path)
    return cache_path


def _discretize(series: pd.Series, bins: int = 4) -> pd.Series:
    # rank(method="first") breaks ties so qcut can always produce `bins`
    # equal-sized buckets even for heavily-zero-inflated columns like
    # capital-gain/capital-loss.
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=bins, labels=False, duplicates="drop").astype(int)


def load_adult() -> LoadedDataset:
    """Adult / Census Income (UCI, 14 attributes after dropping the `fnlwgt`
    sampling weight -- not a real attribute). Roles match PreFair Table 1."""
    path = _download(ADULT_URL, RAW_DIR / "adult.data")
    df = pd.read_csv(
        path, header=None, names=ADULT_COLUMNS, skipinitialspace=True, na_values="?"
    )
    df = df.drop(columns=["fnlwgt"]).dropna().reset_index(drop=True)

    for col in ("age", "capital-gain", "capital-loss", "hours-per-week"):
        df[col] = _discretize(df[col], bins=4)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes

    domain = {col: int(df[col].nunique()) for col in df.columns}
    roles = AttributeRoles.create(
        protected={"sex", "race", "native-country"},
        admissible={
            "workclass", "education", "occupation",
            "capital-gain", "capital-loss", "hours-per-week",
        },
        outcome={"income"},
    )
    return df, domain, roles


def load_compas() -> LoadedDataset:
    """COMPAS two-year recidivism (ProPublica), filtered with ProPublica's
    own standard criteria and reduced to the 8-attribute subset PreFair
    reports (~6172 rows after filtering). Roles match PreFair Table 1."""
    path = _download(COMPAS_URL, RAW_DIR / "compas-scores-two-years.csv")
    raw = pd.read_csv(path)

    raw = raw[
        (raw["days_b_screening_arrest"] <= 30)
        & (raw["days_b_screening_arrest"] >= -30)
        & (raw["is_recid"] != -1)
        & (raw["c_charge_degree"] != "O")
        & (raw["score_text"] != "N/A")
    ]

    keep = [
        "sex", "race", "age_cat", "juv_fel_count", "juv_misd_count",
        "priors_count", "c_charge_degree", "two_year_recid",
    ]
    df = raw[keep].dropna().reset_index(drop=True)

    df["priors_count"] = _discretize(df["priors_count"], bins=4)
    for col in ("juv_fel_count", "juv_misd_count"):
        df[col] = (df[col] > 0).astype(int)
    for col in ("sex", "race", "age_cat", "c_charge_degree"):
        df[col] = df[col].astype("category").cat.codes

    domain = {col: int(df[col].nunique()) for col in df.columns}
    roles = AttributeRoles.create(
        protected={"sex", "race"},
        admissible={"priors_count", "c_charge_degree"},
        outcome={"two_year_recid"},
    )
    return df, domain, roles


DATASETS: Dict[str, Callable[[], LoadedDataset]] = {
    "adult": load_adult,
    "compas": load_compas,
}
