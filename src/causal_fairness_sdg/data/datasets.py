"""Dataset loaders. Protected/admissible/outcome role splits are reused
directly from PreFair (Pujol, Gilad, Machanavajjhala, VLDB'23) Table 1, so
results are comparable to that paper's numbers without re-deriving roles.

Raw files are cached under data/raw/ (gitignored) on first use.
"""

from __future__ import annotations

import os
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


# Both of the larger tables live outside this repo (they are not ours to
# redistribute), so their locations are overridable rather than downloaded.
SNAKE_PATH = Path(
    os.environ.get("SNAKE_PATH", "/home/golobs/SyntheticData_MIA/SNAKE/base.parquet")
)
SBO_PATH = Path(
    os.environ.get(
        "SBO_PATH", "/home/golobs/data/reconstruction_data/nist_sbo/full_data.csv"
    )
)

#: Rows drawn from the big tables. Adult is ~45k after dropping NAs, so this
#: keeps every dataset within the same order of magnitude -- without it SNAKE
#: (201k) and SBO (161k) would make the GAN methods 4x slower than the rest of
#: the grid and quietly turn "which method is better" into "which method got
#: more data". Seeded, so the subsample is identical across every run.
BIG_TABLE_ROWS = 50_000
BIG_TABLE_SEED = 20260803


def _subsample(df: pd.DataFrame, n: int = BIG_TABLE_ROWS) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=BIG_TABLE_SEED).reset_index(drop=True)


def _recode(df: pd.DataFrame) -> pd.DataFrame:
    """Force every column onto contiguous codes 0..k-1.

    Not cosmetic. The declared domain is `nunique`, and both the DP
    synthesizers and DECAF's output heads treat a column's domain as the range
    of values it may take -- so a column carrying its source codes (SBO's
    `SECTOR` runs 11..99 over 20 real values) declares a domain of 100 and
    spends privacy budget, and generator capacity, on 80 categories that can
    never occur. Adult's `education-num` has exactly this bug today; it is
    left alone there only because every published number in the report was
    produced under it, and silently re-encoding would invalidate the
    comparison rather than fix it.
    """
    for col in df.columns:
        df[col] = df[col].astype("category").cat.codes.astype(int)
    return df


def load_snake() -> LoadedDataset:
    """SNAKE (Current Population Survey extract), 15 attributes.

    Sits between COMPAS (8) and SBO (25) on the axis this project cares about
    -- how large a causal graph the fairness mechanisms have to reason over --
    while staying a familiar income-prediction task, so the fairness numbers
    remain interpretable next to Adult's.

    `faminc` is binarised into the outcome at the $75k break (its own category
    boundary, not a quantile) and then dropped, exactly as Adult's `income` is
    already a binarised earnings variable. The high-cardinality numerics are
    quartile-binned like Adult's, and `ownchild` is capped at 3+ because past
    three children the cells are too thin for a 4-way marginal to mean much.
    """
    if not SNAKE_PATH.exists():
        raise FileNotFoundError(
            f"SNAKE base table not found at {SNAKE_PATH}; set $SNAKE_PATH."
        )
    df = pd.read_parquet(SNAKE_PATH)

    # 12 = "$75,000 - $99,999" in the meta.json ordering; >= that is the
    # top three brackets, giving a ~30% positive rate (Adult's is ~25%).
    order = list(df["faminc"].cat.categories) if hasattr(df["faminc"], "cat") else None
    codes = df["faminc"].cat.codes if order else df["faminc"].astype("category").cat.codes
    df = df.drop(columns=["faminc"])
    df["high_income"] = (codes >= 12).astype(int)

    for col in ("age", "hoursut"):
        df[col] = _discretize(df[col], bins=4)
    df["ownchild"] = df["ownchild"].clip(upper=3).astype(int)

    for col in df.columns:
        if str(df[col].dtype) in ("category", "object"):
            df[col] = df[col].astype("category").cat.codes
    df = _recode(df.dropna().reset_index(drop=True))
    df = _subsample(df)

    domain = {col: int(df[col].nunique()) for col in df.columns}
    roles = AttributeRoles.create(
        protected={"female", "wbhaom", "citistat"},
        admissible={"gradeatn", "mocc10", "mind16", "cow1", "hoursut", "ftptstat"},
        outcome={"high_income"},
    )
    return df, domain, roles


#: The 25 SBO columns kept, grouped by the role they play in the graph. The
#: full table has 133, but ~100 of those are yes/no funding-source and
#: language flags that are near-constant and would add nodes to the causal
#: graph without adding structure -- the point of including SBO is a *larger
#: graph*, not a wider table.
SBO_PROTECTED = ["SEX1", "RACE1", "ETH1", "VET1", "BORNUS1"]
SBO_OWNER = ["EDUC1", "AGE1", "HOURS1", "PRMINC1", "SELFEMP1"]
SBO_STRUCTURE = [
    "SECTOR", "FIPST", "ESTABLISHED", "HOMEBASED", "FRANCHISE",
    "NUMOWNERS", "FAMILYBUS",
]
SBO_PRACTICE = ["WEBSITE", "ECOMMERCE", "EXPORTS", "HEALTHINS", "RETIREMENT"]
SBO_SCALE = ["EMPLOYMENT_NOISY", "PAYROLL_NOISY"]
SBO_COLUMNS = (
    SBO_PROTECTED + SBO_OWNER + SBO_STRUCTURE + SBO_PRACTICE + SBO_SCALE
)


def load_sbo() -> LoadedDataset:
    """NIST Survey of Business Owners extract, 25 attributes.

    The largest causal graph in the project, and the reason it is here: the
    fairness mechanisms all act by *cutting or rerouting edges*, so their
    behaviour should depend on how much graph there is to cut. Adult (14) and
    COMPAS (8) cannot distinguish "this mechanism works" from "this graph was
    small enough that every path was direct".

    It is also a genuinely different fairness question -- the protected
    attributes describe the business *owner* while the outcome describes the
    *business*, so a demographic-parity gap here is a statement about access to
    capital and markets rather than about individual treatment.

    `RECEIPTS_NOISY` (already noise-infused by the NIST release) is binarised
    at its median into `high_receipts`. Roughly 23% of rows carry no owner
    detail at all; those are dropped rather than imputed, since imputing the
    protected attributes would manufacture exactly the association the
    fairness metrics are trying to measure.
    """
    if not SBO_PATH.exists():
        raise FileNotFoundError(
            f"SBO table not found at {SBO_PATH}; set $SBO_PATH."
        )
    df = pd.read_csv(SBO_PATH, low_memory=False)

    receipts = pd.to_numeric(df["RECEIPTS_NOISY"], errors="coerce")
    df = df[SBO_COLUMNS].copy()
    df["high_receipts"] = (receipts > receipts.median()).astype(int)
    df = df.dropna().reset_index(drop=True)

    # RACE1 carries 18 values, most of them multi-race combinations with a
    # handful of rows each. Keeping the long tail would give the DP methods a
    # near-empty marginal to spend budget on; collapsing to the five reported
    # single races plus "other" keeps every cell populated.
    top_race = ["W", "B", "A", "I", "P"]
    df["RACE1"] = df["RACE1"].where(df["RACE1"].isin(top_race), "OTHER")
    # FIPST arrives with mixed int/str types from the CSV.
    df["FIPST"] = df["FIPST"].astype(str)

    for col in SBO_SCALE:
        df[col] = _discretize(pd.to_numeric(df[col], errors="coerce"), bins=4)
    for col in df.columns:
        if str(df[col].dtype) in ("category", "object", "float64"):
            df[col] = df[col].astype("category").cat.codes
    df = _recode(df.dropna().reset_index(drop=True))
    df = _subsample(df)

    domain = {col: int(df[col].nunique()) for col in df.columns}
    roles = AttributeRoles.create(
        protected={"SEX1", "RACE1", "ETH1", "VET1", "BORNUS1"},
        admissible=set(SBO_OWNER) | {"SECTOR", "ESTABLISHED", "NUMOWNERS"},
        outcome={"high_receipts"},
    )
    return df, domain, roles


DATASETS: Dict[str, Callable[[], LoadedDataset]] = {
    "adult": load_adult,
    "compas": load_compas,
    "snake": load_snake,
    "sbo": load_sbo,
}
