"""Encoding helpers shared by every dataset loader.

Every loader hands the rest of the project the same contract, and
`validate_encoding` is that contract written down: all-integer columns on
contiguous codes `0..k-1`, a `domain` that says exactly `k` for each, and
roles that name real columns. The DP synthesizers and DECAF's output heads
both read a column's domain as the range of values it may take, so a loader
that breaks the contract does not crash -- it quietly spends privacy budget
and generator capacity on categories that cannot occur.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import pandas as pd

from ..fairness.base import AttributeRoles

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
#: Encoded tables that are expensive to rebuild from `RAW_DIR` (see `acs.py`).
PROCESSED_DIR = DATA_DIR / "processed"

LoadedDataset = Tuple[pd.DataFrame, Dict[str, int], AttributeRoles]

#: Rows drawn from the big tables. Adult is ~45k after dropping NAs, so this
#: keeps every dataset within the same order of magnitude -- without it SNAKE
#: (201k) and SBO (161k) would make the GAN methods 4x slower than the rest of
#: the grid and quietly turn "which method is better" into "which method got
#: more data". Seeded, so the subsample is identical across every run.
BIG_TABLE_ROWS = 50_000
BIG_TABLE_SEED = 20260803


def _discretize(series: pd.Series, bins: int = 4) -> pd.Series:
    # rank(method="first") breaks ties so qcut can always produce `bins`
    # equal-sized buckets even for heavily-zero-inflated columns like
    # capital-gain/capital-loss.
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=bins, labels=False, duplicates="drop").astype(int)


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


def validate_encoding(
    df: pd.DataFrame, domain: Dict[str, int], roles: AttributeRoles
) -> None:
    """Raise `ValueError` listing every way `df` breaks the loader contract."""
    problems = []
    if set(domain) != set(df.columns):
        problems.append(
            f"domain/columns differ: {sorted(set(domain) ^ set(df.columns))}"
        )
    for col in df.columns:
        values = df[col]
        if not pd.api.types.is_integer_dtype(values):
            problems.append(f"{col}: dtype {values.dtype}, expected integer")
            continue
        k = int(values.nunique())
        if values.min() != 0 or values.max() != k - 1:
            problems.append(
                f"{col}: codes span {values.min()}..{values.max()} over {k} "
                "distinct values, expected contiguous 0..k-1"
            )
        if domain.get(col) != k:
            problems.append(f"{col}: declared domain {domain.get(col)}, observed {k}")
    named = roles.protected | roles.admissible | roles.outcome
    if named - set(df.columns):
        problems.append(f"roles name absent columns: {sorted(named - set(df.columns))}")
    for col in roles.outcome & set(df.columns):
        if df[col].nunique() != 2:
            problems.append(f"outcome {col}: {df[col].nunique()} classes, expected 2")
    if problems:
        raise ValueError("dataset breaks the encoding contract:\n  " + "\n  ".join(problems))


# --- Labelled encodings ----------------------------------------------------
# Loaders that map source codes to named levels build ordered pandas
# Categoricals with the helpers below, and integer-code them only at the end
# (`_to_codes`), so the code -> label correspondence is never implicit.

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


def _write_atomically(path: Path, write: Callable[[Path], None]) -> None:
    # Concurrent shards may all find the cache missing; a rename makes the
    # last writer win with a complete file instead of interleaving.
    scratch = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write(scratch)
    os.replace(scratch, path)
