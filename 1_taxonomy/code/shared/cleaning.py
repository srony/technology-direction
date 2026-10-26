"""
shared/cleaning.py
~~~~~~~~~~~~~~~~~~
Data-cleaning helpers shared across all ST4TE sub-projects.

Every path operation uses ``pathlib.Path`` so the code is portable
across Windows and macOS.
"""

from __future__ import annotations

import re
from typing import List

import pandas as pd


# ------------------------------------------------------------------
# Identifier construction
# ------------------------------------------------------------------

def add_identifier(df: pd.DataFrame, KEY) -> pd.DataFrame:
    """Build a deterministic ``identifier`` column from one or more key columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    KEY : str or list[str]
        Column name(s) used to construct the identifier.  Values are
        lowered, stripped and non-alphanumeric characters removed; when
        multiple columns are given, they are joined with ``||``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* (rows with NA keys dropped) with an extra
        ``identifier`` column.
    """
    out = df.dropna(subset=KEY).copy()

    temp = (
        out[KEY].astype(str)
                 .apply(lambda s: s.str.strip().str.lower()
                                   .str.replace(r"[^a-z0-9]", "", regex=True))
    )
    out["identifier"] = temp.agg("||".join, axis=1)
    return out


# ------------------------------------------------------------------
# Regex compilation helper
# ------------------------------------------------------------------

def compile_any(patterns: List[str]) -> re.Pattern:
    """Compile a list of regex patterns into a single OR pattern."""
    return re.compile("|".join(f"(?:{p})" for p in patterns), flags=re.IGNORECASE)


# ------------------------------------------------------------------
# Noise / relevance filtering (shared by P2 taxonomy scripts)
# ------------------------------------------------------------------

def flag_noise_pages(
    df: pd.DataFrame,
    title_col: str,
    desc_col: str,
    noise_title_patterns: List[str],
    bad_path_patterns: List[str] | None = None,
    noise_title_only_patterns: List[str] | None = None,
    noise_desc_only_patterns: List[str] | None = None,
    exclusion_titles: set | None = None,
) -> pd.DataFrame:
    """Flag structural-noise Wikipedia pages based on title/description patterns.

    Adds a boolean column ``flag_noise_structural`` to the returned copy.

    ``noise_title_patterns`` are matched against the combined title + description.
    ``noise_title_only_patterns`` are matched against the title only.
    ``noise_desc_only_patterns`` are matched against the description only.
    """
    df = df.copy()
    titles = df[title_col].astype(str).fillna("")
    descs = (
        df[desc_col].astype(str).fillna("") if desc_col in df.columns
        else pd.Series([""] * len(df))
    )

    flags = []
    for i, (t, d) in enumerate(zip(titles.tolist(), descs.tolist())):
        text = (t + " " + d).lower()
        is_noise = False

        for pat in noise_title_patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                is_noise = True
                break

        if not is_noise and noise_title_only_patterns:
            t_lower = t.lower()
            for pat in noise_title_only_patterns:
                if re.search(pat, t_lower, flags=re.IGNORECASE):
                    is_noise = True
                    break

        if not is_noise and noise_desc_only_patterns:
            d_lower = d.lower()
            for pat in noise_desc_only_patterns:
                if re.search(pat, d_lower, flags=re.IGNORECASE):
                    is_noise = True
                    break

        if not is_noise and bad_path_patterns and "category_links" in df.columns:
            cats = df.iloc[i]["category_links"]
            if isinstance(cats, str):
                cats_text = cats.lower()
            elif isinstance(cats, (list, tuple)):
                cats_text = " ".join(str(c).lower() for c in cats)
            else:
                cats_text = ""
            for pat in bad_path_patterns:
                if re.search(pat, cats_text, flags=re.IGNORECASE):
                    is_noise = True
                    break

        if not is_noise and exclusion_titles and t in exclusion_titles:
            is_noise = True

        flags.append(is_noise)

    df["flag_noise_structural"] = flags
    return df


def relevance_score(blob: str, mode: str, green_pat: re.Pattern, digital_pat: re.Pattern) -> float:
    """Count keyword matches for green / digital relevance scoring."""
    if mode == "green":
        return float(len(green_pat.findall(blob)))
    if mode == "digital":
        return float(len(digital_pat.findall(blob)))
    return float(len(green_pat.findall(blob)) + len(digital_pat.findall(blob)))


def clean_description_text(text: str) -> str:
    """Normalize whitespace in a description string."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def print_stats(df: pd.DataFrame, stage: str, prev_rows: int | None = None) -> int:
    """Print row-count statistics for a pipeline stage."""
    n = int(len(df))
    u = int(df["page_title"].nunique()) if "page_title" in df.columns else 0
    if prev_rows is None:
        print(f"[STATS] {stage:<36} rows={n:,} unique_pages={u:,}")
    else:
        dropped = int(prev_rows - n)
        retained = (n / prev_rows * 100.0) if prev_rows else 0.0
        print(f"[STATS] {stage:<36} rows={n:,} unique_pages={u:,} dropped={dropped:,} retained={retained:.2f}%")
    return n
