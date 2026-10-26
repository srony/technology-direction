"""
shared/utils.py
~~~~~~~~~~~~~~~
General-purpose helpers used across the ST4TE project.

Every path operation uses ``pathlib.Path`` so the code is portable
across Windows and macOS.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ------------------------------------------------------------------
# Filesystem helpers
# ------------------------------------------------------------------

def ensure_dir(p) -> None:
    """Create *p* (and parents) if it does not already exist."""
    Path(p).mkdir(parents=True, exist_ok=True)


def checkpoint_path(output_csv: Path) -> Path:
    """Return the checkpoint companion path for *output_csv*."""
    return Path(output_csv).with_suffix(".checkpoint.csv")


def write_checkpoint(df_out: pd.DataFrame, output_csv: Path) -> None:
    """Persist *df_out* as a checkpoint CSV next to *output_csv*."""
    cp = checkpoint_path(output_csv)
    cp.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(cp, index=False)


def load_checkpoint_if_any(output_csv: Path) -> Optional[pd.DataFrame]:
    """Load the checkpoint CSV if it exists, otherwise return ``None``."""
    cp = checkpoint_path(output_csv)
    if cp.exists():
        return pd.read_csv(cp)
    return None


# ------------------------------------------------------------------
# String / model-name helpers
# ------------------------------------------------------------------

def safe_alias(model_id: str) -> str:
    """Derive a filesystem-safe alias from an OpenRouter model id.

    Example::

        >>> safe_alias("openai/gpt-4o-mini")
        'gpt4omini'
    """
    alias = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    return re.sub(r"[^A-Za-z0-9]+", "", alias).lower()


def parse_int_or_none(x):
    """Try to cast *x* to ``int``; return ``None`` on failure."""
    try:
        return int(x)
    except Exception:
        return None


# ------------------------------------------------------------------
# CSV / JSON column parsers
# ------------------------------------------------------------------

def parse_csv_floats(s: str) -> List[float]:
    """Parse a comma-separated string of floats.

    Example::

        >>> parse_csv_floats("0.1, 0.5, 0.9")
        [0.1, 0.5, 0.9]
    """
    s = (s or "").strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_ints(s: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated string of ints (e.g. CLI ``--kmins "8,3,3"``).

    Returns ``None`` when *s* is ``None`` or blank.
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def parse_links_json(s: str) -> List[str]:
    """Deserialise a JSON array of link titles stored in a CSV cell.

    Gracefully returns ``[]`` on any parse error.
    """
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        x = json.loads(s)
        if isinstance(x, list):
            return [t for t in x if isinstance(t, str) and t.strip()]
        return []
    except Exception:
        return []


def dump_links_json(lst: List[str]) -> str:
    """Serialise a list of link titles to a compact JSON string."""
    return json.dumps(lst, ensure_ascii=False)


# ------------------------------------------------------------------
# Text extraction
# ------------------------------------------------------------------

def extract_tasks(text: str) -> Optional[List[str]]:
    """Pull lettered task items from O*NET-style DWA descriptions.

    Pattern matches ``(a) some task;`` and returns the task text.
    Returns ``None`` when nothing matches.
    """
    pattern = r'\([a-z]\)\s+([^;]+);'
    tasks = re.findall(pattern, text)
    return [task.strip() for task in tasks] if tasks else None
