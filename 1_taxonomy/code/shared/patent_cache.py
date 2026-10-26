"""
shared/patent_cache.py
~~~~~~~~~~~~~~~~~~~~~~
Shared patent corpus caching and embedding pre-computation.

Centralises logic previously duplicated across
``p2_taxonomy/data_extraction_patent_topics.py`` and
``p2_taxonomy/cleaning_patent_wiki_match.py``.

Provides:
- ``ingest_corpus``              -- TSV to per-mode parquet (ALL columns)
- ``compute_and_cache_embeddings`` -- pre-compute embeddings for multiple views
- ``load_embeddings``            -- load cached embeddings
- Text / CPC helpers used by the rest of the pipeline
- ``device_auto``                -- CUDA / CPU device selection
- ``batch_encode``               -- batch SentenceTransformer encoding (torch tensors)

Corpus filter (v2, applied unconditionally in ``ingest_corpus``):
    Drops rows where ``country`` is in ``COUNTRY_FILTER_BAD_SET`` (CN/TW/RU)
    *unless* ``cpc_details`` contains at least one country code different from
    the row's own ``country`` -- i.e., another patent office also classified
    the family. Rationale: CN/TW/RU rows whose CPC info is empty or self-only
    contribute no usable cross-jurisdictional CPC signal. v2 outputs live at
    ``_corpus_cache_v2/`` and ``_embedding_cache_v2/``; old caches are
    preserved as snapshots.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from shared.utils import ensure_dir

# Optional pyarrow for fast row counts
try:
    import pyarrow.parquet as pq
    _HAVE_PQ = True
except Exception:
    _HAVE_PQ = False


# ------------------------------------------------------------------
# Device detection
# ------------------------------------------------------------------

def device_auto() -> str:
    """Return ``'cuda'`` when a GPU is available, otherwise ``'cpu'``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# Batch encoding (torch tensors)
# ------------------------------------------------------------------

def batch_encode(texts, model, batch_size: int = 64):
    """Encode *texts* in batches with a SentenceTransformer *model*.

    Returns a single ``torch.Tensor`` of shape ``(len(texts), dim)``.
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_emb = model.encode(texts[i : i + batch_size], convert_to_tensor=True)
        embeddings.append(batch_emb)
    return torch.cat(embeddings) if embeddings else torch.empty(0)


# ------------------------------------------------------------------
# Default paths
# ------------------------------------------------------------------

INPUT_TSV = r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\patent\state_patent_export.tsv"
CORPUS_ROOT = r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\patent\_corpus_cache_v2"
CACHE_ROOT = r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\patent\_embedding_cache_v2"

DEFAULT_EMBED_MODEL = "AI-Growth-Lab/PatentSBERTa"
CHUNKSIZE = 300_000
EMBED_BATCH = 64

# Countries whose patents are kept only when cpc_details shows another office
# also classified the family (interpretation B). See module docstring.
COUNTRY_FILTER_BAD_SET = frozenset({"CN", "TW", "RU"})


# ------------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------------

def normalize_text(s: pd.Series) -> pd.Series:
    """Collapse whitespace in a Series of strings."""
    s = s.fillna("").astype(str)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def build_view_text(df: pd.DataFrame, view: str) -> pd.Series:
    """Build a text column from the chosen *view* (title/abstract/combined)."""
    t = normalize_text(df["appln_title"])
    a = normalize_text(df["appln_abstract"])
    if view == "title":
        return t
    if view == "abstract":
        return a
    if view == "combined":
        return (t + ". " + a).str.strip()
    raise ValueError(f"Unknown view: {view}")


# ------------------------------------------------------------------
# CPC helpers
# ------------------------------------------------------------------

def extract_cpc_codes_before_space(cpc_details: str) -> List[str]:
    """Parse CPC codes from a semicolon-separated string."""
    if not isinstance(cpc_details, str):
        return []
    txt = cpc_details.strip()
    if not txt:
        return []
    codes: List[str] = []
    for seg in txt.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        sp = seg.find(" ")
        if sp == -1:
            bar = seg.find("|")
            code = seg[:bar] if bar != -1 else seg
        else:
            code = seg[:sp]
        code = code.strip().upper()
        if code:
            codes.append(code)
    return codes


def coarse_prefix(code: str) -> str:
    """Return the first 4 chars of a CPC code (special-case H01L)."""
    code = (code or "").upper().strip()
    if code.startswith("H01L"):
        return "H01L"
    return code[:4] if len(code) >= 3 else code


def _extract_cpc_country_codes(cpc_details: str) -> frozenset:
    """Return the set of country codes encoded in ``cpc_details``.

    Format of each segment is ``<CPC_code>|<COUNTRY>|<I/A>|<date>``.
    """
    if not isinstance(cpc_details, str) or not cpc_details:
        return frozenset()
    out = set()
    for seg in cpc_details.split(";"):
        parts = seg.split("|")
        if len(parts) >= 2:
            c = parts[1].strip()
            if c:
                out.add(c)
    return frozenset(out)


# ------------------------------------------------------------------
# Part-file helpers
# ------------------------------------------------------------------

def _list_part_files(parts_dir: str) -> List[Path]:
    files = sorted(Path(parts_dir).glob("part_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet parts found in: {parts_dir}")
    return files


def _iter_parts(parts: List[Path], columns: List[str]) -> Iterable[pd.DataFrame]:
    for p in parts:
        yield pd.read_parquet(p, columns=columns)


def _fast_row_count(parts: List[Path]) -> int:
    if _HAVE_PQ:
        return sum(
            pq.ParquetFile(str(p)).metadata.num_rows for p in parts
        )
    return sum(len(pd.read_parquet(p, columns=["appln_id"])) for p in parts)


# ------------------------------------------------------------------
# 1) Ingest: TSV -> per-mode parquet with ALL columns
# ------------------------------------------------------------------

def ingest_corpus(
    mode: str,
    input_tsv: str = INPUT_TSV,
    cache_root: str = CORPUS_ROOT,
    chunksize: int = CHUNKSIZE,
) -> str:
    """Convert the raw TSV to per-mode parquet parts preserving ALL columns.

    Parameters
    ----------
    mode : str
        ``"green"`` or ``"digital"`` -- rows are filtered by this flag column.
    input_tsv : str
        Path to the raw patent TSV export.
    cache_root : str
        Root directory for corpus caches. Parts are written to
        ``{cache_root}/{mode}/part_XXXXX.parquet``.
    chunksize : int
        Number of TSV rows per read chunk.

    Returns
    -------
    str
        The parts directory path (``{cache_root}/{mode}``).
    """
    parts_dir = os.path.join(cache_root, mode)
    ensure_dir(parts_dir)

    existing = sorted(Path(parts_dir).glob("part_*.parquet"))
    if existing:
        print(f"[ingest] Found {len(existing)} existing parts for mode={mode}. Skipping.")
        return parts_dir

    print(f"[ingest] Reading TSV: {input_tsv}")
    flag = mode  # column name matches mode ("green" or "digital")

    reader = pd.read_csv(
        input_tsv, sep="\t", encoding="utf-8", low_memory=False,
        chunksize=chunksize, on_bad_lines="skip",
    )

    part_idx = 0
    kept = 0
    dropped_country_filter = 0
    for chunk in tqdm(reader, desc=f"[ingest] chunks ({mode})"):
        # Filter to rows where the mode flag == 1
        if flag not in chunk.columns:
            raise ValueError(
                f"Column '{flag}' not found in TSV. Available: {list(chunk.columns)[:20]}"
            )
        chunk = chunk[chunk[flag] == 1]
        if chunk.empty:
            continue

        # Country filter (interpretation B): for rows whose `country` is in
        # COUNTRY_FILTER_BAD_SET, keep only when `cpc_details` contains at
        # least one country code different from the row's own `country`.
        if "country" in chunk.columns and "cpc_details" in chunk.columns:
            in_bad = chunk["country"].isin(COUNTRY_FILTER_BAD_SET)
            if in_bad.any():
                bad_rows = chunk.loc[in_bad, ["country", "cpc_details"]]
                keep_bad = [
                    any(c != cc for c in _extract_cpc_country_codes(cpc))
                    for cc, cpc in zip(bad_rows["country"].values, bad_rows["cpc_details"].values)
                ]
                drop_mask = in_bad.copy()
                drop_mask.loc[in_bad] = [not k for k in keep_bad]
                dropped_country_filter += int(drop_mask.sum())
                chunk = chunk.loc[~drop_mask]
                if chunk.empty:
                    continue

        # Derive cpc_coarse_list from cpc_details (if present)
        if "cpc_details" in chunk.columns:
            cpcs = chunk["cpc_details"].apply(extract_cpc_codes_before_space)
            coarse = cpcs.apply(lambda lst: sorted(set(coarse_prefix(x) for x in lst if x)))
            chunk = chunk.copy()
            chunk["cpc_coarse_list"] = coarse.apply(lambda xs: "|".join(xs)).values

        out_path = os.path.join(parts_dir, f"part_{part_idx:05d}.parquet")
        chunk.to_parquet(out_path, index=False)
        part_idx += 1
        kept += len(chunk)

    print(
        f"[ingest] DONE mode={mode} kept={kept:,} parts={part_idx} "
        f"dropped_country_filter={dropped_country_filter:,}"
    )
    return parts_dir


# ------------------------------------------------------------------
# 2) Embedding cache: compute and store per-view embeddings
# ------------------------------------------------------------------

def _embedding_cache_paths(mode: str, view: str, embed_model: str, cache_root: str = CACHE_ROOT) -> Dict[str, str]:
    """Return file paths for a cached embedding set."""
    raw = f"{mode}|{view}|{embed_model}"
    cid = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    base = os.path.join(cache_root, f"emb_{cid}")
    return {
        "meta": base + ".meta.json",
        "ids": base + ".appln_id.npy",
        "emb": base + ".embeddings.npy",
    }


def compute_and_cache_embeddings(
    mode: str,
    cache_root: str = CACHE_ROOT,
    corpus_root: str = CORPUS_ROOT,
    embed_model: str = DEFAULT_EMBED_MODEL,
    views: Optional[List[str]] = None,
    embed_batch: int = EMBED_BATCH,
) -> None:
    """Pre-compute and cache L2-normalised embeddings for requested views.

    Parameters
    ----------
    mode : str
        ``"green"`` or ``"digital"``.
    cache_root : str
        Directory for embedding cache files.
    corpus_root : str
        Root corpus cache (parts live at ``{corpus_root}/{mode}/``).
    embed_model : str
        SentenceTransformer model id.
    views : list[str] or None
        Views to compute; defaults to ``["title", "abstract", "combined"]``.
    embed_batch : int
        Encoding batch size.
    """
    from sentence_transformers import SentenceTransformer

    if views is None:
        views = ["title", "abstract", "combined"]

    parts_dir = os.path.join(corpus_root, mode)
    ensure_dir(cache_root)
    parts = _list_part_files(parts_dir)
    n_rows = _fast_row_count(parts)

    dev = device_auto()
    embedder = SentenceTransformer(embed_model, device=dev)
    dim = int(embedder.get_sentence_embedding_dimension())

    for view in views:
        paths = _embedding_cache_paths(mode, view, embed_model, cache_root)

        # Check cache validity
        if os.path.exists(paths["meta"]) and os.path.exists(paths["ids"]) and os.path.exists(paths["emb"]):
            try:
                meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
                if (
                    meta.get("rows") == n_rows
                    and meta.get("mode") == mode
                    and meta.get("view") == view
                    and meta.get("embed_model") == embed_model
                ):
                    ids = np.load(paths["ids"], allow_pickle=False)
                    emb = np.load(paths["emb"], mmap_mode="r")
                    if len(ids) == n_rows and emb.shape[0] == n_rows:
                        print(f"[embed-cache] HIT view={view}: {paths['emb']}")
                        del ids, emb
                        continue
            except Exception:
                pass

        # -- Part-level checkpointing for crash recovery --
        # A small progress file records (cursor, part_index) after each
        # part is fully encoded and flushed.  On restart we reopen the
        # existing memmap / ids array and skip already-finished parts.
        progress_path = paths["emb"] + ".progress.json"
        resume_cursor = 0
        resume_part = 0

        if os.path.exists(progress_path):
            try:
                prog = json.loads(Path(progress_path).read_text(encoding="utf-8"))
                if (
                    prog.get("n_rows") == n_rows
                    and prog.get("mode") == mode
                    and prog.get("view") == view
                    and prog.get("embed_model") == embed_model
                ):
                    resume_cursor = prog["cursor"]
                    resume_part = prog["part_index"]
                    print(f"[embed-cache] RESUMING view={view} from part {resume_part}/{len(parts)} "
                          f"(cursor={resume_cursor:,}/{n_rows:,})")
            except Exception:
                pass

        if resume_cursor > 0:
            # Reopen existing files instead of overwriting
            ids_all = np.load(paths["ids"], allow_pickle=False).copy()
            emb_mm = np.lib.format.open_memmap(paths["emb"], mode="r+")
        else:
            print(f"[embed-cache] MISS view={view}. Computing for {n_rows:,} docs.")
            ids_all = np.empty((n_rows,), dtype=np.int64)
            emb_mm = np.lib.format.open_memmap(
                paths["emb"], mode="w+", dtype=np.float32, shape=(n_rows, dim),
            )

        cursor = 0
        for part_i, df in enumerate(tqdm(
            _iter_parts(parts, ["appln_id", "appln_title", "appln_abstract"]),
            total=len(parts), desc=f"[embed] parts ({view})",
        )):
            n = len(df)
            if part_i < resume_part:
                # Already encoded — just advance cursor
                cursor += n
                continue

            texts = build_view_text(df, view).tolist()
            ids = df["appln_id"].astype(np.int64).values
            embs = embedder.encode(
                texts, batch_size=embed_batch, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True,
            ).astype(np.float32, copy=False)

            ids_all[cursor : cursor + n] = ids
            emb_mm[cursor : cursor + n, :] = embs
            cursor += n

            # Flush after every part and write progress checkpoint
            emb_mm.flush()
            np.save(paths["ids"], ids_all, allow_pickle=False)
            Path(progress_path).write_text(json.dumps({
                "mode": mode, "view": view, "embed_model": embed_model,
                "n_rows": n_rows, "cursor": cursor, "part_index": part_i + 1,
            }), encoding="utf-8")

            del df, texts, ids, embs
            gc.collect()

        emb_mm.flush()
        np.save(paths["ids"], ids_all, allow_pickle=False)

        meta = {
            "mode": mode,
            "view": view,
            "embed_model": embed_model,
            "rows": int(n_rows),
            "dim": int(dim),
            "normalized": True,
        }
        Path(paths["meta"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Clean up progress file — no longer needed
        if os.path.exists(progress_path):
            os.remove(progress_path)

        print(f"[embed-cache] Saved view={view}: {paths['emb']}")

    del embedder
    gc.collect()


def load_embeddings(
    mode: str,
    view: str,
    embed_model: str = DEFAULT_EMBED_MODEL,
    cache_root: str = CACHE_ROOT,
    corpus_root: str = CORPUS_ROOT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached embeddings for the given mode/view.

    Parameters
    ----------
    mode : str
        ``"green"`` or ``"digital"``.
    view : str
        ``"title"``, ``"abstract"``, or ``"combined"``.
    embed_model : str
        SentenceTransformer model id.
    cache_root : str
        Directory where embedding caches live.
    corpus_root : str
        Root corpus cache (used for row-count validation).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(embeddings, appln_ids)`` -- embeddings are memory-mapped.

    Raises
    ------
    FileNotFoundError
        If the cache does not exist or is invalid.
    """
    paths = _embedding_cache_paths(mode, view, embed_model, cache_root)

    if not (os.path.exists(paths["meta"]) and os.path.exists(paths["ids"]) and os.path.exists(paths["emb"])):
        raise FileNotFoundError(
            f"No cached embeddings for mode={mode} view={view} model={embed_model}. "
            f"Run `python -m p2_taxonomy.main ingest` first."
        )

    meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))

    # Validate against corpus parts
    parts_dir = os.path.join(corpus_root, mode)
    parts = _list_part_files(parts_dir)
    n_rows = _fast_row_count(parts)

    if meta.get("rows") != n_rows:
        raise FileNotFoundError(
            f"Embedding cache row count ({meta.get('rows')}) != corpus rows ({n_rows}). "
            f"Re-run `python -m p2_taxonomy.main ingest`."
        )

    ids = np.load(paths["ids"], allow_pickle=False)
    emb = np.load(paths["emb"], mmap_mode="r")

    if len(ids) != n_rows or emb.shape[0] != n_rows:
        raise FileNotFoundError(
            f"Embedding cache shape mismatch. Re-run `python -m p2_taxonomy.main ingest`."
        )

    print(f"[embed-cache] Loaded view={view}: {paths['emb']}")
    return emb, ids
