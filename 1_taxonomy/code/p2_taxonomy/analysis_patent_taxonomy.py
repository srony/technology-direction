"""
Clean, scalable patent -> taxonomy-page tagging (millions of patents)
LONG output only (no wide option).

CLI:
  --taxonomy_csv PATH        (mutually exclusive with --taxonomy_xlsx)
  --pages_csv PATH           (required with --taxonomy_csv)
  --taxonomy_xlsx PATH       (mutually exclusive with --taxonomy_csv)
  --filtered_csv PATH        (required with --taxonomy_xlsx; provides descriptions)
  --patents_dir PATH
  --output_dir PATH
  --threshold FLOAT (default 0.5)
  --aggregate (optional, legacy)

Outputs (written as Parquet datasets for scale):
  output_dir/
    taxonomy_used.csv               <-- the taxonomy that was matched against
    faiss_pages_ivfpq.index
    checkpoint.json
    matches_long/   (parquet chunks)
    patent_technologies.parquet     <-- per-patent technology profile (always written)
    page_aggregation.csv            (if --aggregate)
    taxonomy_L2_aggregation.csv     (if --aggregate)
    taxonomy_cpc_aggregation.csv    (if --aggregate) <-- L2 x CPC Balassa (RCA)

Requirements:
  pip install pandas numpy tqdm sentence-transformers faiss-cpu pyarrow openpyxl
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from shared.utils import ensure_dir as _ensure_dir
from shared.patent_cache import (
    device_auto,
    extract_cpc_codes_before_space,
)


# ============================================================
# CONFIG (fixed for reproducibility; change here if needed)
# ============================================================

EMBED_MODEL = "AI-Growth-Lab/PatentSBERTa"

# Which cached patent-embedding view to use ("title", "abstract", or "combined").
# "abstract" matches what coverage analysis uses and gives a richer signal.
PATENT_EMB_VIEW = "abstract"

# Input column names (your files)
TAX_PAGE_COL = "page"
PAGE_TITLE_COL = "page_title"
PAGE_DESC_COL = "description"

PAT_ID_COL = "appln_id"
PAT_TITLE_COL = "appln_title"

# Your CPC lives here
PAT_CPC_DETAILS_COL = "cpc_coarse_list"
OUT_CPC_DETAILS_COL = "cpc_coarse_list"   # written to outputs

# Streaming + embedding
PATENT_BATCH_ROWS = 200_000
PATENT_EMBED_BATCH = 64
PAGE_EMBED_BATCH = 64

# Matching
K_CAND = 500


# ============================================================
# Utilities
# ============================================================

ensure_dir = _ensure_dir


def normalize_key(x: object) -> str:
    return "" if x is None else str(x).strip()


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()




def dedup_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ============================================================
# Load taxonomy + pages, build eligible pages
# ============================================================

def load_taxonomy_mapping(taxonomy_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(taxonomy_csv, low_memory=False)
    required = {"root", "L1", "L2", TAX_PAGE_COL, "root_given"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Taxonomy missing columns: {missing}")

    df = df.copy()
    df["page_key"] = df[TAX_PAGE_COL].map(normalize_key)
    df = df[df["page_key"] != ""]
    df = df.drop_duplicates(subset=["page_key"], keep="first")
    return df[["page_key", "root", "L1", "L2", "root_given", TAX_PAGE_COL]]


def load_pages(pages_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pages_csv, low_memory=False)
    required = {PAGE_TITLE_COL, PAGE_DESC_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pages missing columns: {missing}")

    df = df.copy()
    df["page_key"] = df[PAGE_TITLE_COL].map(normalize_key)
    df[PAGE_DESC_COL] = df[PAGE_DESC_COL].astype("string").fillna("").astype(str)
    df = df[df["page_key"] != ""]
    return df


def build_eligible_pages(
    taxonomy_df: pd.DataFrame,
    pages_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge pages with taxonomy. Cleaning (noise, dedup, empty desc) is
    expected to have been done in the filter-matches step."""
    kept = pages_df.merge(
        taxonomy_df,
        on="page_key",
        how="inner",
        validate="many_to_one",
    ).copy()

    tax_keys = set(taxonomy_df["page_key"].tolist())
    page_keys = set(pages_df["page_key"].tolist())
    missing_in_pages = sorted(tax_keys - page_keys)
    dropped_missing = pd.DataFrame({"page_key": missing_in_pages})
    if not dropped_missing.empty:
        dropped_missing = dropped_missing.merge(taxonomy_df, on="page_key", how="left")
        dropped_missing["drop_reason"] = "missing_in_pages"

    return kept, dropped_missing


def expand_xlsx_to_taxonomy_dataframe(
    xlsx_path: Path,
    filtered_csv: Path,
    sheet: str = "taxonomy",
) -> pd.DataFrame:
    """Convert a v3-style taxonomy xlsx + filtered descriptions CSV into a
    long DataFrame that can serve as both the taxonomy and pages source.

    The xlsx has columns ``root, L1, L2, pages`` where ``pages`` is a
    semicolon-joined string of Wikipedia page titles. This function
    explodes that into one row per (root, L1, L2, page) and joins
    descriptions from the filtered CSV (column ``page_title, description``).
    Pages without a description fall back to ``page_title`` with
    underscores converted to spaces.

    Returns
    -------
    pd.DataFrame
        Columns: ``[root, L1, L2, page, root_given, page_title, description]``.
    """
    xlsx_path = Path(xlsx_path)
    filtered_csv = Path(filtered_csv)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Taxonomy xlsx not found: {xlsx_path}")
    if not filtered_csv.exists():
        raise FileNotFoundError(f"Filtered descriptions CSV not found: {filtered_csv}")

    raw = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
    required_cols = {"root", "L1", "L2", "pages"}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(f"xlsx sheet {sheet!r} missing columns: {missing}")

    rows = []
    for _, r in raw.iterrows():
        pages_str = r["pages"] if isinstance(r["pages"], str) else ""
        for p in pages_str.split(";"):
            p = p.strip()
            if not p:
                continue
            rows.append({
                "root": r["root"],
                "L1": r["L1"],
                "L2": r["L2"],
                "page": p,
            })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError(f"No pages found after exploding xlsx {xlsx_path.name}")

    # de-duplicate (page must be unique for the many_to_one merge later)
    n_before = len(long_df)
    long_df = long_df.drop_duplicates(subset=["page"], keep="first").reset_index(drop=True)
    n_dup = n_before - len(long_df)
    if n_dup > 0:
        print(f"[xlsx] dropped {n_dup} duplicate page rows (kept first occurrence)")

    long_df["root_given"] = long_df["root"]
    long_df["page_title"] = long_df["page"]

    # Join descriptions
    desc_df = pd.read_csv(filtered_csv, low_memory=False, usecols=["page_title", "description"])
    desc_df["description"] = desc_df["description"].astype("string").fillna("").astype(str)
    desc_df = desc_df.drop_duplicates(subset=["page_title"], keep="first")

    merged = long_df.merge(desc_df, on="page_title", how="left")

    n_missing = int(merged["description"].isna().sum() + (merged["description"] == "").sum())
    # Fallback: empty description -> use page_title (underscores -> spaces)
    fallback_mask = merged["description"].isna() | (merged["description"].astype(str).str.strip() == "")
    n_fallback = int(fallback_mask.sum())
    merged.loc[fallback_mask, "description"] = (
        merged.loc[fallback_mask, "page_title"].astype(str).str.replace("_", " ", regex=False)
    )
    if n_fallback > 0:
        print(f"[xlsx] {n_fallback} pages had no description; filled with page title fallback")

    # Final column order
    out = merged[["root", "L1", "L2", "page", "root_given", "page_title", "description"]].copy()
    out["page_key"] = out["page"].map(normalize_key)
    out = out[out["page_key"] != ""]
    print(f"[xlsx] expanded {n_before} pages -> {len(out)} unique tagged pages "
          f"({len(set(zip(out['root'], out['L1'], out['L2'])))} clusters)")
    return out


# ============================================================
# Page embeddings cache
# ============================================================

def page_cache_paths(cache_dir: Path, pages_csv: Path, taxonomy_csv: Path) -> Dict[str, Path]:
    ensure_dir(cache_dir)
    payload = {
        "pages_md5": file_md5(pages_csv),
        "taxonomy_md5": file_md5(taxonomy_csv),
        "embed_model": EMBED_MODEL,
        "desc_col": PAGE_DESC_COL,
        "title_col": PAGE_TITLE_COL,
    }
    cid = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    base = cache_dir / f"pages_{cid}"
    return {
        "meta": base.with_suffix(".meta.json"),
        "npz": base.with_suffix(".embeddings.npz"),
        "titles": base.with_suffix(".titles.npy"),
    }


def load_or_build_page_embeddings(
    eligible_pages_df: pd.DataFrame,
    cache_paths: Dict[str, Path],
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    meta_path, npz_path, titles_path = cache_paths["meta"], cache_paths["npz"], cache_paths["titles"]

    if meta_path.exists() and npz_path.exists() and titles_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ok = (
                meta.get("embed_model") == EMBED_MODEL
                and bool(meta.get("normalized", False))
                and int(meta.get("n_pages", -1)) == len(eligible_pages_df)
            )
            if ok:
                emb = np.load(npz_path)["emb"].astype(np.float32, copy=False)
                titles = np.load(titles_path, allow_pickle=False)
                if emb.shape[0] == len(titles) == len(eligible_pages_df):
                    print(f"[pages-cache] HIT: {npz_path}")
                    return emb, titles
        except Exception:
            pass

    print("[pages-cache] MISS. Embedding page descriptions...")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    texts = eligible_pages_df[PAGE_DESC_COL].astype(str).fillna("").tolist()
    emb = model.encode(
        texts,
        batch_size=PAGE_EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    titles = eligible_pages_df[PAGE_TITLE_COL].astype("string").fillna("").astype(str).to_numpy(dtype=np.str_)
    np.savez_compressed(npz_path, emb=emb)
    np.save(titles_path, titles, allow_pickle=False)

    meta = {
        "embed_model": EMBED_MODEL,
        "n_pages": int(len(eligible_pages_df)),
        "dim": int(emb.shape[1]),
        "normalized": True,
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[pages-cache] Saved: {npz_path}")
    return emb, titles


# ============================================================
# FAISS IVF-PQ (pages)
# ============================================================

def build_or_load_faiss_index(page_emb: np.ndarray, index_path: Path) -> "faiss.Index":
    """Build or load an exact FlatIP page index.

    The page set here is small (~7-10k pages), so FlatIP gives exact recall
    in microseconds — no need for IVF-PQ approximation. This matches what
    coverage analysis (`analysis_cluster_quality.build_wiki_faiss_index`)
    already does. Stored at `index_path` for reuse.
    """
    import faiss

    ensure_dir(index_path.parent)
    n = int(page_emb.shape[0])
    dim = int(page_emb.shape[1])

    if index_path.exists():
        idx = faiss.read_index(str(index_path))
        if idx.ntotal == n:
            print(f"[faiss] Loaded FlatIP page index: n={idx.ntotal:,}")
            return idx
        print(f"[faiss] Existing index n={idx.ntotal} != current n={n}; rebuilding")

    print(f"[faiss] Build FlatIP pages: n={n:,} dim={dim}")
    idx = faiss.IndexFlatIP(dim)
    idx.add(np.ascontiguousarray(page_emb, dtype=np.float32))
    faiss.write_index(idx, str(index_path))
    print(f"[faiss] Saved: {index_path}")
    return idx


# ============================================================
# Checkpointing
# ============================================================

def load_checkpoint(cp_path: Path) -> Dict:
    if cp_path.exists():
        try:
            return json.loads(cp_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"batch_idx": 0, "rows_long": 0, "updated_utc": None}


def save_checkpoint(cp_path: Path, state: Dict) -> None:
    st = dict(state)
    st["updated_utc"] = datetime.utcnow().isoformat() + "Z"
    cp_path.write_text(json.dumps(st, indent=2), encoding="utf-8")


# ============================================================
# Streaming patents + writing parquet chunks
# ============================================================

def iter_patent_batches(patents_dir: Path, batch_rows: int):
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(patents_dir), format="parquet")
    cols = [PAT_ID_COL, PAT_TITLE_COL, PAT_CPC_DETAILS_COL]
    for i, rb in enumerate(dataset.to_batches(columns=cols, batch_size=batch_rows)):
        yield i, rb


def append_parquet_chunk(df: pd.DataFrame, out_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ensure_dir(out_dir)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    fname = out_dir / f"chunk_{ts}_{hashlib.sha1(os.urandom(16)).hexdigest()[:8]}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), fname, compression="zstd")


# ============================================================
# Main pipeline
# ============================================================

def run(
    taxonomy_csv: Optional[Path] = None,
    pages_csv: Optional[Path] = None,
    patents_dir: Path = None,
    output_dir: Path = None,
    threshold: float = 0.5,
    aggregate: bool = False,
    taxonomy_df: Optional[pd.DataFrame] = None,
) -> None:
    """Run patent-vs-page tagging.

    Two input modes:
    1. Pass `taxonomy_csv` + `pages_csv` (legacy): load from CSV files.
    2. Pass `taxonomy_df` (xlsx flow): a pre-built DataFrame with columns
       `[root, L1, L2, page, root_given, page_title, description, page_key]`.
    """
    from sentence_transformers import SentenceTransformer

    if patents_dir is None or output_dir is None:
        raise ValueError("patents_dir and output_dir are required")

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # 1) Build eligible pages from either input mode
    if taxonomy_df is not None:
        eligible_pages = taxonomy_df.copy()
        if "page_key" not in eligible_pages.columns:
            eligible_pages["page_key"] = eligible_pages["page"].map(normalize_key)
            eligible_pages = eligible_pages[eligible_pages["page_key"] != ""]
        print(f"[pages] eligible={len(eligible_pages):,} (from taxonomy_df)")
    else:
        if taxonomy_csv is None or pages_csv is None:
            raise ValueError("Either `taxonomy_df` or both `taxonomy_csv` and `pages_csv` must be provided")
        tax_df = load_taxonomy_mapping(taxonomy_csv)
        pages_df = load_pages(pages_csv)
        eligible_pages, _dropped_pages = build_eligible_pages(tax_df, pages_df)
        print(f"[pages] eligible={len(eligible_pages):,} dropped={len(_dropped_pages):,}")

    # Save the taxonomy that was actually used (audit trail)
    cols_to_save = ["root", "L1", "L2", "page", "root_given", "page_title", "description"]
    cols_present = [c for c in cols_to_save if c in eligible_pages.columns]
    eligible_pages[cols_present].to_csv(output_dir / "taxonomy_used.csv", index=False)

    # 2) Page embeddings + FAISS
    dev = device_auto()
    print(f"[embed] device={dev} model={EMBED_MODEL}")

    cache_dir = output_dir / "_cache"
    ensure_dir(cache_dir)
    if taxonomy_df is not None:
        # Hash the taxonomy DataFrame contents for stable caching
        df_hash = hashlib.sha256(
            pd.util.hash_pandas_object(eligible_pages[cols_present], index=False).values.tobytes()
        ).hexdigest()[:16]
        base = cache_dir / f"pages_{df_hash}"
        cache_paths = {
            "meta": base.with_suffix(".meta.json"),
            "npz": base.with_suffix(".embeddings.npz"),
            "titles": base.with_suffix(".titles.npy"),
        }
    else:
        cache_paths = page_cache_paths(cache_dir, pages_csv, taxonomy_csv)
    page_emb, page_titles = load_or_build_page_embeddings(eligible_pages, cache_paths, dev)

    index_path = output_dir / "faiss_pages_ivfpq.index"
    index = build_or_load_faiss_index(page_emb, index_path)

    # Pre-aligned taxonomy values for fast writing
    tax_cols = ["root", "L1", "L2", "root_given"]
    page_tax_arr = eligible_pages[tax_cols].astype("string").fillna("").astype(str).to_numpy()

    # 3) Load cached patent embeddings (no re-embedding!)
    from shared.patent_cache import load_embeddings as _load_patent_embeddings, CORPUS_ROOT

    patents_dir = Path(patents_dir)
    mode = patents_dir.name  # last directory ("digital" or "green")
    corpus_root = str(patents_dir.parent)
    print(f"[embed-cache] loading patent embeddings: mode={mode} view={PATENT_EMB_VIEW}")
    patent_emb, patent_ids = _load_patent_embeddings(
        mode=mode,
        view=PATENT_EMB_VIEW,
        embed_model=EMBED_MODEL,
        corpus_root=corpus_root,
    )
    n_patents = len(patent_ids)
    print(f"[embed-cache] loaded {n_patents:,} patent embeddings (memmap)")

    # 3b) Load corpus metadata (title + cpc) once, indexed by appln_id
    print("[corpus] loading metadata (title + cpc) for output rows...")
    parts = sorted(patents_dir.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No patent corpus parts under {patents_dir}")
    meta_pieces: List[pd.DataFrame] = []
    for p in tqdm(parts, desc="[corpus] read parts", unit="part"):
        meta_pieces.append(
            pd.read_parquet(p, columns=[PAT_ID_COL, PAT_TITLE_COL, PAT_CPC_DETAILS_COL])
        )
    corpus_meta = pd.concat(meta_pieces, ignore_index=True)
    corpus_meta = corpus_meta.drop_duplicates(subset=[PAT_ID_COL], keep="first")
    corpus_meta[PAT_ID_COL] = corpus_meta[PAT_ID_COL].astype(np.int64)
    corpus_meta = corpus_meta.set_index(PAT_ID_COL)
    print(f"[corpus] indexed {len(corpus_meta):,} patents by appln_id")
    del meta_pieces
    gc.collect()

    # 4) Outputs (parquet dataset)
    long_dir = output_dir / "matches_long"
    ensure_dir(long_dir)

    # 5) Resume
    checkpoint_path = output_dir / "checkpoint.json"
    cp = load_checkpoint(checkpoint_path)
    start_batch = int(cp.get("batch_idx", 0))
    rows_long = int(cp.get("rows_long", 0))
    print(f"[checkpoint] start_batch={start_batch} rows_long={rows_long}")

    # 6) FAISS search over cached embeddings, chunked
    chunk_size = PATENT_BATCH_ROWS
    n_batches = (n_patents + chunk_size - 1) // chunk_size
    k = min(K_CAND, page_emb.shape[0])

    for batch_idx in tqdm(range(n_batches), desc="[patents] FAISS chunks", unit="chunk"):
        if batch_idx < start_batch:
            continue
        s = batch_idx * chunk_size
        e = min(s + chunk_size, n_patents)
        chunk_ids = np.asarray(patent_ids[s:e]).astype(np.int64, copy=False)
        chunk_emb = np.ascontiguousarray(patent_emb[s:e], dtype=np.float32)

        D, I = index.search(chunk_emb, k)

        # Look up metadata for this chunk in one vectorized pass
        try:
            chunk_meta = corpus_meta.reindex(chunk_ids)
        except Exception:
            # Fallback: row-by-row lookup
            chunk_meta = pd.DataFrame(
                {PAT_TITLE_COL: ["" for _ in chunk_ids],
                 PAT_CPC_DETAILS_COL: ["" for _ in chunk_ids]},
                index=chunk_ids,
            )
        titles = chunk_meta[PAT_TITLE_COL].astype("string").fillna("").astype(str).tolist()
        cpc_details_list = chunk_meta[PAT_CPC_DETAILS_COL].astype("string").fillna("").astype(str).tolist()

        long_rows: List[Dict] = []
        for i in range(len(chunk_ids)):
            pid = int(chunk_ids[i])
            ptitle = titles[i]
            pcpc_details = cpc_details_list[i]

            cand_idx = I[i]
            cand_sim = D[i]
            valid = cand_idx >= 0
            cand_idx = cand_idx[valid]
            cand_sim = cand_sim[valid]

            keep = cand_sim >= threshold
            kept_idx = cand_idx[keep]
            kept_sim = cand_sim[keep]

            if kept_sim.size > 1:
                order = np.argsort(-kept_sim)
                kept_idx = kept_idx[order]
                kept_sim = kept_sim[order]

            n_matched = int(kept_sim.size)
            if n_matched > 0:
                for j in range(n_matched):
                    pi = int(kept_idx[j])
                    tax_vals = page_tax_arr[pi]
                    long_rows.append({
                        PAT_ID_COL: pid,
                        PAT_TITLE_COL: ptitle,
                        OUT_CPC_DETAILS_COL: pcpc_details,
                        PAGE_TITLE_COL: str(page_titles[pi]),
                        "similarity": float(kept_sim[j]),
                        "root": tax_vals[0],
                        "L1": tax_vals[1],
                        "L2": tax_vals[2],
                        "root_given": tax_vals[3],
                    })

        if long_rows:
            df_long = pd.DataFrame(long_rows)
            append_parquet_chunk(df_long, long_dir)
            rows_long += len(df_long)

        save_checkpoint(checkpoint_path, {
            "batch_idx": batch_idx + 1,
            "rows_long": rows_long,
        })

        del chunk_ids, chunk_emb, D, I, long_rows, titles, cpc_details_list
        gc.collect()

    print("[done] tagging complete")

    # Always produce per-patent technology profile
    aggregate_patent_technologies(
        matches_long_dir=long_dir,
        patents_dir=Path(patents_dir),
        out_path=output_dir / "patent_technologies.parquet",
        threshold=threshold,
    )
    print("[done] patent_technologies.parquet written")

    if aggregate:
        aggregate_from_long(long_dir, output_dir)
        print("[done] legacy aggregations complete")


# ============================================================
# Per-patent technology aggregation
# ============================================================

# Patent corpus columns to carry through to patent_technologies.parquet.
PATENT_PASSTHROUGH_COLS = [
    "appln_id",
    "docdb_family_id",
    "appln_title",
    "appln_abstract",
    "appln_filing_year",
    "country",
    "green",
    "digital",
    "forward_citation_global",
    "forward_citation_internal",
]


def aggregate_patent_technologies(
    matches_long_dir: Path,
    patents_dir: Path,
    out_path: Path,
    threshold: float,
) -> None:
    """Aggregate matches_long parquet chunks into a per-patent profile.

    For each patent (appln_id) with at least one match above ``threshold``,
    writes a row to ``out_path`` (parquet, zstd) with patent metadata and a
    nested ``technology`` column listing per-L2 stats:
    ``[{root, L1, L2, n_pages, avg_similarity, min_similarity, max_similarity}, ...]``.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    matches_long_dir = Path(matches_long_dir)
    patents_dir = Path(patents_dir)
    out_path = Path(out_path)

    if not any(matches_long_dir.glob("chunk_*.parquet")):
        print(f"[per-patent] no matches under {matches_long_dir} — skipping")
        return

    print(f"[per-patent] aggregating matches_long with DuckDB (threshold={threshold})...")
    import duckdb

    # Each (patent, page) match is unique across the dataset (each patent is
    # processed in exactly one FAISS chunk), so COUNT(*) per (patent, L2) ==
    # nunique(page_title). DuckDB scans the parquet dataset directly --
    # streaming, vectorised, no Python-level per-row loop.
    matches_glob = str(matches_long_dir / "chunk_*.parquet").replace("\\", "/")
    con = duckdb.connect(":memory:")
    agg = con.execute(
        f"""
        SELECT
            CAST(appln_id AS BIGINT) AS appln_id,
            root,
            L1,
            L2,
            COUNT(*)             AS n_pages,
            AVG(similarity)      AS avg_similarity,
            MIN(similarity)      AS min_similarity,
            MAX(similarity)      AS max_similarity
        FROM read_parquet('{matches_glob}')
        WHERE similarity >= {float(threshold)}
        GROUP BY appln_id, root, L1, L2
        ORDER BY appln_id, n_pages DESC, avg_similarity DESC
        """
    ).fetch_df()
    con.close()
    print(f"[per-patent] DuckDB returned {len(agg):,} (patent, L2) groups")

    if agg.empty:
        print("[per-patent] nothing to aggregate; skipping write")
        return

    # Build the per-patent technology list (one record per L2, ordered).
    # Convert rows to dicts once, then group the record objects — avoids the slow
    # per-group DataFrame-returning groupby.apply over millions of patents. `agg` is
    # pre-sorted (appln_id, n_pages DESC, avg_similarity DESC) and groupby preserves
    # within-group order, so the per-patent lists are identical to the apply version.
    tech_cols = ["root", "L1", "L2", "n_pages", "avg_similarity",
                 "min_similarity", "max_similarity"]
    _recs = pd.Series(agg[tech_cols].to_dict(orient="records"), index=agg.index)
    tagging_df = (
        _recs.groupby(agg["appln_id"], sort=True).agg(list)
        .reset_index(name="technology")
    )
    print(f"[per-patent] {len(tagging_df):,} patents with at least one match")

    # Read patent corpus columns we need; inner-join on appln_id.
    parts = sorted(patents_dir.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No patent corpus parts under {patents_dir}")

    tagged_ids = set(tagging_df["appln_id"].astype(np.int64).tolist())
    pieces: List[pd.DataFrame] = []
    for p in tqdm(parts, desc="[per-patent] join corpus", unit="part"):
        # Defensive: only request columns that exist in this part
        present = pq.ParquetFile(str(p)).schema_arrow.names
        cols_here = [c for c in PATENT_PASSTHROUGH_COLS if c in present]
        df_part = pd.read_parquet(p, columns=cols_here)
        df_part = df_part[df_part["appln_id"].isin(tagged_ids)]
        if not df_part.empty:
            pieces.append(df_part)

    if not pieces:
        raise RuntimeError(
            "[per-patent] no patents matched between matches_long and patents_dir; "
            "check that the corpus parts contain the appln_ids referenced in matches_long"
        )
    corpus_df = pd.concat(pieces, ignore_index=True)
    corpus_df = corpus_df.drop_duplicates(subset=["appln_id"], keep="first")

    merged = corpus_df.merge(tagging_df, on="appln_id", how="inner")
    n_lost = len(tagging_df) - len(merged)
    if n_lost > 0:
        print(f"[per-patent] WARNING: {n_lost} tagged patents had no corpus row (dropped)")

    # Build Arrow schema explicitly
    tech_struct = pa.struct([
        pa.field("root", pa.string()),
        pa.field("L1", pa.string()),
        pa.field("L2", pa.string()),
        pa.field("n_pages", pa.int32()),
        pa.field("avg_similarity", pa.float64()),
        pa.field("min_similarity", pa.float64()),
        pa.field("max_similarity", pa.float64()),
    ])

    schema_fields = []
    for col in PATENT_PASSTHROUGH_COLS:
        if col not in merged.columns:
            continue
        if col in {"appln_id", "docdb_family_id", "appln_filing_year", "green", "digital",
                   "forward_citation_global", "forward_citation_internal"}:
            schema_fields.append(pa.field(col, pa.int64()))
        else:
            schema_fields.append(pa.field(col, pa.string()))
    schema_fields.append(pa.field("technology", pa.list_(tech_struct)))
    schema = pa.schema(schema_fields)

    # Coerce dtypes for Arrow
    for col, field in zip([f.name for f in schema_fields], schema_fields):
        if col == "technology":
            continue
        if pa.types.is_integer(field.type):
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")
        else:
            merged[col] = merged[col].astype("string").fillna("").astype(str)

    # Sort by appln_id for cheap range scans
    merged = merged.sort_values("appln_id").reset_index(drop=True)

    table = pa.Table.from_pandas(merged, schema=schema, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd")
    print(f"[per-patent] wrote {len(merged):,} rows -> {out_path}")


# ============================================================
# Legacy aggregations (page / L2 / CPC) — kept behind --aggregate
# ============================================================

def aggregate_from_long(long_dir: Path, output_dir: Path) -> None:
    """
    Produces:
      - page_aggregation.csv
      - taxonomy_L2_aggregation.csv
      - taxonomy_cpc_aggregation.csv  (Balassa RCA for L2 × CPC)
    """
    import pyarrow.dataset as ds

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    dataset = ds.dataset(str(long_dir), format="parquet")

    # Page similarity aggregation
    page_sum: Dict[str, float] = {}
    page_cnt: Dict[str, int] = {}

    # Temp pairs dataset for unique patent counting + taxonomy + CPC details
    pairs_dir = output_dir / "_tmp_pairs_ds"
    ensure_dir(pairs_dir)

    for rb in tqdm(
        dataset.to_batches(
            columns=[PAGE_TITLE_COL, "similarity", "root", "L1", "L2", PAT_ID_COL, OUT_CPC_DETAILS_COL],
            batch_size=500_000
        ),
        desc="[agg] scanning matches_long",
        unit="batch",
    ):
        df = rb.to_pandas()
        df[PAGE_TITLE_COL] = df[PAGE_TITLE_COL].astype("string").fillna("").astype(str)
        df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.0)
        df["root"] = df["root"].astype("string").fillna("").astype(str)
        df["L1"] = df["L1"].astype("string").fillna("").astype(str)
        df["L2"] = df["L2"].astype("string").fillna("").astype(str)
        df[OUT_CPC_DETAILS_COL] = df[OUT_CPC_DETAILS_COL].astype("string").fillna("").astype(str)

        # Page similarity aggregation
        g = df.groupby(PAGE_TITLE_COL)["similarity"].agg(["sum", "count"])
        for page, r in g.iterrows():
            page_sum[page] = page_sum.get(page, 0.0) + float(r["sum"])
            page_cnt[page] = page_cnt.get(page, 0) + int(r["count"])

        # Pairs (unique patent per page per taxonomy) + CPC details for later taxonomy×CPC
        pairs = df[[PAGE_TITLE_COL, PAT_ID_COL, "root", "L1", "L2", OUT_CPC_DETAILS_COL]].drop_duplicates()
        append_parquet_chunk(pairs, pairs_dir)

        del df, g, pairs, rb
        gc.collect()

    # --- page_aggregation.csv ---
    pages = sorted(page_cnt.keys())
    page_agg = pd.DataFrame({
        PAGE_TITLE_COL: pages,
        "n_matches": [page_cnt[p] for p in pages],
        "avg_similarity": [page_sum[p] / max(1, page_cnt[p]) for p in pages],
    })

    # Read pairs (smaller than full long)
    pairs_all = pd.concat([pd.read_parquet(p) for p in pairs_dir.glob("chunk_*.parquet")], ignore_index=True)
    pairs_all[PAGE_TITLE_COL] = pairs_all[PAGE_TITLE_COL].astype("string").fillna("").astype(str)
    pairs_all[PAT_ID_COL] = pd.to_numeric(pairs_all[PAT_ID_COL], errors="coerce")
    pairs_all["root"] = pairs_all["root"].astype("string").fillna("").astype(str)
    pairs_all["L1"] = pairs_all["L1"].astype("string").fillna("").astype(str)
    pairs_all["L2"] = pairs_all["L2"].astype("string").fillna("").astype(str)
    pairs_all[OUT_CPC_DETAILS_COL] = pairs_all[OUT_CPC_DETAILS_COL].astype("string").fillna("").astype(str)

    pairs_all = pairs_all.dropna(subset=[PAT_ID_COL])
    pairs_all[PAT_ID_COL] = pairs_all[PAT_ID_COL].astype(np.int64)

    # Unique patent count per page
    pairs_page_pat = pairs_all.drop_duplicates(subset=[PAGE_TITLE_COL, PAT_ID_COL])
    page_unique = pairs_page_pat.groupby(PAGE_TITLE_COL)[PAT_ID_COL].nunique().rename("n_patents_unique").reset_index()
    page_tax = pairs_all.drop_duplicates(subset=[PAGE_TITLE_COL])[[PAGE_TITLE_COL, "root", "L1", "L2"]]

    page_agg = page_agg.merge(page_unique, on=PAGE_TITLE_COL, how="left").merge(page_tax, on=PAGE_TITLE_COL, how="left")
    page_agg["n_patents_unique"] = page_agg["n_patents_unique"].fillna(0).astype(int)
    page_agg.to_csv(output_dir / "page_aggregation.csv", index=False)

    # --- taxonomy_L2_aggregation.csv ---
    pairs_l2_pat = pairs_all.drop_duplicates(subset=["root", "L1", "L2", PAT_ID_COL])
    l2_unique = (
        pairs_l2_pat.groupby(["root", "L1", "L2"])[PAT_ID_COL]
        .nunique().rename("n_patents_unique").reset_index()
    )
    l2_pages = (
        pairs_all.drop_duplicates(subset=["root", "L1", "L2", PAGE_TITLE_COL])
        .groupby(["root", "L1", "L2"])[PAGE_TITLE_COL]
        .nunique().rename("n_pages_matched").reset_index()
    )

    # match-weighted mean by L2: scan long again (only 4 cols)
    l2_sum: Dict[Tuple[str, str, str], float] = {}
    l2_cnt: Dict[Tuple[str, str, str], int] = {}
    for rb in tqdm(
        dataset.to_batches(columns=["root", "L1", "L2", "similarity"], batch_size=500_000),
        desc="[agg] L2 match-weighted",
        unit="batch",
    ):
        df = rb.to_pandas()
        df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.0)
        df["root"] = df["root"].astype("string").fillna("").astype(str)
        df["L1"] = df["L1"].astype("string").fillna("").astype(str)
        df["L2"] = df["L2"].astype("string").fillna("").astype(str)

        g = df.groupby(["root", "L1", "L2"])["similarity"].agg(["sum", "count"])
        for idx, r in g.iterrows():
            l2_sum[idx] = l2_sum.get(idx, 0.0) + float(r["sum"])
            l2_cnt[idx] = l2_cnt.get(idx, 0) + int(r["count"])

        del df, g, rb
        gc.collect()

    l2_match_weighted = pd.DataFrame(
        [{
            "root": k[0], "L1": k[1], "L2": k[2],
            "n_matches_total": l2_cnt[k],
            "avg_similarity_match_weighted": l2_sum[k] / max(1, l2_cnt[k]),
        } for k in l2_cnt.keys()]
    )

    l2_page_balanced = (
        page_agg.groupby(["root", "L1", "L2"])["avg_similarity"]
        .mean().rename("avg_similarity_page_balanced").reset_index()
    )

    l2_agg = (
        l2_match_weighted
        .merge(l2_unique, on=["root", "L1", "L2"], how="left")
        .merge(l2_pages, on=["root", "L1", "L2"], how="left")
        .merge(l2_page_balanced, on=["root", "L1", "L2"], how="left")
    )
    l2_agg["n_patents_unique"] = l2_agg["n_patents_unique"].fillna(0).astype(int)
    l2_agg["n_pages_matched"] = l2_agg["n_pages_matched"].fillna(0).astype(int)
    l2_agg.to_csv(output_dir / "taxonomy_L2_aggregation.csv", index=False)

    # --- taxonomy_cpc_aggregation.csv (Balassa / RCA) ---
    base = pairs_all.drop_duplicates(subset=["root", "L1", "L2", PAT_ID_COL])[
        ["root", "L1", "L2", PAT_ID_COL, OUT_CPC_DETAILS_COL]
    ].copy()

    base["cpc_code"] = base[OUT_CPC_DETAILS_COL].apply(extract_cpc_codes_before_space).apply(dedup_preserve_order)
    base = base.explode("cpc_code")
    base["cpc_code"] = base["cpc_code"].astype("string").fillna("").astype(str).str.strip().str.upper()
    base = base[base["cpc_code"].str.len() > 0].copy()

    if base.empty:
        pd.DataFrame(columns=[
            "root", "L1", "L2", "cpc_code",
            "n_patents_l2_cpc",
            "n_patents_l2_total",
            "n_patents_cpc_total",
            "n_patents_global_total",
            "share_l2_cpc",
            "share_global_cpc",
            "balassa_rca"
        ]).to_csv(output_dir / "taxonomy_cpc_aggregation.csv", index=False)
    else:
        x_l2_cpc = (
            base.drop_duplicates(subset=["root", "L1", "L2", PAT_ID_COL, "cpc_code"])
            .groupby(["root", "L1", "L2", "cpc_code"])[PAT_ID_COL]
            .nunique()
            .rename("n_patents_l2_cpc")
            .reset_index()
        )

        x_l2_total = (
            pairs_all.drop_duplicates(subset=["root", "L1", "L2", PAT_ID_COL])
            .groupby(["root", "L1", "L2"])[PAT_ID_COL]
            .nunique()
            .rename("n_patents_l2_total")
            .reset_index()
        )

        x_cpc_total = (
            base.drop_duplicates(subset=[PAT_ID_COL, "cpc_code"])
            .groupby("cpc_code")[PAT_ID_COL]
            .nunique()
            .rename("n_patents_cpc_total")
            .reset_index()
        )

        n_patents_global_total = int(pairs_all[PAT_ID_COL].nunique())

        out = (
            x_l2_cpc
            .merge(x_l2_total, on=["root", "L1", "L2"], how="left")
            .merge(x_cpc_total, on="cpc_code", how="left")
        )

        out["n_patents_l2_total"] = out["n_patents_l2_total"].fillna(0).astype(int)
        out["n_patents_cpc_total"] = out["n_patents_cpc_total"].fillna(0).astype(int)
        out["n_patents_global_total"] = n_patents_global_total

        out["share_l2_cpc"] = out["n_patents_l2_cpc"] / out["n_patents_l2_total"].replace(0, np.nan)
        out["share_global_cpc"] = out["n_patents_cpc_total"] / (n_patents_global_total if n_patents_global_total > 0 else np.nan)
        out["balassa_rca"] = out["share_l2_cpc"] / out["share_global_cpc"].replace(0, np.nan)

        out = out.replace([np.inf, -np.inf], np.nan)
        out["balassa_rca"] = out["balassa_rca"].fillna(0.0)

        out = out.sort_values(["root", "L1", "L2", "balassa_rca"], ascending=[True, True, True, False])
        out.to_csv(output_dir / "taxonomy_cpc_aggregation.csv", index=False)

    # cleanup temp pairs
    try:
        for p in pairs_dir.glob("chunk_*.parquet"):
            p.unlink()
        pairs_dir.rmdir()
    except Exception:
        pass


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Patent-taxonomy tagging (long-only) + per-patent technology profile")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--taxonomy_csv", help="Long-format taxonomy CSV (legacy input)")
    grp.add_argument("--taxonomy_xlsx", help="v3-style taxonomy xlsx with [root, L1, L2, pages] columns")
    p.add_argument("--pages_csv", help="Pages CSV with [page_title, description] (required with --taxonomy_csv)")
    p.add_argument("--filtered_csv", help="Filtered CSV with [page_title, description] (required with --taxonomy_xlsx)")
    p.add_argument("--patents_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--aggregate", action="store_true",
                   help="Also write legacy page/L2/CPC aggregations (off by default)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.taxonomy_xlsx:
        if not args.filtered_csv:
            raise SystemExit("--filtered_csv is required when using --taxonomy_xlsx")
        tax_df = expand_xlsx_to_taxonomy_dataframe(
            Path(args.taxonomy_xlsx), Path(args.filtered_csv)
        )
        run(
            taxonomy_df=tax_df,
            patents_dir=Path(args.patents_dir),
            output_dir=Path(args.output_dir),
            threshold=float(args.threshold),
            aggregate=bool(args.aggregate),
        )
    else:
        if not args.pages_csv:
            raise SystemExit("--pages_csv is required when using --taxonomy_csv")
        run(
            taxonomy_csv=Path(args.taxonomy_csv),
            pages_csv=Path(args.pages_csv),
            patents_dir=Path(args.patents_dir),
            output_dir=Path(args.output_dir),
            threshold=float(args.threshold),
            aggregate=bool(args.aggregate),
        )


if __name__ == "__main__":
    main()