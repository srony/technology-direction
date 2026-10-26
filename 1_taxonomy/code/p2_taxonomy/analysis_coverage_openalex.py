"""
p2_taxonomy/analysis_coverage_openalex.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Patent-topic coverage diagnostics using the OpenAlex topic taxonomy.

Measures how well the 4,516 OpenAlex topics cover the patent corpus, with
breakdowns by domain (4) and field (26).  Runs two analyses in sequence —
one using topic **keywords**, one using topic **summary** — both compared
against patent **abstracts** via FAISS inner-product search.

Outputs per text source (keywords / summary):

1. Overall similarity distribution  — histogram + KDE + stats CSV
2. Domain-level breakdown           — per-domain stats, faceted histograms, grouped bar
3. Field-level breakdown            — per-field stats, heatmap, sorted bar
4. Threshold sensitivity            — coverage curves at 0.0–1.0, overall + per-domain
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from shared.patent_cache import (
    CACHE_ROOT,
    CORPUS_ROOT,
    DEFAULT_EMBED_MODEL,
    EMBED_BATCH,
    device_auto,
    load_embeddings,
)
from shared.utils import ensure_dir

# -- plotting setup (non-interactive backend) --------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


# -- Default path to the OpenAlex Excel file ---------------------------------
_DEFAULT_XLSX = Path(
    r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data"
    r"\taxonomy\open_alex\OpenAlex_topic_mapping_table.xlsx"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_openalex_taxonomy(xlsx_path: Path) -> pd.DataFrame:
    """Read the OpenAlex topic mapping Excel file and validate columns."""
    required = ["topic_id", "topic_name", "keywords", "summary",
                 "domain_id", "domain_name", "field_id", "field_name"]
    df = pd.read_excel(xlsx_path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"OpenAlex Excel missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
    print(f"[openalex] Loaded {len(df):,} topics from {xlsx_path}")
    print(f"           {df['domain_name'].nunique()} domains, "
          f"{df['field_name'].nunique()} fields")
    return df


def _prepare_topic_texts(df: pd.DataFrame, text_source: str) -> List[str]:
    """Prepare topic texts for embedding.

    For ``keywords``: replace semicolons with spaces.
    For ``summary``: use as-is.
    """
    if text_source == "keywords":
        return df["keywords"].fillna("").str.replace(";", " ", regex=False).tolist()
    elif text_source == "summary":
        return df["summary"].fillna("").astype(str).tolist()
    else:
        raise ValueError(f"Unknown text_source: {text_source}")


# ---------------------------------------------------------------------------
# Topic embedding with cache
# ---------------------------------------------------------------------------

def _encode_topics(
    texts: List[str],
    embed_model: str,
    device: str,
    cache_dir: Path,
    text_source: str,
) -> np.ndarray:
    """Encode topic texts with SentenceTransformer, caching to .npz.

    Cache key: ``openalex_{text_source}_{ModelShortName}.npz``
    """
    model_short = embed_model.split("/")[-1]
    cache_path = cache_dir / f"openalex_{text_source}_{model_short}.npz"

    if cache_path.exists():
        try:
            emb = np.load(cache_path)["emb"]
            if emb.shape[0] == len(texts):
                print(f"[openalex-emb] Cache HIT ({text_source}): {cache_path}")
                return emb.astype(np.float32, copy=False)
            print(f"[openalex-emb] Cache row mismatch ({emb.shape[0]} vs {len(texts)}), recomputing")
        except Exception as exc:
            print(f"[openalex-emb] Cache load failed ({exc}), recomputing")

    from sentence_transformers import SentenceTransformer

    print(f"[openalex-emb] Encoding {len(texts):,} topics ({text_source}) on {device} ...")
    model = SentenceTransformer(embed_model, device=device)
    emb = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    del model
    gc.collect()

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, emb=emb)
    print(f"[openalex-emb] Cached -> {cache_path}")
    return emb


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------

def _build_faiss_index(emb: np.ndarray):
    """Build a FAISS FlatIP (exact inner product) index."""
    import faiss

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(emb, dtype=np.float32))
    print(f"[faiss] Built FlatIP index  dim={dim}  n={index.ntotal:,}")
    return index


def _compute_max_similarities(
    patent_emb: np.ndarray,
    index,
    chunk_size: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Chunked k=1 FAISS search.

    Returns ``(max_sims[N], best_topic_idx[N])``.
    """
    n = patent_emb.shape[0]
    max_sims = np.empty(n, dtype=np.float32)
    best_idx = np.empty(n, dtype=np.int64)

    for start in tqdm(range(0, n, chunk_size), desc="[faiss] Searching"):
        end = min(n, start + chunk_size)
        chunk = np.ascontiguousarray(patent_emb[start:end], dtype=np.float32)
        D, I = index.search(chunk, 1)
        max_sims[start:end] = D[:, 0]
        best_idx[start:end] = I[:, 0]

    return max_sims, best_idx


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def _compute_stats(sims: np.ndarray) -> dict:
    """Compute summary statistics for a similarity array."""
    n = len(sims)
    if n == 0:
        return {"n": 0, "mean": 0, "median": 0, "std": 0,
                "pct_ge_0.3": 0, "pct_ge_0.4": 0, "pct_ge_0.5": 0, "pct_ge_0.6": 0}
    return {
        "n": int(n),
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "std": float(np.std(sims)),
        "pct_ge_0.3": float((sims >= 0.3).mean() * 100),
        "pct_ge_0.4": float((sims >= 0.4).mean() * 100),
        "pct_ge_0.5": float((sims >= 0.5).mean() * 100),
        "pct_ge_0.6": float((sims >= 0.6).mean() * 100),
    }


# ---------------------------------------------------------------------------
# Analysis 1: Overall similarity distribution
# ---------------------------------------------------------------------------

def _analysis_overall(
    max_sims: np.ndarray,
    text_source: str,
    out_dir: Path,
) -> dict:
    """Histogram + KDE of per-patent top-1 similarity, plus stats CSV."""
    stats = _compute_stats(max_sims)

    # -- Stats CSV --
    pd.DataFrame([stats]).to_csv(out_dir / "overall_stats.csv", index=False)

    # -- Plot --
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(max_sims, bins=100, range=(0, 1), density=True,
            alpha=0.5, color="steelblue", label="histogram")
    try:
        sns.kdeplot(max_sims, ax=ax, color="darkblue", linewidth=1.5, label="KDE")
    except Exception:
        pass

    for val, label, color, ls in [
        (stats["median"], f"median={stats['median']:.3f}", "red", "--"),
        (0.4, "0.4", "orange", ":"),
        (0.5, "0.5", "green", ":"),
        (0.6, "0.6", "purple", ":"),
    ]:
        ax.axvline(val, color=color, linestyle=ls, linewidth=1.2, label=label)

    stats_text = (
        f"N={stats['n']:,}\n"
        f"mean={stats['mean']:.4f}\n"
        f"std={stats['std']:.4f}\n"
        f">=0.4: {stats['pct_ge_0.4']:.1f}%\n"
        f">=0.5: {stats['pct_ge_0.5']:.1f}%\n"
        f">=0.6: {stats['pct_ge_0.6']:.1f}%"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Top-1 cosine similarity to nearest OpenAlex topic")
    ax.set_ylabel("Density")
    ax.set_title(f"Patent-OpenAlex Similarity Distribution  [{text_source}]")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "overall_similarity.png", dpi=200)
    plt.close(fig)

    return stats


# ---------------------------------------------------------------------------
# Analysis 2: By domain
# ---------------------------------------------------------------------------

def _analysis_by_domain(
    max_sims: np.ndarray,
    best_idx: np.ndarray,
    oa_df: pd.DataFrame,
    text_source: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Per-domain stats, faceted histograms, and grouped bar chart."""
    domains = oa_df["domain_name"].values
    patent_domains = np.array([domains[i] for i in best_idx])
    unique_domains = sorted(oa_df["domain_name"].unique())

    # -- Stats CSV --
    rows = []
    for dom in unique_domains:
        mask = patent_domains == dom
        stats = _compute_stats(max_sims[mask])
        stats["domain"] = dom
        rows.append(stats)
    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(out_dir / "domain_stats.csv", index=False)

    # -- Faceted histograms --
    n_dom = len(unique_domains)
    fig, axes = plt.subplots(1, n_dom, figsize=(5 * n_dom, 4), sharey=True)
    if n_dom == 1:
        axes = [axes]
    for ax, dom in zip(axes, unique_domains):
        sims_dom = max_sims[patent_domains == dom]
        ax.hist(sims_dom, bins=60, range=(0, 1), density=True,
                alpha=0.6, color="steelblue")
        try:
            sns.kdeplot(sims_dom, ax=ax, color="darkblue", linewidth=1.2)
        except Exception:
            pass
        ax.set_title(dom, fontsize=10)
        ax.set_xlabel("Similarity")
        med = float(np.median(sims_dom))
        ax.axvline(med, color="red", linestyle="--", linewidth=1, label=f"med={med:.3f}")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Density")
    fig.suptitle(f"Similarity by Domain  [{text_source}]", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "domain_histograms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -- Grouped bar chart (mean, median, pct_ge_0.5) --
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_dom)
    w = 0.25
    ax.bar(x - w, df_stats["mean"], w, label="Mean", color="#3498db")
    ax.bar(x, df_stats["median"], w, label="Median", color="#e67e22")
    ax.bar(x + w, df_stats["pct_ge_0.5"] / 100, w, label="Frac >= 0.5", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(unique_domains, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title(f"Domain Comparison  [{text_source}]")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "domain_barplot.png", dpi=200)
    plt.close(fig)

    return df_stats


# ---------------------------------------------------------------------------
# Analysis 3: By field
# ---------------------------------------------------------------------------

def _analysis_by_field(
    max_sims: np.ndarray,
    best_idx: np.ndarray,
    oa_df: pd.DataFrame,
    text_source: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Per-field stats, heatmap, and sorted bar chart."""
    fields = oa_df["field_name"].values
    domain_for_field = dict(zip(oa_df["field_name"], oa_df["domain_name"]))
    patent_fields = np.array([fields[i] for i in best_idx])
    unique_fields = sorted(oa_df["field_name"].unique())

    # -- Stats CSV --
    rows = []
    for fld in unique_fields:
        mask = patent_fields == fld
        stats = _compute_stats(max_sims[mask])
        stats["field"] = fld
        stats["domain"] = domain_for_field.get(fld, "?")
        rows.append(stats)
    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(out_dir / "field_stats.csv", index=False)

    # -- Heatmap (fields × metrics) --
    metrics = ["mean", "median", "pct_ge_0.3", "pct_ge_0.4", "pct_ge_0.5", "pct_ge_0.6"]
    hm = df_stats.set_index("field")[metrics].astype(float)
    hm = hm.sort_values("median", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(8, len(hm) * 0.35)))
    sns.heatmap(hm, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.6})
    ax.set_title(f"Field-Level Coverage Metrics  [{text_source}]")
    plt.tight_layout()
    fig.savefig(out_dir / "field_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -- Sorted bar chart (median similarity) --
    df_sorted = df_stats.sort_values("median", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(6, len(df_sorted) * 0.3)))
    colors = sns.color_palette("tab20", n_colors=len(df_sorted))
    ax.barh(df_sorted["field"], df_sorted["median"], color=colors)
    ax.set_xlabel("Median similarity")
    ax.set_title(f"Fields by Median Patent Similarity  [{text_source}]")
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(row["median"] + 0.005, i, f'{row["median"]:.3f}', va="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_dir / "field_barplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return df_stats


# ---------------------------------------------------------------------------
# Analysis 4: Threshold sensitivity
# ---------------------------------------------------------------------------

def _analysis_threshold_sensitivity(
    max_sims: np.ndarray,
    best_idx: np.ndarray,
    oa_df: pd.DataFrame,
    text_source: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Coverage curves at thresholds 0.0–1.0, overall + per-domain."""
    thresholds = np.arange(0.0, 1.01, 0.05)
    n = len(max_sims)

    domains = oa_df["domain_name"].values
    patent_domains = np.array([domains[i] for i in best_idx])
    unique_domains = sorted(oa_df["domain_name"].unique())

    rows = []
    for t in thresholds:
        t_val = float(t)
        covered = max_sims >= t_val
        row = {"threshold": t_val, "overall_pct": float(covered.sum() / n * 100)}
        for dom in unique_domains:
            dom_mask = patent_domains == dom
            n_dom = dom_mask.sum()
            if n_dom > 0:
                row[dom] = float((covered & dom_mask).sum() / n_dom * 100)
            else:
                row[dom] = 0.0
        rows.append(row)

    df_ts = pd.DataFrame(rows)
    df_ts.to_csv(out_dir / "threshold_sensitivity.csv", index=False)

    # -- Plot --
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_ts["threshold"], df_ts["overall_pct"],
            marker="o", markersize=4, linewidth=2, color="black", label="Overall")
    palette = sns.color_palette("Set2", n_colors=len(unique_domains))
    for dom, color in zip(unique_domains, palette):
        ax.plot(df_ts["threshold"], df_ts[dom],
                marker="s", markersize=3, linewidth=1.2, color=color,
                label=dom, alpha=0.8)

    # Annotate key overall thresholds
    for t_val in [0.3, 0.4, 0.5, 0.6]:
        pct = float((max_sims >= t_val).sum() / n * 100)
        ax.annotate(
            f"{pct:.1f}%",
            xy=(t_val, pct),
            xytext=(t_val + 0.02, pct + 3),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("% patents with >= 1 topic above threshold")
    ax.set_title(f"Threshold Sensitivity  [{text_source}]")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "threshold_sensitivity.png", dpi=200)
    plt.close(fig)

    return df_ts


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_openalex_coverage(
    mode: str,
    xlsx_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    cache_root: str = CACHE_ROOT,
    corpus_root: str = CORPUS_ROOT,
    chunk_size: int = 50_000,
) -> dict:
    """Run OpenAlex coverage analysis for both keywords and summary text sources.

    Parameters
    ----------
    mode : str
        ``"green"`` or ``"digital"`` — selects the patent subset.
    xlsx_path : Path or None
        Path to the OpenAlex topic mapping Excel file.
    output_dir : Path or None
        Root output directory.  Defaults to the directory containing *xlsx_path*.
    embed_model : str
        SentenceTransformer model id.
    cache_root / corpus_root : str
        Embedding and corpus cache paths.
    chunk_size : int
        FAISS search chunk size.

    Returns
    -------
    dict
        Summary with per-source results.
    """
    if xlsx_path is None:
        xlsx_path = _DEFAULT_XLSX
    xlsx_path = Path(xlsx_path)

    if output_dir is None:
        output_dir = xlsx_path.parent
    output_dir = Path(output_dir)

    cache_dir = output_dir / "embedding_cache"
    ensure_dir(cache_dir)

    print(f"\n{'='*60}")
    print(f"  OPENALEX COVERAGE ANALYSIS  mode={mode}")
    print(f"  xlsx       = {xlsx_path}")
    print(f"  output     = {output_dir}")
    print(f"  embed      = {embed_model}")
    print(f"{'='*60}\n")

    # --- Load OpenAlex taxonomy (once) ---
    oa_df = _load_openalex_taxonomy(xlsx_path)

    # --- Load patent embeddings (abstract view, once) ---
    print("\n[patents] Loading abstract embeddings ...")
    patent_emb, patent_ids = load_embeddings(
        mode=mode,
        view="abstract",
        embed_model=embed_model,
        cache_root=cache_root,
        corpus_root=corpus_root,
    )
    print(f"[patents] {patent_emb.shape[0]:,} patents loaded (mmap)")

    device = device_auto()
    summary_results: dict = {"mode": mode, "embed_model": embed_model, "sources": {}}

    text_sources = ["keywords", "summary"]

    for text_source in text_sources:
        print(f"\n{'─'*50}")
        print(f"  Text source: {text_source}")
        print(f"{'─'*50}")

        src_dir = output_dir / text_source
        ensure_dir(src_dir)

        # Encode topics
        texts = _prepare_topic_texts(oa_df, text_source)
        topic_emb = _encode_topics(texts, embed_model, device, cache_dir, text_source)

        # Build FAISS index
        topic_index = _build_faiss_index(topic_emb)

        # Search patents against topics
        max_sims, best_idx = _compute_max_similarities(
            patent_emb, topic_index, chunk_size=chunk_size,
        )

        # Verify: domain/field assignments sum to total
        n_total = len(max_sims)
        print(f"[check] {n_total:,} patents searched")

        # Run analyses
        print("[1/4] Overall similarity distribution ...")
        overall = _analysis_overall(max_sims, text_source, src_dir)

        print("[2/4] Domain breakdown ...")
        domain_df = _analysis_by_domain(max_sims, best_idx, oa_df, text_source, src_dir)

        print("[3/4] Field breakdown ...")
        field_df = _analysis_by_field(max_sims, best_idx, oa_df, text_source, src_dir)

        print("[4/4] Threshold sensitivity ...")
        thresh_df = _analysis_threshold_sensitivity(
            max_sims, best_idx, oa_df, text_source, src_dir,
        )

        summary_results["sources"][text_source] = {
            "overall": overall,
            "n_domains": len(domain_df),
            "n_fields": len(field_df),
        }

        # Free per-source arrays
        del topic_emb, topic_index, max_sims, best_idx
        gc.collect()

    # --- Write combined summary JSON ---
    summary_path = output_dir / "openalex_coverage_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\n[done] Summary -> {summary_path}")

    return summary_results
