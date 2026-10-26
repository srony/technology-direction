"""
p2_taxonomy/analysis_cluster_quality.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unified post-cluster validation module.

Consolidates three validation capabilities:

1. **Per-cluster margin analysis** — within-sim vs nearest-other-centroid-sim
   (original contents of this file).
2. **Page-distance analysis** — per-page semantic distances to cluster centroids
   (from ``analysis_page_distance.py``).
3. **Patent-page coverage diagnostics** — 7 analyses comparing patent embeddings
   against wiki page embeddings via FAISS (from ``analysis_coverage.py``).
4. **LLM-based cluster verification** — send page lists to an LLM for coherence
   rating and misplacement detection (from ``analysis_cluster_llm_verify.py``).

The ``compute_cluster_margins`` function is used by
``python -m p2_taxonomy.main validate-clusters`` (step 07) as part of the
unified post-cluster validation pipeline.

This module also has a standalone CLI for quick margin checks::

    python -m p2_taxonomy.analysis_cluster_quality --run-dir <path> --mode digital
    python -m p2_taxonomy.analysis_cluster_quality --run-dir <path> --mode digital --skip-coverage
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm

try:
    import seaborn as sns
except ImportError:
    sns = None

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p2_taxonomy.analysis_cluster import (
    CACHE_ROOT,
    clean_description_text,
    load_or_build_embeddings,
)
from shared.llm import read_openrouter_key
from p2_taxonomy.llm import _make_chat_fn as make_chat_fn

# Chat function type: (prompt, system_prompt, max_tokens, json_schema=None) -> response_text
ChatFn = Callable[[str, str, int, Optional[dict]], str]
from shared.patent_cache import (
    CACHE_ROOT as PATENT_CACHE_ROOT,
    CORPUS_ROOT,
    DEFAULT_EMBED_MODEL,
    EMBED_BATCH,
    _iter_parts,
    _list_part_files,
    device_auto,
    load_embeddings,
)
from shared.utils import ensure_dir

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT = Path(
    r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\taxonomy"
    r"\wiki\pipeline\04_filter\digital_technology_taxonomy_filtered.csv"
)

LEVEL_COLS = ["root", "L1", "L2"]


# ---------------------------------------------------------------------------
# Per-cluster margin computation
# ---------------------------------------------------------------------------

def _compute_margins_flat(
    emb: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Compute per-cluster margins for a flat set of clusters (no hierarchy)."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        rows = []
        for cid in unique_labels:
            idx = np.where(labels == cid)[0]
            vecs = emb[idx]
            centroid = vecs.mean(axis=0)
            nrm = np.linalg.norm(centroid)
            if nrm > 0:
                centroid /= nrm
            within_sim = float(np.mean(vecs @ centroid))
            rows.append({
                "cluster_id": cid,
                "n_pages": len(idx),
                "within_sim": within_sim,
                "nearest_other_sim": np.nan,
                "margin": np.nan,
                "nearest_cluster_id": "",
            })
        return pd.DataFrame(rows)

    # Build centroids
    centroids = {}
    for cid in unique_labels:
        idx = np.where(labels == cid)[0]
        vecs = emb[idx]
        centroid = vecs.mean(axis=0)
        nrm = np.linalg.norm(centroid)
        if nrm > 0:
            centroid /= nrm
        centroids[cid] = centroid

    # Centroid similarity matrix (K x K)
    cid_list = list(centroids.keys())
    centroid_mat = np.vstack([centroids[c] for c in cid_list])
    sim_mat = centroid_mat @ centroid_mat.T

    rows = []
    for i, cid in enumerate(cid_list):
        idx = np.where(labels == cid)[0]
        n_pages = len(idx)
        vecs = emb[idx]
        within_sim = float(np.mean(vecs @ centroids[cid]))

        sims_to_others = sim_mat[i].copy()
        sims_to_others[i] = -np.inf
        nearest_idx = int(np.argmax(sims_to_others))
        nearest_other_sim = float(sims_to_others[nearest_idx])
        nearest_cluster = cid_list[nearest_idx]

        margin = within_sim - nearest_other_sim

        rows.append({
            "cluster_id": cid,
            "n_pages": n_pages,
            "within_sim": round(within_sim, 6),
            "nearest_other_sim": round(nearest_other_sim, 6),
            "margin": round(margin, 6),
            "nearest_cluster_id": nearest_cluster,
        })

    return pd.DataFrame(rows)


def compute_cluster_margins(
    emb: np.ndarray,
    labels: np.ndarray,
    parent_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute per-cluster margin analysis (hierarchical: within-parent).

    When *parent_labels* is provided, margins are computed **within each
    parent group** — each cluster is only compared against its siblings
    (same parent), matching the runtime clustering metric.

    When *parent_labels* is None, falls back to global comparison (all
    clusters compared against each other).

    Parameters
    ----------
    emb : (N, D) normalized embeddings
    labels : (N,) string cluster labels
    parent_labels : (N,) string parent-level labels, or None for global

    Returns
    -------
    DataFrame with columns:
        cluster_id, parent_id, n_pages, within_sim, nearest_other_sim,
        margin, nearest_cluster_id
    """
    if parent_labels is None:
        df = _compute_margins_flat(emb, labels)
        df["parent_id"] = ""
        return df

    # Group by parent, compute margins within each parent group
    all_rows = []
    unique_parents = np.unique(parent_labels)

    for parent in unique_parents:
        parent_mask = parent_labels == parent
        sub_emb = emb[parent_mask]
        sub_labels = labels[parent_mask]

        sub_df = _compute_margins_flat(sub_emb, sub_labels)
        sub_df["parent_id"] = parent
        all_rows.append(sub_df)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Margin / CPC / citation plotting helpers
# ---------------------------------------------------------------------------

def _plot_margin_bar_chart(
    margins_df: pd.DataFrame,
    level: str,
    mode: str,
    output_dir: Path,
) -> None:
    """Horizontal bar chart of per-cluster margins, green/red by sign."""
    df = margins_df.sort_values("margin").copy()
    n = len(df)
    if n == 0:
        return

    # If too many clusters, show worst 30 + best 30
    if n > 60:
        df = pd.concat([df.head(30), df.tail(30)])

    labels = [str(c)[:40] for c in df["cluster_id"]]
    values = df["margin"].values
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    fig_h = max(5, 0.3 * len(df))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(range(len(df)), values, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)

    n_neg = int((margins_df["margin"] < 0).sum())
    title_extra = f"  ({n_neg}/{len(margins_df)} negative)"
    if n > 60:
        title_extra += f"  [showing worst 30 + best 30 of {len(margins_df)}]"
    ax.set_xlabel("Margin  (within_sim − nearest_other_sim)")
    ax.set_title(f"Cluster Margins  [{level}]  ({mode}){title_extra}")
    plt.tight_layout()
    out_path = output_dir / f"cluster_margin_bars_{mode}_{level}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] {out_path}", flush=True)


def _plot_within_vs_nearest_scatter(
    margins_df: pd.DataFrame,
    level: str,
    mode: str,
    output_dir: Path,
) -> None:
    """Scatter: within_sim (x) vs nearest_other_sim (y), size=n_pages."""
    df = margins_df.dropna(subset=["within_sim", "nearest_other_sim"])
    if len(df) == 0:
        return

    x = df["within_sim"].values
    y = df["nearest_other_sim"].values
    sizes = np.clip(df["n_pages"].values, 5, 500)
    sizes_norm = 20 + 200 * (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1)
    colors = ["#2ecc71" if m >= 0 else "#e74c3c" for m in df["margin"].values]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, s=sizes_norm, c=colors, alpha=0.6, edgecolors="grey", linewidth=0.3)

    # Diagonal: within = nearest (margin = 0)
    lim_lo = min(float(x.min()), float(y.min())) - 0.02
    lim_hi = max(float(x.max()), float(y.max())) + 0.02
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="black", linestyle="--",
            linewidth=1, label="margin = 0", alpha=0.5)

    # Label worst 3
    worst = df.nsmallest(3, "margin")
    for _, r in worst.iterrows():
        lbl = str(r["cluster_id"])[:30]
        ax.annotate(lbl, (r["within_sim"], r["nearest_other_sim"]),
                    fontsize=6, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points")

    n_neg = int((df["margin"] < 0).sum())
    stats_text = (
        f"N = {len(df)} clusters\n"
        f"above diag (negative margin): {n_neg}\n"
        f"size ∝ n_pages"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Within-cluster similarity (mean cos to own centroid)")
    ax.set_ylabel("Nearest-other centroid similarity")
    ax.set_title(f"Within vs Nearest-Other Similarity  [{level}]  ({mode})")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_path = output_dir / f"cluster_within_vs_nearest_{mode}_{level}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] {out_path}", flush=True)


def _plot_margin_vs_size(
    margins_df: pd.DataFrame,
    level: str,
    mode: str,
    output_dir: Path,
) -> None:
    """Scatter: cluster size (x, log) vs margin (y)."""
    df = margins_df.dropna(subset=["margin", "n_pages"])
    if len(df) == 0:
        return

    x = df["n_pages"].values.astype(float)
    y = df["margin"].values
    colors = ["#2ecc71" if m >= 0 else "#e74c3c" for m in y]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=30, c=colors, alpha=0.6, edgecolors="grey", linewidth=0.3)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Cluster size (n_pages, log scale)")
    ax.set_ylabel("Margin")
    ax.set_title(f"Margin vs Cluster Size  [{level}]  ({mode})")
    plt.tight_layout()
    out_path = output_dir / f"cluster_margin_vs_size_{mode}_{level}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] {out_path}", flush=True)








def _plot_validation_summary_dashboard(
    unified: dict,
    mode: str,
    output_dir: Path,
) -> None:
    """Consolidated validation dashboard: one heatmap of all metrics × levels.

    Rows = taxonomy levels (root, L1, L2).
    Columns = Coherence, CPC Ratio, Citation Ratio, Wiki Cat Ratio,
              Page Dist %, Expert Purity.
    Cells annotated with raw values, coloured by column-normalised score.
    """
    levels_data = unified.get("levels", {})
    levels = [l for l in ["root", "L1", "L2"] if l in levels_data]
    if not levels:
        return

    # Define metric specs: (display_name, extractor_func)
    metric_specs: list[tuple[str, callable]] = [
        ("Coherence", lambda d: d.get("coherence", {}).get("mean")),
        ("Margin\n% positive", lambda d: d.get("page_distance", {}).get("pct_positive_margin")),
        ("CPC Ratio\n(3-digit)", lambda d: d.get("cpc_3digit", d.get("cpc_2digit", {})).get("median_cpc_ratio")),
        ("Citation\nRatio", lambda d: d.get("citation", {}).get("median_citation_ratio")),
    ]

    # Build matrix
    raw_matrix = []
    for level in levels:
        ld = levels_data[level]
        row = [extractor(ld) for _, extractor in metric_specs]
        raw_matrix.append(row)

    raw = np.array(raw_matrix, dtype=float)
    metric_labels = [name for name, _ in metric_specs]

    # Drop columns that are all NaN
    valid_cols = ~np.all(np.isnan(raw), axis=0)
    raw = raw[:, valid_cols]
    metric_labels = [m for m, v in zip(metric_labels, valid_cols) if v]

    if raw.shape[1] == 0:
        return

    # Column-normalise for colour (0–1 per column)
    norm = np.full_like(raw, 0.5)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            vmin, vmax = np.nanmin(col), np.nanmax(col)
            if vmax > vmin:
                norm[:, j] = np.where(valid, (col - vmin) / (vmax - vmin), 0.5)

    fig_w = max(6, 1.4 * len(metric_labels))
    fig_h = max(3, 0.8 * len(levels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if sns is not None:
        # Build annotation strings
        annot = np.full_like(raw, "", dtype=object)
        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                v = raw[i, j]
                if np.isnan(v):
                    annot[i, j] = "—"
                elif v >= 100:
                    annot[i, j] = f"{v:.0f}"
                elif v >= 10:
                    annot[i, j] = f"{v:.1f}"
                else:
                    annot[i, j] = f"{v:.2f}"

        sns.heatmap(
            norm, ax=ax, annot=annot, fmt="",
            cmap="YlGn", vmin=0, vmax=1,
            linewidths=1, linecolor="white",
            xticklabels=metric_labels,
            yticklabels=levels,
            cbar=False,
            annot_kws={"fontsize": 12, "fontweight": "bold"},
        )
    else:
        ax.imshow(norm, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels(levels, fontsize=10)
        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                v = raw[i, j]
                txt = "—" if np.isnan(v) else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_title(f"Validation Summary — {mode}", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=10, rotation=0)
    ax.tick_params(axis="y", labelsize=11)
    plt.tight_layout()
    out_path = output_dir / f"validation_summary_dashboard_{mode}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out_path}", flush=True)


def plot_cluster_quality(
    margins_by_level: Dict[str, pd.DataFrame],
    mode: str,
    output_dir: Path,
    cpc_by_level: Optional[Dict[str, pd.DataFrame]] = None,
    citation_by_level: Optional[Dict[str, pd.DataFrame]] = None,
    wikicat_by_level: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Generate per-metric plots for each taxonomy level."""
    for level, mdf in margins_by_level.items():
        _plot_margin_bar_chart(mdf, level, mode, output_dir)
        _plot_within_vs_nearest_scatter(mdf, level, mode, output_dir)
        _plot_margin_vs_size(mdf, level, mode, output_dir)

    if citation_by_level:
        for level, cdf in citation_by_level.items():
            pass
# ---------------------------------------------------------------------------
# Patent → cluster mapping helper
# ---------------------------------------------------------------------------

def _build_patent_to_cluster(
    page_to_cluster: Dict[str, str],
    wiki_page_titles: List[str],
    max_sims: np.ndarray,
    best_wiki_idx: np.ndarray,
    patent_ids: np.ndarray,
    threshold: float = 0.5,
) -> Dict[int, str]:
    """Map above-threshold patents to their cluster label.

    For each patent with max_sim >= threshold, look up its best wiki page
    and then the cluster label for that page.

    Returns ``{appln_id: cluster_label}`` (only patents with a valid mapping).
    """
    above = max_sims >= threshold
    patent_to_cluster: Dict[int, str] = {}
    for i in np.where(above)[0]:
        wiki_idx = int(best_wiki_idx[i])
        if wiki_idx < 0 or wiki_idx >= len(wiki_page_titles):
            continue
        page_title = wiki_page_titles[wiki_idx]
        cluster = page_to_cluster.get(page_title)
        if cluster is not None:
            patent_to_cluster[int(patent_ids[i])] = cluster
    return patent_to_cluster


# ---------------------------------------------------------------------------
# CPC homogeneity per cluster
# ---------------------------------------------------------------------------



def compute_cluster_cpc_ratio(
    page_to_cluster: Dict[str, str],
    wiki_page_titles: List[str],
    max_sims: np.ndarray,
    best_wiki_idx: np.ndarray,
    patent_ids: np.ndarray,
    cpc_dict: Dict[int, List[str]],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute per-cluster CPC sharing ratio (base-rate-corrected).

    For each cluster k at a given CPC digit level:
    - For each CPC code c: n_c_k = patents in k with code c, n_c = total with c
    - intra_k = sum_c n_c_k * (n_c_k - 1)  — CPC co-occurrence pairs within cluster
    - inter_k = sum_c n_c_k * (n_c - n_c_k)  — CPC co-occurrence pairs across clusters
    - possible_intra = n_k * (n_k - 1), possible_inter = n_k * (N - n_k)
    - cpc_ratio = (intra_k / possible_intra) / (inter_k / possible_inter)

    Ratio > 1 means cluster patents share CPC codes more than expected by chance.
    """
    patent_to_cluster = _build_patent_to_cluster(
        page_to_cluster, wiki_page_titles, max_sims, best_wiki_idx,
        patent_ids, threshold,
    )

    # Group patents by cluster, collecting CPC codes per patent
    cluster_patents: Dict[str, List[int]] = defaultdict(list)
    for aid, cluster in patent_to_cluster.items():
        cluster_patents[cluster].append(aid)

    N = len(patent_to_cluster)

    # Global CPC code counts (across all mapped patents)
    global_code_counts: Counter = Counter()
    patent_codes: Dict[int, List[str]] = {}
    for aid in patent_to_cluster:
        codes = cpc_dict.get(aid, [])
        patent_codes[aid] = codes
        for c in codes:
            global_code_counts[c] += 1

    rows = []
    for cluster in sorted(cluster_patents.keys()):
        aids = cluster_patents[cluster]
        n_k = len(aids)

        # Count CPC codes within this cluster
        cluster_code_counts: Counter = Counter()
        for aid in aids:
            for c in patent_codes.get(aid, []):
                cluster_code_counts[c] += 1

        # Compute intra and inter
        intra = sum(n_c_k * (n_c_k - 1)
                    for c, n_c_k in cluster_code_counts.items())
        inter = sum(n_c_k * (global_code_counts[c] - n_c_k)
                    for c, n_c_k in cluster_code_counts.items())

        possible_intra = n_k * (n_k - 1)
        possible_inter = n_k * (N - n_k)

        if possible_intra > 0 and possible_inter > 0 and inter > 0:
            ratio = (intra / possible_intra) / (inter / possible_inter)
        else:
            ratio = float("nan")

        rows.append({
            "cluster_id": cluster,
            "n_patents": n_k,
            "cpc_sharing_ratio": round(ratio, 4) if not np.isnan(ratio) else ratio,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Balassa RCA: cluster × CPC cross-tabulation
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Citation coherence per cluster
# ---------------------------------------------------------------------------

def compute_cluster_citation_coherence(
    page_to_cluster: Dict[str, str],
    wiki_page_titles: List[str],
    max_sims: np.ndarray,
    best_wiki_idx: np.ndarray,
    patent_ids: np.ndarray,
    citation_graph: Dict[int, Set[int]],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute per-cluster citation coherence.

    For each cluster:
    - n_intra_edges: citation edges where both patents are in the same cluster
    - n_inter_edges: citation edges where patents are in different clusters
    - intra_rate: n_intra / (n_intra + n_inter)
    """
    patent_to_cluster = _build_patent_to_cluster(
        page_to_cluster, wiki_page_titles, max_sims, best_wiki_idx,
        patent_ids, threshold,
    )

    # Count edges per cluster
    cluster_intra: Dict[str, int] = Counter()
    cluster_inter: Dict[str, int] = Counter()
    cluster_n_patents: Dict[str, int] = Counter()

    # Count patents per cluster
    for cluster in patent_to_cluster.values():
        cluster_n_patents[cluster] += 1

    n_total_mapped = len(patent_to_cluster)

    # Scan citation edges (only count each directed edge once from the source side)
    mapped_aids = set(patent_to_cluster.keys())
    for aid in mapped_aids:
        cluster_a = patent_to_cluster[aid]
        neighbors = citation_graph.get(aid, set())
        for nbr in neighbors:
            cluster_b = patent_to_cluster.get(nbr)
            if cluster_b is None:
                continue  # neighbor not in any cluster
            if cluster_a == cluster_b:
                cluster_intra[cluster_a] += 1
            else:
                cluster_inter[cluster_a] += 1

    rows = []
    for cluster in sorted(cluster_n_patents.keys()):
        n_pat = cluster_n_patents[cluster]
        n_intra = cluster_intra.get(cluster, 0)
        n_inter = cluster_inter.get(cluster, 0)
        total_edges = n_intra + n_inter
        intra_rate = n_intra / total_edges if total_edges > 0 else 0.0

        # Citation ratio: (intra_citations / possible_intra_pairs)
        #                / (inter_citations / possible_inter_pairs)
        n_outside = n_total_mapped - n_pat
        possible_intra = n_pat * (n_pat - 1)  # directed pairs within cluster
        possible_inter = n_pat * n_outside     # directed pairs across clusters
        if possible_intra > 0 and possible_inter > 0 and n_inter > 0:
            intra_rate_pair = n_intra / possible_intra
            inter_rate_pair = n_inter / possible_inter
            citation_ratio = intra_rate_pair / inter_rate_pair
        else:
            citation_ratio = float("nan")

        rows.append({
            "cluster_id": cluster,
            "n_patents": n_pat,
            "n_intra_edges": n_intra,
            "n_inter_edges": n_inter,
            "intra_rate": round(intra_rate, 4),
            "citation_ratio": round(citation_ratio, 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Wiki category share per cluster
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# Semantic coherence plots
# ---------------------------------------------------------------------------

def _per_cluster_coherence(emb: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Compute per-cluster semantic coherence: mean cosine(member, centroid).

    Parameters
    ----------
    emb : (N, D) array, assumed L2-normalised
    labels : (N,) string or int array of cluster assignments

    Returns DataFrame with columns: cluster_id, n_pages, coherence
    """
    unique = np.unique(labels)
    rows = []
    for cid in unique:
        idx = np.where(labels == cid)[0]
        if len(idx) < 2:
            rows.append({"cluster_id": cid, "n_pages": len(idx), "coherence": 1.0})
            continue
        vecs = emb[idx]
        centroid = vecs.mean(axis=0)
        nrm = np.linalg.norm(centroid)
        if nrm > 0:
            centroid = centroid / nrm
        sims = vecs @ centroid
        rows.append({
            "cluster_id": cid,
            "n_pages": len(idx),
            "coherence": float(np.mean(sims)),
        })
    return pd.DataFrame(rows)












# ---------------------------------------------------------------------------
# Orchestrator: run CPC + citation cluster validation
# ---------------------------------------------------------------------------

def run_cluster_external_validation(
    run_dir: Path,
    pages_csv: Path,
    mode: str,
    output_dir: Optional[Path] = None,
    *,
    embed_model: str = "AI-Growth-Lab/PatentSBERTa",
    threshold: float = 0.5,
    citation_csv: Optional[Path] = None,
    level_cols: Optional[List[str]] = None,
    do_plot: bool = True,
    cpc_digits: str = "2,3,4",
    skip_wiki_category: bool = False,
    skip_modularity: bool = False,
    citation_graph: Optional[Dict[int, Set[int]]] = None,
) -> Tuple[Path, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame],
           Dict[str, pd.DataFrame]]:
    """Run CPC homogeneity + citation coherence + wiki categories.

    Loads patent embeddings, runs FAISS search (patent→wiki), then computes
    per-cluster CPC, citation ratio, and wiki category metrics for each
    taxonomy level.
    """
    if level_cols is None:
        level_cols = LEVEL_COLS

    run_dir = Path(run_dir)
    pages_csv = Path(pages_csv)
    if output_dir is None:
        output_dir = run_dir / "validation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if citation_csv is None:
        citation_csv = Path(CITATION_CSV)
    else:
        citation_csv = Path(citation_csv)

    print(f"\n{'='*60}")
    print(f"  EXTERNAL CLUSTER VALIDATION (CPC + Citations)")
    print(f"  run_dir    = {run_dir}")
    print(f"  pages_csv  = {pages_csv}")
    print(f"  mode       = {mode}")
    print(f"  output_dir = {output_dir}")
    print(f"{'='*60}\n")

    # ---- Load page assignments ----
    assign_files = list(run_dir.glob("page_assignments_L2*.csv"))
    if not assign_files:
        assign_files = list(run_dir.glob("page_assignments_*.csv"))
    if not assign_files:
        raise FileNotFoundError(f"No page_assignments CSV in {run_dir}")
    assign_csv = assign_files[0]

    df_assign = pd.read_csv(assign_csv)
    df_assign["page"] = df_assign["page"].astype(str)
    print(f"[ext] Assignments: {assign_csv.name} ({len(df_assign)} pages)")

    # Filter out peripheral pages
    for level in level_cols:
        if level in df_assign.columns:
            df_assign[level] = df_assign[level].astype(str)

    # ---- Load wiki embeddings + FAISS index ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wiki_emb, wiki_df = load_wiki_embeddings(pages_csv, embed_model, device)
    wiki_page_titles = wiki_df["page_title"].tolist()

    wiki_index_path = _WIKI_EMB_CACHE_DIR / "faiss_wiki.index"
    wiki_index = build_wiki_faiss_index(wiki_emb, save_path=wiki_index_path)

    # ---- Load CPC dicts (multiple digit levels) ----
    digit_levels = [int(d.strip()) for d in cpc_digits.split(",") if d.strip()]
    max_digits = max(digit_levels) if digit_levels else 3
    cpc_dict_full = load_cpc_dict(mode, CORPUS_ROOT, cpc_digits=max_digits)

    # Build truncated dicts for each digit level
    cpc_dicts_by_digit: Dict[int, Dict[int, List[str]]] = {}
    for dl in digit_levels:
        if dl == max_digits:
            cpc_dicts_by_digit[dl] = cpc_dict_full
        else:
            cpc_dicts_by_digit[dl] = {
                aid: list(set(c[:dl] for c in codes))
                for aid, codes in cpc_dict_full.items()
            }

    # ---- Load wiki category hierarchy paths ----
    hierarchy_paths: Dict[str, str] = {}
    if not skip_wiki_category:
        df_pages_full = pd.read_csv(pages_csv)
        if "hierarchy_path" in df_pages_full.columns:
            for title, path in zip(
                df_pages_full["page_title"].astype(str),
                df_pages_full["hierarchy_path"].fillna("").astype(str),
            ):
                if path:
                    hierarchy_paths[title] = path
            print(f"[ext] Loaded hierarchy paths for {len(hierarchy_paths)} pages")
        else:
            print("[ext] No hierarchy_path column in pages_csv, skipping wiki category")
            skip_wiki_category = True

    # ---- Load citation graph (reuse a pre-built one if the caller supplied it) ----
    has_citations = citation_csv.exists()
    if citation_graph is None:
        citation_graph = {}
        if has_citations:
            family_to_applns, appln_to_family = load_family_to_applns(mode, CORPUS_ROOT)
            corpus_appln_ids = set(appln_to_family.keys())
            citation_graph = load_citation_graph(
                citation_csv, corpus_appln_ids, appln_to_family, family_to_applns,
            )
            del family_to_applns, appln_to_family, corpus_appln_ids
            gc.collect()
        else:
            print(f"[ext] Citation CSV not found: {citation_csv} — skipping citation analysis")

    # ---- Load patent embeddings + FAISS search (abstract view) ----
    view = "abstract"
    patent_emb, patent_ids = load_embeddings(
        mode=mode, view=view, embed_model=embed_model,
    )
    print(f"[ext] {patent_emb.shape[0]:,} patents loaded (mmap, view={view})")

    max_sims, best_wiki_idx = compute_max_similarities(
        patent_emb, patent_ids, wiki_index,
    )
    n_above = int((max_sims >= threshold).sum())
    print(f"[ext] {n_above:,}/{len(max_sims):,} patents above threshold {threshold}")

    # ---- Run analyses for each level ----
    summary = {"mode": mode, "threshold": threshold, "view": view, "levels": {}}
    cpc_by_level: Dict[str, pd.DataFrame] = {}
    citation_by_level: Dict[str, pd.DataFrame] = {}
    wikicat_by_level: Dict[str, pd.DataFrame] = {}

    for level in level_cols:
        if level not in df_assign.columns:
            continue

        labels = df_assign[level].values
        # Build page_to_cluster (exclude Peripheral/nan/None)
        page_to_cluster: Dict[str, str] = {}
        for page, lbl in zip(df_assign["page"], labels):
            if lbl in ("Peripheral", "nan", "None"):
                continue
            page_to_cluster[page] = lbl

        n_clusters = len(set(page_to_cluster.values()))
        print(f"\n--- Level: {level} ({n_clusters} clusters, {len(page_to_cluster)} core pages) ---")

        level_summary: dict = {"n_clusters": n_clusters}

        # ---- CPC sharing ratio at each digit level ----
        for dl in digit_levels:
            digit_label = f"{dl}digit"
            cpc_ratio_df = compute_cluster_cpc_ratio(
                page_to_cluster, wiki_page_titles, max_sims, best_wiki_idx,
                patent_ids, cpc_dicts_by_digit[dl], threshold,
            )
            ratio_csv = output_dir / f"cluster_cpc_ratio_{mode}_{level}_{digit_label}.csv"
            cpc_ratio_df.to_csv(ratio_csv, index=False)
            median_cpc_ratio = float(cpc_ratio_df["cpc_sharing_ratio"].median()) if len(cpc_ratio_df) else 0
            print(f"[cpc-{dl}d] cpc_ratio={median_cpc_ratio:.2f}")

            if dl == min(digit_levels):
                cpc_by_level[level] = cpc_ratio_df  # keep for dashboard

            level_summary[f"cpc_{digit_label}"] = {
                "median_cpc_ratio": round(median_cpc_ratio, 4),
            }

        # ---- Citation ratio ----
        if citation_graph:
            cit_df = compute_cluster_citation_coherence(
                page_to_cluster, wiki_page_titles, max_sims, best_wiki_idx,
                patent_ids, citation_graph, threshold,
            )
            cit_csv = output_dir / f"cluster_citation_{mode}_{level}.csv"
            cit_df.to_csv(cit_csv, index=False)
            print(f"[cit] {cit_csv.name}: {len(cit_df)} clusters")
            citation_by_level[level] = cit_df

            median_ratio = float(cit_df["citation_ratio"].median()) if len(cit_df) else 0
            print(f"[cit] Citation ratio: median={median_ratio:.1f}")

            level_summary["citation"] = {
                "median_citation_ratio": round(median_ratio, 2),
            }

        summary["levels"][level] = level_summary

    # ---- Write summary JSON ----
    summary_path = output_dir / f"cluster_external_validation_{mode}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[ext] Summary → {summary_path}")

    # Free memory
    del patent_emb, patent_ids, max_sims, best_wiki_idx
    gc.collect()

    return output_dir, cpc_by_level, citation_by_level, wikicat_by_level


# ---------------------------------------------------------------------------
# Main analysis (semantic margins + coverage)
# ---------------------------------------------------------------------------

def run_cluster_quality(
    run_dir: Path,
    input_csv: Path,
    output_dir: Optional[Path] = None,
    embed_model: str = "AI-Growth-Lab/PatentSBERTa",
    level_cols: Optional[List[str]] = None,
    mode: Optional[str] = None,
    skip_coverage: bool = False,
) -> Path:
    """Run per-cluster quality analysis on a completed clustering run.

    Two parts:
    1. **Per-cluster margin analysis** — for each cluster, compute within-sim
       vs nearest-other-centroid-sim and report margins.
    2. **Post-clustering coverage** (unless *skip_coverage*) — filter the input
       CSV to core pages only (excluding Leiden peripherals), then run the full
       patent-wiki coverage diagnostics from ``analysis_coverage``.  Results are
       saved in ``<run_dir>/coverage_report_post_clustering/``.

    Parameters
    ----------
    run_dir : Path
        Directory containing ``page_assignments_L2*.csv`` from a clustering run.
    input_csv : Path
        The original input CSV used for clustering (needed for descriptions).
    output_dir : Path or None
        Where to write margin results. Defaults to ``run_dir/cluster_quality``.
    embed_model : str
        SentenceTransformer model (must match what clustering used).
    level_cols : list[str] or None
        Taxonomy level columns. Defaults to ``["root", "L1", "L2"]``.
    mode : str or None
        ``"digital"`` or ``"green"``.  Required unless *skip_coverage* is True.
    skip_coverage : bool
        If True, skip the post-clustering patent-wiki coverage analysis.

    Returns
    -------
    Path
        The output directory.
    """
    import torch

    if level_cols is None:
        level_cols = LEVEL_COLS

    if output_dir is None:
        output_dir = run_dir / "cluster_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Find page assignments ----
    assign_files = list(run_dir.glob("page_assignments_L2*.csv"))
    if not assign_files:
        assign_files = list(run_dir.glob("page_assignments_*.csv"))
    if not assign_files:
        raise FileNotFoundError(f"No page_assignments CSV found in {run_dir}")

    assign_csv = assign_files[0]
    print(f"[QUALITY] Page assignments: {assign_csv}")

    df_assign = pd.read_csv(assign_csv)
    df_assign["page"] = df_assign["page"].astype(str)
    print(f"[QUALITY] {len(df_assign)} pages loaded")

    # ---- Load input CSV for descriptions ----
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df_input = pd.read_csv(input_csv)
    df_input["page_title"] = df_input["page_title"].astype(str)

    # ---- Load/build embeddings (will hit cache) ----
    desc_texts = [
        clean_description_text(x)
        for x in df_input["description"].fillna("").astype(str).tolist()
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb_all = load_or_build_embeddings(
        desc_texts=desc_texts,
        input_path=input_csv,
        out_dir=CACHE_ROOT,
        embed_model=embed_model,
        device=device,
        desc_col="description",
    )

    # ---- Map pages to embedding indices ----
    title_to_idx = {t: i for i, t in enumerate(df_input["page_title"].tolist())}

    pages = df_assign["page"].tolist()
    valid_mask = np.array([p in title_to_idx for p in pages])
    emb_indices = np.array([title_to_idx.get(p, 0) for p in pages])

    n_valid = int(valid_mask.sum())
    print(f"[QUALITY] {n_valid}/{len(pages)} pages matched to embeddings")

    emb_pages = emb_all[emb_indices]

    # ---- Per-level analysis ----
    summary_lines: list[str] = []
    summary_lines.append("=" * 80)
    summary_lines.append("CLUSTER QUALITY REPORT -- Per-Cluster Margin Analysis")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Run:   {run_dir}")
    summary_lines.append(f"Input: {input_csv}")
    summary_lines.append(f"Pages: {n_valid} matched / {len(pages)} total")
    summary_lines.append("")

    level_stats: list[tuple[str, int, int, float]] = []

    for level in level_cols:
        if level not in df_assign.columns:
            print(f"[QUALITY] Column '{level}' not in assignments, skipping")
            continue

        labels = df_assign[level].astype(str).values

        # Filter: valid embedding match + non-Peripheral + not NaN
        mask = (
            valid_mask
            & (labels != "Peripheral")
            & (labels != "nan")
            & (labels != "None")
        )
        if mask.sum() < 2:
            print(f"[QUALITY] Level '{level}': <2 valid pages, skipping")
            continue

        labels_filtered = labels[mask]
        emb_filtered = emb_pages[mask]

        unique = np.unique(labels_filtered)
        if len(unique) < 2:
            print(f"[QUALITY] Level '{level}': <2 clusters, skipping")
            continue

        # Compute per-cluster margins
        margins_df = compute_cluster_margins(emb_filtered, labels_filtered)
        margins_df = margins_df.sort_values("margin")

        # Write per-cluster CSV
        csv_path = output_dir / f"cluster_margins_{level}.csv"
        margins_df.to_csv(csv_path, index=False)
        print(f"[QUALITY] Wrote {csv_path}")

        # Coherence
        coh_df = _per_cluster_coherence(emb_filtered, labels_filtered)
        coh_csv = output_dir / f"cluster_coherence_{level}.csv"
        coh_df.to_csv(coh_csv, index=False)
        print(f"[QUALITY] Coherence: mean={coh_df['coherence'].mean():.4f}")
        if mode:
            pass
        # Summary stats
        n_neg = int((margins_df["margin"] < 0).sum())
        pct_neg = 100 * n_neg / len(margins_df)
        median_margin = float(margins_df["margin"].median())
        mean_margin = float(margins_df["margin"].mean())
        min_margin = float(margins_df["margin"].min())
        max_margin = float(margins_df["margin"].max())

        level_stats.append((level, n_neg, len(unique), median_margin))

        summary_lines.append(f"--- Level: {level} ({len(unique)} clusters) ---")
        summary_lines.append(f"  Negative margin: {n_neg}/{len(unique)} ({pct_neg:.1f}%)")
        summary_lines.append(f"  Mean margin:     {mean_margin:.4f}")
        summary_lines.append(f"  Median margin:   {median_margin:.4f}")
        summary_lines.append(f"  Min margin:      {min_margin:.4f}")
        summary_lines.append(f"  Max margin:      {max_margin:.4f}")
        summary_lines.append("")
        summary_lines.append("  Worst 5 clusters:")
        for _, r in margins_df.head(5).iterrows():
            cid = str(r["cluster_id"])
            # Truncate long cluster IDs for display
            cid_short = cid if len(cid) <= 50 else cid[:47] + "..."
            summary_lines.append(
                f"    {cid_short:<52} "
                f"margin={r['margin']:+.4f}  "
                f"within={r['within_sim']:.4f}  "
                f"nearest={r['nearest_other_sim']:.4f}  "
                f"n={r['n_pages']}"
            )
            nearest = str(r["nearest_cluster_id"])
            if len(nearest) > 55:
                nearest = nearest[:52] + "..."
            summary_lines.append(f"      -> nearest: {nearest}")
        summary_lines.append("")

        # Best 5 clusters
        summary_lines.append("  Best 5 clusters:")
        for _, r in margins_df.tail(5).iloc[::-1].iterrows():
            cid = str(r["cluster_id"])
            cid_short = cid if len(cid) <= 50 else cid[:47] + "..."
            summary_lines.append(
                f"    {cid_short:<52} "
                f"margin={r['margin']:+.4f}  "
                f"within={r['within_sim']:.4f}  "
                f"nearest={r['nearest_other_sim']:.4f}  "
                f"n={r['n_pages']}"
            )
        summary_lines.append("")

    # ---- Overall assessment ----
    summary_lines.append("=" * 80)
    summary_lines.append("OVERALL ASSESSMENT")
    summary_lines.append("=" * 80)

    for level, n_neg, total, med in level_stats:
        if total == 0:
            status = "N/A"
        elif n_neg == 0:
            status = "GOOD"
        elif n_neg / total < 0.1:
            status = "OK"
        elif n_neg / total < 0.25:
            status = "WARN"
        else:
            status = "POOR"
        summary_lines.append(
            f"  {level:<6}: {n_neg}/{total} negative-margin clusters "
            f"[{status}]  median margin={med:.4f}"
        )

    summary_lines.append("")

    # Write summary
    summary_path = output_dir / "cluster_quality_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[QUALITY] Wrote {summary_path}")

    # Print summary to stdout
    print()
    for line in summary_lines:
        print(line)

    # ---- Post-clustering coverage (core pages only) ----
    if not skip_coverage:
        if mode is None:
            print("[QUALITY] Skipping post-clustering coverage: --mode not specified")
        else:
            # Identify core pages (non-peripheral at L2 level)
            l2_col = "L2" if "L2" in df_assign.columns else None
            if l2_col is None:
                for col in reversed(LEVEL_COLS):
                    if col in df_assign.columns:
                        l2_col = col
                        break
            if l2_col is not None:
                labels_cov = df_assign[l2_col].astype(str)
                is_peripheral = (
                    labels_cov.str.contains("Peripheral", case=False, na=False)
                    | (labels_cov == "nan")
                    | (labels_cov == "None")
                )
                core_pages = set(df_assign.loc[~is_peripheral, "page"].tolist())
                print(f"\n[QUALITY] Post-clustering coverage: {len(core_pages)} core pages, "
                      f"{int(is_peripheral.sum())} peripheral excluded")

                df_core = df_input[df_input["page_title"].isin(core_pages)].copy()
                print(f"[QUALITY] Filtered input: {len(df_core)} rows (from {len(df_input)})")

                core_csv = run_dir / "core_pages_filtered.csv"
                df_core.to_csv(core_csv, index=False)

                coverage_output = run_dir / "coverage_report_post_clustering"
                try:
                    run_coverage_analysis(
                        wiki_csv=core_csv,
                        mode=mode,
                        output_dir=coverage_output,
                        embed_model=embed_model,
                        threshold=0.5,
                    )
                    print(f"[QUALITY] Post-clustering coverage → {coverage_output}")
                except Exception as e:
                    print(f"[QUALITY] Post-clustering coverage failed: {e}")
            else:
                print("[QUALITY] No level columns found, skipping coverage")

    return output_dir


# ===========================================================================
# Page Distance Functions  (from analysis_page_distance.py)
# ===========================================================================


def normalize_rows(x: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm


def build_centroids(
    emb: np.ndarray,
    cluster_ids: np.ndarray,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """
    Returns:
      C: (K, D) normalized centroid matrix
      clusters: list of cluster ids (strings) in row order
      cluster_to_row: mapping cluster id -> row index in C
    """
    clusters = pd.unique(cluster_ids.astype(str))
    clusters = [str(c) for c in clusters if str(c) != "nan"]
    # stable order
    clusters.sort()

    D = emb.shape[1]
    C = np.zeros((len(clusters), D), dtype=np.float32)

    for i, cid in enumerate(clusters):
        idx = np.where(cluster_ids.astype(str) == cid)[0]
        if len(idx) == 0:
            continue
        centroid = emb[idx].mean(axis=0)
        nrm = float(np.linalg.norm(centroid))
        if nrm > 0:
            centroid = centroid / nrm
        C[i] = centroid.astype(np.float32)

    cluster_to_row = {cid: i for i, cid in enumerate(clusters)}
    return C, clusters, cluster_to_row


def closest_other(
    sims: np.ndarray,
    own_col: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    sims: (N,K) cosine similarity
    own_col: (N,) int column index of the page's own cluster centroid in sims,
             or -1 if unknown
    Returns:
      best_col: (N,) argmax column excluding own_col (or argmax if own_col == -1)
      best_sim: (N,) max similarity value excluding own
    """
    N, K = sims.shape
    best_col = np.zeros(N, dtype=np.int32)
    best_sim = np.full(N, -np.inf, dtype=np.float32)

    # If K==1, closest other doesn't exist; return -1 and -inf
    if K <= 1:
        return np.full(N, -1, dtype=np.int32), np.full(N, -np.inf, dtype=np.float32)

    # Vectorized approach: mask own column by setting to -inf, then argmax
    masked = sims.copy()
    valid = own_col >= 0
    rows = np.where(valid)[0]
    masked[rows, own_col[rows]] = -np.inf

    best_col = masked.argmax(axis=1).astype(np.int32)
    best_sim = masked[np.arange(N), best_col].astype(np.float32)
    return best_col, best_sim


# ---------------------------------------------------------------------------
# Page distance plotting helpers
# ---------------------------------------------------------------------------

def _plot_page_margin_histogram(
    df: pd.DataFrame,
    level: str,
    mode: str,
    output_dir: Path,
) -> None:
    """Histogram + KDE of per-page margin for one taxonomy level."""
    col = f"{level}_margin"
    if col not in df.columns:
        return
    vals = df[col].dropna().values.astype(float)
    if len(vals) == 0:
        return

    pct_pos = float((vals > 0).mean() * 100)
    mean_val = float(np.mean(vals))
    median_val = float(np.median(vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals, bins=60, density=True, alpha=0.5, color="steelblue", label="histogram")
    if sns is not None:
        try:
            sns.kdeplot(vals, ax=ax, color="darkblue", linewidth=1.5, label="KDE")
        except Exception:
            pass

    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="margin = 0")
    ax.axvline(median_val, color="orange", linestyle=":", linewidth=1.2,
               label=f"median = {median_val:.4f}")

    stats_text = (
        f"N = {len(vals):,}\n"
        f"mean = {mean_val:.4f}\n"
        f"median = {median_val:.4f}\n"
        f"positive margin: {pct_pos:.1f}%"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Margin  (closest-other dist − own dist)")
    ax.set_ylabel("Density")
    ax.set_title(f"Page Margin Distribution  [{level}]  ({mode})")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path = output_dir / f"page_margin_dist_{mode}_{level}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] {out_path}", flush=True)


def plot_page_distances(
    df: pd.DataFrame,
    level_cols: list,
    mode: str,
    output_dir: Path,
) -> None:
    """Generate all page-distance plots for each taxonomy level."""
    for level in level_cols:
        _plot_page_margin_histogram(df, level, mode, output_dir)


# ---------------------------------------------------------------------------
# Page distance core computation
# ---------------------------------------------------------------------------

def run_page_distances(
    taxonomy_csv: Path,
    pages_csv: Path,
    *,
    taxonomy_page_col: str = "page",
    taxonomy_cols: str = "root,L1,L2",
    pages_title_col: str = "page_title",
    pages_desc_col: str = "description",
    embed_model: str = "AI-Growth-Lab/PatentSBERTa",
    device: str | None = None,
    batch_size: int = 64,
    chunk_size: int = 5000,
    out: str | None = None,
    keep_missing: bool = True,
    cache_tag: str = "",
    exclude_peripheral: bool = False,
    emb_transform: "callable | None" = None,
    mode: str = "",
    do_plot: bool = True,
) -> Path:
    """Compute per-page semantic distances to cluster centroids.

    Returns the output CSV path.
    """
    taxonomy_path = Path(taxonomy_csv)
    pages_path = Path(pages_csv)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not taxonomy_path.exists():
        raise SystemExit(f"taxonomy_csv not found: {taxonomy_path}")
    if not pages_path.exists():
        raise SystemExit(f"pages_csv not found: {pages_path}")

    level_cols = [c.strip() for c in taxonomy_cols.split(",") if c.strip()]
    page_col = taxonomy_page_col

    df_tax = pd.read_csv(taxonomy_path)
    missing = [c for c in (level_cols + [page_col]) if c not in df_tax.columns]
    if missing:
        raise SystemExit(f"taxonomy_csv missing columns: {missing}")

    # Filter out peripheral pages
    if exclude_peripheral:
        before = len(df_tax)
        for col in level_cols:
            vals = df_tax[col].astype(str)
            mask = (
                vals.str.contains("Peripheral", case=False, na=False)
                | (vals == "nan")
                | (vals == "None")
            )
            df_tax = df_tax[~mask]
        df_tax = df_tax.reset_index(drop=True)
        print(f"[INFO] Excluded {before - len(df_tax)} peripheral pages, {len(df_tax)} core remain", flush=True)

    df_pages = pd.read_csv(pages_path)
    if pages_title_col not in df_pages.columns:
        raise SystemExit(f"pages_csv missing title col: {pages_title_col}")
    if pages_desc_col not in df_pages.columns:
        df_pages[pages_desc_col] = ""

    # Map page -> description from pages_csv (source of truth)
    df_pages[pages_title_col] = df_pages[pages_title_col].astype(str)
    df_pages[pages_desc_col] = df_pages[pages_desc_col].fillna("").astype(str)
    desc_map = df_pages.set_index(pages_title_col)[pages_desc_col].to_dict()

    df = df_tax[level_cols + [page_col]].copy()
    df[page_col] = df[page_col].astype(str)

    # Attach description
    df["description"] = df[page_col].map(desc_map)
    df["description"] = df["description"].fillna("").astype(str)
    df["desc_len"] = df["description"].astype(str).str.len()

    if not keep_missing:
        before = len(df)
        df = df[df["desc_len"] > 0].reset_index(drop=True)
        print(f"[INFO] Dropped pages with missing/empty description: {before - len(df):,}", flush=True)

    print(f"[INFO] Pages for distance calc: {len(df):,}", flush=True)
    print(f"[EMBED] model={embed_model} device={device} (using cache)", flush=True)

    # Use cached embeddings: load full pages_csv embeddings, then map to taxonomy pages
    desc_texts_all = [
        clean_description_text(x)
        for x in df_pages[pages_desc_col].fillna("").astype(str).tolist()
    ]
    emb_all = load_or_build_embeddings(
        desc_texts=desc_texts_all,
        input_path=pages_path,
        out_dir=CACHE_ROOT,
        embed_model=embed_model,
        device=device,
        desc_col=pages_desc_col,
        cache_tag=cache_tag,
    )
    emb_all = normalize_rows(emb_all)

    # Optional embedding transformation (PCA, whitening, etc.)
    if emb_transform is not None:
        print(f"[INFO] Applying embedding transform before distance computation", flush=True)
        emb_all = emb_transform(emb_all)

    # Map taxonomy pages to their embedding indices in the pages_csv
    title_to_idx = {t: i for i, t in enumerate(df_pages[pages_title_col].tolist())}
    emb_indices = np.array([title_to_idx.get(p, 0) for p in df[page_col].tolist()])
    emb = emb_all[emb_indices]

    # Output frame
    out_df = df[[page_col] + level_cols + ["desc_len"]].copy()

    # Compute distances per level
    for lvl in level_cols:
        cluster_ids = df[lvl].astype(str).values

        C, clusters, cluster_to_row = build_centroids(emb, cluster_ids)
        K = C.shape[0]
        print(f"[LEVEL {lvl}] clusters={K:,}", flush=True)

        own_col = np.array([cluster_to_row.get(str(cid), -1) for cid in cluster_ids], dtype=np.int32)

        N = emb.shape[0]
        own_sim = np.full(N, np.nan, dtype=np.float32)
        best_other_sim = np.full(N, np.nan, dtype=np.float32)
        best_other_col = np.full(N, -1, dtype=np.int32)

        if K > 0:
            for start in tqdm(range(0, N, chunk_size), desc=f"Scoring {lvl}", unit="chunk"):
                end = min(N, start + chunk_size)
                Eblk = emb[start:end]
                sims = (Eblk @ C.T).astype(np.float32)

                oc = own_col[start:end]
                valid = oc >= 0
                blk_own = np.full(end - start, np.nan, dtype=np.float32)
                blk_own[valid] = sims[np.arange(end - start)[valid], oc[valid]]
                own_sim[start:end] = blk_own

                boc, bsim = closest_other(sims, oc)
                best_other_col[start:end] = boc
                best_other_sim[start:end] = bsim

        out_df[f"{lvl}_dist_to_own"] = (1.0 - own_sim).astype(np.float32)
        out_df[f"{lvl}_closest_other_dist"] = (1.0 - best_other_sim).astype(np.float32)

        def col_to_cluster(col_idx: int) -> str:
            if col_idx < 0 or col_idx >= len(clusters):
                return ""
            return clusters[col_idx]

        out_df[f"{lvl}_closest_other_cluster"] = [col_to_cluster(int(i)) for i in best_other_col.tolist()]
        out_df[f"{lvl}_margin"] = (out_df[f"{lvl}_closest_other_dist"] - out_df[f"{lvl}_dist_to_own"]).astype(np.float32)

    # Output path
    if out is None:
        out_path = taxonomy_path.parent / f"page_distances__from_{taxonomy_path.stem}.csv"
    else:
        out_path = Path(out)

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {out_path}", flush=True)

    if do_plot and mode:
        plot_page_distances(out_df, level_cols, mode, out_path.parent)

    return out_path


# ===========================================================================
# Coverage Analysis Functions  (from analysis_coverage.py)
# ===========================================================================


# ---------------------------------------------------------------------------
# Wiki embedding + FAISS index
# ---------------------------------------------------------------------------

_WIKI_EMB_CACHE_DIR = Path(
    r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\taxonomy"
    r"\wiki\cache\wiki_embedding_cache"
)


def load_wiki_embeddings(
    wiki_csv: Path,
    embed_model: str,
    device: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load wiki page description embeddings from shared cache.

    Returns (L2-normed float32 array, DataFrame).  Loads from
    ``wiki_embedding_cache/wiki_<ModelName>.npz`` (shared with cluster step).  If not found or shape mismatches,
    computes and saves there.
    """
    df = pd.read_csv(wiki_csv)
    df["page_title"] = df["page_title"].astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    model_short = embed_model.split("/")[-1]
    cache_path = _WIKI_EMB_CACHE_DIR / f"wiki_{model_short}.npz"

    if cache_path.exists():
        try:
            emb = np.load(cache_path)["emb"]
            if emb.shape[0] == len(df):
                print(f"[wiki-emb] Cache HIT: {cache_path}")
                return emb.astype(np.float32, copy=False), df
            print(f"[wiki-emb] Cache row mismatch ({emb.shape[0]} vs {len(df)}), recomputing")
        except Exception as exc:
            print(f"[wiki-emb] Cache load failed ({exc}), recomputing")

    from sentence_transformers import SentenceTransformer

    texts = df["description"].tolist()
    print(f"[wiki-emb] Encoding {len(texts):,} wiki descriptions on {device} …")

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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, emb=emb)
    print(f"[wiki-emb] Cached → {cache_path}")
    return emb, df


def build_wiki_faiss_index(wiki_emb: np.ndarray, save_path: Optional[Path] = None):
    """Build or load a FAISS FlatIP (exact inner product) index from wiki embeddings.

    If *save_path* is given and a file already exists whose vector count matches
    ``wiki_emb``, it is loaded directly.  Otherwise the index is built from
    *wiki_emb* and written to *save_path*.
    """
    import faiss

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.exists():
            idx = faiss.read_index(str(save_path))
            if idx.ntotal == wiki_emb.shape[0]:
                print(f"[faiss] Loaded FlatIP index from {save_path}  n={idx.ntotal:,}")
                return idx
            print(f"[faiss] Index n mismatch ({idx.ntotal} vs {wiki_emb.shape[0]}), rebuilding")

    dim = wiki_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(wiki_emb)
    print(f"[faiss] Built FlatIP index  dim={dim}  n={index.ntotal:,}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(save_path))
        print(f"[faiss] Saved → {save_path}")

    return index


# ---------------------------------------------------------------------------
# CPC dictionary
# ---------------------------------------------------------------------------

def load_cpc_dict(mode: str, corpus_root: str = CORPUS_ROOT,
                  cpc_digits: int = 3) -> Dict[int, List[str]]:
    """Stream parquet parts and parse pipe-separated ``cpc_coarse_list``.

    Returns ``{appln_id: ["H04", "G06", ...]}``.

    Parameters
    ----------
    cpc_digits : int
        Truncate CPC codes to this many characters (3 = class, 4 = subclass).
    """
    parts_dir = os.path.join(corpus_root, mode)
    parts = _list_part_files(parts_dir)

    cpc_dict: Dict[int, List[str]] = {}
    cols = ["appln_id", "cpc_coarse_list"]

    for df in tqdm(_iter_parts(parts, cols), total=len(parts), desc="[cpc] Loading"):
        df["appln_id"] = df["appln_id"].astype(np.int64)
        df["cpc_coarse_list"] = df["cpc_coarse_list"].fillna("").astype(str)
        for aid, cpc_str in zip(df["appln_id"], df["cpc_coarse_list"]):
            codes = list(dict.fromkeys(c[:cpc_digits] for c in cpc_str.split("|") if c))
            if codes:
                cpc_dict[int(aid)] = codes

    print(f"[cpc] Loaded {len(cpc_dict):,} patents with CPC codes (level={cpc_digits})")
    return cpc_dict


# ---------------------------------------------------------------------------
# Citation graph
# ---------------------------------------------------------------------------

CITATION_CSV = r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\patent\state_patent_citations.csv"


def load_family_to_applns(
    mode: str,
    corpus_root: str = CORPUS_ROOT,
) -> Tuple[Dict[int, Set[int]], Dict[int, int]]:
    """Stream parquet parts reading [appln_id, docdb_family_id].

    Returns
    -------
    family_to_applns : dict[int, set[int]]
        Maps each family id to its set of application ids in the corpus.
    appln_to_family : dict[int, int]
        Maps each application id to its family id.
    """
    parts_dir = os.path.join(corpus_root, mode)
    parts = _list_part_files(parts_dir)
    cols = ["appln_id", "docdb_family_id"]

    family_to_applns: Dict[int, Set[int]] = defaultdict(set)
    appln_to_family: Dict[int, int] = {}

    for df in tqdm(_iter_parts(parts, cols), total=len(parts), desc="[citation] Loading families"):
        df["appln_id"] = df["appln_id"].astype(np.int64)
        df["docdb_family_id"] = df["docdb_family_id"].astype(np.int64)
        for aid, fid in zip(df["appln_id"], df["docdb_family_id"]):
            aid, fid = int(aid), int(fid)
            family_to_applns[fid].add(aid)
            appln_to_family[aid] = fid

    print(f"[citation] {len(appln_to_family):,} applns in {len(family_to_applns):,} families")
    return family_to_applns, appln_to_family


def load_citation_graph(
    citation_csv: Path,
    corpus_appln_ids: Set[int],
    appln_to_family: Dict[int, int],
    family_to_applns: Dict[int, Set[int]],
) -> Dict[int, Set[int]]:
    """Build bidirectional citation graph over corpus application ids.

    Reads the citation CSV in chunks.  For each row the citing appln is
    mapped to its family, then to all corpus applns in that family, and
    likewise for the cited side.  Edges are added in *both* directions so
    that a 1-hop lookup finds all neighbours regardless of citation
    direction.  Only edges where both endpoints are in the corpus are kept.

    Returns
    -------
    graph : dict[int, set[int]]
        Bidirectional adjacency list keyed by ``appln_id``.
    """
    graph: Dict[int, Set[int]] = defaultdict(set)
    n_edges = 0

    reader = pd.read_csv(
        citation_csv,
        usecols=["docdb_family_id", "cited_docdb_family_id"],
        chunksize=100_000,
        dtype={"docdb_family_id": np.int64, "cited_docdb_family_id": np.int64},
    )

    for chunk in tqdm(reader, desc="[citation] Building graph"):
        for citing_fam, cited_fam in zip(
            chunk["docdb_family_id"], chunk["cited_docdb_family_id"]
        ):
            citing_fam, cited_fam = int(citing_fam), int(cited_fam)
            citing_applns = family_to_applns.get(citing_fam, set())
            cited_applns = family_to_applns.get(cited_fam, set())
            if not citing_applns or not cited_applns:
                continue
            # Only keep applns that are in the corpus
            citing_in = citing_applns & corpus_appln_ids
            cited_in = cited_applns & corpus_appln_ids
            if not citing_in or not cited_in:
                continue
            for a in citing_in:
                graph[a].update(cited_in)
            for b in cited_in:
                graph[b].update(citing_in)
            n_edges += len(citing_in) * len(cited_in)

    print(f"[citation] Graph: {len(graph):,} nodes, ~{n_edges:,} directed edges")
    return dict(graph)


# ---------------------------------------------------------------------------
# Chunked FAISS search
# ---------------------------------------------------------------------------

def compute_max_similarities(
    patent_emb: np.ndarray,
    patent_ids: np.ndarray,
    wiki_index,
    chunk_size: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search patents against wiki FAISS index (k=1) in chunks.

    Returns ``(max_sims[N], best_wiki_idx[N])`` — both arrays of length N.
    """
    n = patent_emb.shape[0]
    max_sims = np.empty(n, dtype=np.float32)
    best_wiki_idx = np.empty(n, dtype=np.int64)

    for start in tqdm(range(0, n, chunk_size), desc="[faiss] Searching"):
        end = min(n, start + chunk_size)
        # mmap slice → contiguous copy for FAISS
        chunk = np.ascontiguousarray(patent_emb[start:end], dtype=np.float32)
        D, I = wiki_index.search(chunk, 1)  # (chunk_len, 1)
        max_sims[start:end] = D[:, 0]
        best_wiki_idx[start:end] = I[:, 0]

    return max_sims, best_wiki_idx


# ---------------------------------------------------------------------------
# Coverage Analysis 1: Similarity distribution
# ---------------------------------------------------------------------------

def _analysis_similarity_distribution(
    max_sims: np.ndarray,
    view: str,
    output_dir: Path,
    mode: str,
) -> dict:
    """Histogram + KDE of per-patent top-1 similarity to closest wiki page."""
    median_val = float(np.median(max_sims))
    mean_val = float(np.mean(max_sims))
    std_val = float(np.std(max_sims))
    pct_above_05 = float((max_sims >= 0.5).mean() * 100)
    pct_above_06 = float((max_sims >= 0.6).mean() * 100)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(max_sims, bins=100, range=(0, 1), density=True, alpha=0.5, color="steelblue", label="histogram")
    try:
        sns.kdeplot(max_sims, ax=ax, color="darkblue", linewidth=1.5, label="KDE")
    except Exception:
        pass

    for val, label, color, ls in [
        (median_val, f"median={median_val:.3f}", "red", "--"),
        (0.5, "0.5", "orange", ":"),
        (0.6, "0.6", "green", ":"),
    ]:
        ax.axvline(val, color=color, linestyle=ls, linewidth=1.2, label=label)

    stats_text = (
        f"N={len(max_sims):,}\n"
        f"mean={mean_val:.4f}\n"
        f"std={std_val:.4f}\n"
        f">=0.5: {pct_above_05:.1f}%\n"
        f">=0.6: {pct_above_06:.1f}%"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Top-1 cosine similarity to nearest wiki page")
    ax.set_ylabel("Density")
    ax.set_title(f"Patent–Wiki Similarity Distribution  [{view}]")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / f"similarity_distribution_{mode}_{view}.png", dpi=200)
    plt.close(fig)

    return {
        "view": view,
        "n_patents": int(len(max_sims)),
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "pct_above_0.5": pct_above_05,
        "pct_above_0.6": pct_above_06,
    }


# ---------------------------------------------------------------------------
# Coverage Analysis 3: Threshold sensitivity
# ---------------------------------------------------------------------------

def _analysis_threshold_sensitivity(
    max_sims: np.ndarray,
    view: str,
    output_dir: Path,
    mode: str = "",
) -> dict:
    """Patent coverage (%) at different similarity thresholds."""
    thresholds = np.arange(0.0, 1.01, 0.05)
    n = len(max_sims)
    coverage = [(float(t), float((max_sims >= t).sum() / n * 100)) for t in thresholds]
    df_ts = pd.DataFrame(coverage, columns=["threshold", "pct_covered"])

    csv_path = output_dir / f"threshold_sensitivity_{mode}_{view}.csv"
    df_ts.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_ts["threshold"], df_ts["pct_covered"], marker="o", markersize=4, color="navy")

    # Annotate key points
    for t_val in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pct = float((max_sims >= t_val).sum() / n * 100)
        ax.annotate(
            f"{pct:.1f}%",
            xy=(t_val, pct),
            xytext=(t_val + 0.02, pct + 3),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("% patents with >= 1 page above threshold")
    ax.set_title(f"Threshold Sensitivity  [{view}]")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f"threshold_sensitivity_{mode}_{view}.png", dpi=200)
    plt.close(fig)

    return {"view": view, "thresholds": {f"{t:.2f}": pct for t, pct in coverage}}


# ---------------------------------------------------------------------------
# Coverage Analysis 6: Citation coherence (modularity ratio)
# ---------------------------------------------------------------------------

def _analysis_citation_modularity(
    max_sims: np.ndarray,
    best_wiki_idx: np.ndarray,
    patent_ids: np.ndarray,
    citation_graph: Dict[int, Set[int]],
    wiki_df: pd.DataFrame,
    view: str,
    output_dir: Path,
    threshold: float,
    mode: str = "",
) -> dict:
    """Do patents matched to the same page cite each other more than expected?

    Among citation edges where both endpoints are above threshold:
    - observed_rate = fraction connecting two patents on the same page
    - expected_rate = sum_p (n_p / N)^2  (Herfindahl index under random assignment)
    - modularity_ratio = observed / expected
    Ratio >> 1 means the taxonomy captures citation structure.
    """
    above_mask = max_sims >= threshold

    # Map above-threshold patents to their page index
    pat_to_page: Dict[int, int] = {}
    page_counts: Counter = Counter()
    for i in np.where(above_mask)[0]:
        pid = int(patent_ids[i])
        widx = int(best_wiki_idx[i])
        pat_to_page[pid] = widx
        page_counts[widx] += 1

    above_set = set(pat_to_page.keys())
    N = len(above_set)

    # Count intra-page and total citation edges (both endpoints above threshold)
    # Per-page tracking
    page_intra: Counter = Counter()
    page_total_edges: Counter = Counter()

    n_intra = 0
    n_total = 0

    for pid in above_set:
        widx = pat_to_page[pid]
        neighbors = citation_graph.get(pid, set())
        above_neighbors = neighbors & above_set
        for nb in above_neighbors:
            n_total += 1
            page_total_edges[widx] += 1
            if pat_to_page[nb] == widx:
                n_intra += 1
                page_intra[widx] += 1

    # Each edge counted twice (bidirectional graph), so rates are consistent
    n_inter = n_total - n_intra

    # Empirical intra/inter rates: normalise by possible pairs
    n_possible_intra = sum(n_p * (n_p - 1) for n_p in page_counts.values())  # directed pairs
    n_possible_inter = N * (N - 1) - n_possible_intra  # directed pairs across pages

    intra_rate = n_intra / max(n_possible_intra, 1)
    inter_rate = n_inter / max(n_possible_inter, 1)
    modularity_ratio = intra_rate / max(inter_rate, 1e-12)

    # Per-page breakdown
    rows = []
    for widx in sorted(page_counts.keys()):
        if widx < 0 or widx >= len(wiki_df):
            continue
        n_p = page_counts[widx]
        intra = page_intra.get(widx, 0)
        total_e = page_total_edges.get(widx, 0)
        inter_e = total_e - intra
        # Empirical: intra citations per possible intra pair vs inter citations per possible inter pair
        possible_intra_p = n_p * (n_p - 1) if n_p > 1 else 1
        possible_inter_p = n_p * (N - n_p) if (N - n_p) > 0 else 1
        intra_rate_p = intra / max(possible_intra_p, 1)
        inter_rate_p = inter_e / max(possible_inter_p, 1)
        page_ratio = intra_rate_p / max(inter_rate_p, 1e-12)
        rows.append({
            "wiki_page": wiki_df.iloc[widx]["page_title"],
            "n_patents": n_p,
            "n_intra_edges": intra,
            "n_total_edges": total_e,
            "intra_rate": intra_rate_p,
            "inter_rate": inter_rate_p,
            "coherence_ratio": page_ratio,
        })

    df_cit = pd.DataFrame(rows)
    df_cit.to_csv(output_dir / f"citation_modularity_{mode}_{view}.csv", index=False)

    # -- Plot: 3 panels --
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # (A) Intra vs inter rate bar
    ax1.bar(["Intra-page\ncitation rate", "Inter-page\ncitation rate"],
            [intra_rate, inter_rate],
            color=["steelblue", "salmon"])
    ax1.set_ylabel("Citations per possible pair")
    ax1.set_title(f"Citation Modularity  [{view}]\nratio={modularity_ratio:.2f}")
    for i, val in enumerate([intra_rate, inter_rate]):
        ax1.text(i, val + val * 0.05, f"{val:.6f}", ha="center", va="bottom", fontsize=10)

    # (B) Histogram of per-page coherence ratio
    if len(df_cit) > 0 and df_cit["coherence_ratio"].notna().any():
        finite_ratios = df_cit.loc[np.isfinite(df_cit["coherence_ratio"]), "coherence_ratio"]
        if len(finite_ratios) > 0:
            ax2.hist(finite_ratios, bins=40, color="teal", alpha=0.7, edgecolor="white")
            ax2.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="ratio=1")
            ax2.set_xlabel("Per-page coherence ratio")
            ax2.set_ylabel("# pages")
            ax2.set_title(f"Per-Page Modularity Distribution  [{view}]")
            ax2.legend(fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No finite ratios", ha="center", va="center", transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)

    # (C) Most/least modular 20 pages by ratio
    if len(df_cit) >= 2:
        finite_df = df_cit[np.isfinite(df_cit["coherence_ratio"])].copy()
        finite_df = finite_df[finite_df["n_total_edges"] >= 5]  # require minimum edges
        if len(finite_df) >= 2:
            top20 = finite_df.nlargest(20, "coherence_ratio")
            bottom20 = finite_df.nsmallest(20, "coherence_ratio")
            combined = pd.concat([top20, bottom20]).drop_duplicates()
            combined = combined.sort_values("coherence_ratio")
            top_pages = set(top20["wiki_page"])
            colors = ["#2ecc71" if p in top_pages else "#e74c3c" for p in combined["wiki_page"]]
            ax3.barh(combined["wiki_page"], combined["coherence_ratio"], color=colors)
            ax3.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
            ax3.set_xlabel("Modularity ratio")
            ax3.set_title(f"Most / Least Modular Pages (>=5 edges)  [{view}]")
            ax3.tick_params(axis="y", labelsize=6)
        else:
            ax3.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)

    plt.tight_layout()
    fig.savefig(output_dir / f"citation_modularity_{mode}_{view}.png", dpi=200)
    plt.close(fig)

    return {
        "view": view,
        "threshold": threshold,
        "n_above_threshold": N,
        "n_total_edges": n_total,
        "n_intra_edges": n_intra,
        "intra_rate": intra_rate,
        "inter_rate": inter_rate,
        "modularity_ratio": modularity_ratio,
        "n_pages_in_breakdown": len(df_cit),
    }


# ---------------------------------------------------------------------------
# Coverage analysis orchestrator
# ---------------------------------------------------------------------------

def run_coverage_analysis(
    wiki_csv: Path,
    mode: str,
    output_dir: Optional[Path] = None,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    threshold: float = 0.5,
    cache_root: str = PATENT_CACHE_ROOT,
    corpus_root: str = CORPUS_ROOT,
    chunk_size: int = 50_000,
    citation_csv: Optional[Path] = None,
    min_component_size: int = 10,
    citation_graph: Optional[Dict[int, Set[int]]] = None,
) -> dict:
    """Run all 7 coverage analyses for both title and abstract views.

    Parameters
    ----------
    wiki_csv : Path
        Filtered wiki taxonomy CSV (must have ``page_title``, ``description``).
    mode : str
        ``"green"`` or ``"digital"``.
    output_dir : Path or None
        Output directory; defaults to ``pipeline/05_coverage_pre/`` next to *wiki_csv*'s parent.
    embed_model : str
        SentenceTransformer model id.
    threshold : float
        Similarity threshold for coverage boundary (default 0.5).
    cache_root / corpus_root : str
        Embedding and corpus cache paths.
    chunk_size : int
        FAISS search chunk size.
    citation_csv : Path or None
        Path to citation CSV.  Defaults to ``CITATION_CSV``.
    min_component_size : int
        Minimum size for an unmatched patent cluster to be reported (default 10).

    Returns
    -------
    dict
        Summary of all analyses.
    """
    wiki_csv = Path(wiki_csv)
    if not wiki_csv.exists():
        raise FileNotFoundError(f"Wiki CSV not found: {wiki_csv}")

    if citation_csv is None:
        citation_csv = Path(CITATION_CSV)
    else:
        citation_csv = Path(citation_csv)

    if output_dir is None:
        output_dir = wiki_csv.parent.parent / "05_coverage_pre" / mode
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    print(f"\n{'='*60}")
    print(f"  COVERAGE DIAGNOSTICS  mode={mode}  threshold={threshold}")
    print(f"  wiki_csv     = {wiki_csv}")
    print(f"  citation_csv = {citation_csv}")
    print(f"  output       = {output_dir}")
    print(f"{'='*60}\n")

    # --- Load wiki embeddings & build FAISS index (once) ---
    device = device_auto()
    wiki_emb, wiki_df = load_wiki_embeddings(wiki_csv, embed_model, device)
    wiki_index_path = _WIKI_EMB_CACHE_DIR / "faiss_wiki.index"
    wiki_index = build_wiki_faiss_index(wiki_emb, save_path=wiki_index_path)

    # --- Load citation graph (once; reuse a pre-built one if the caller supplied it) ---
    has_citations = citation_csv.exists()
    if citation_graph is None:
        citation_graph = {}
        if has_citations:
            family_to_applns, appln_to_family = load_family_to_applns(mode, corpus_root)
            corpus_appln_ids = set(appln_to_family.keys())
            citation_graph = load_citation_graph(
                citation_csv, corpus_appln_ids, appln_to_family, family_to_applns,
            )
            # Free intermediate maps
            del family_to_applns, appln_to_family, corpus_appln_ids
            gc.collect()
        else:
            print(f"[warning] Citation CSV not found: {citation_csv} — skipping citation analyses (#6–#7)")

    summary: dict = {
        "mode": mode,
        "threshold": threshold,
        "wiki_pages": len(wiki_df),
        "embed_model": embed_model,
        "analyses": {},
    }

    views = ["abstract"]

    for view in views:
        print(f"\n--- View: {view} ---")

        # Load pre-computed patent embeddings (memory-mapped)
        patent_emb, patent_ids = load_embeddings(
            mode=mode,
            view=view,
            embed_model=embed_model,
            cache_root=cache_root,
            corpus_root=corpus_root,
        )
        print(f"[patents] {patent_emb.shape[0]:,} patents loaded (mmap)")

        # FAISS search
        max_sims, best_wiki_idx = compute_max_similarities(
            patent_emb, patent_ids, wiki_index, chunk_size=chunk_size,
        )

        # Run core analyses (CPC spread/gap/overlap/unmatched removed)
        print("[1/3] Similarity distribution …")
        r1 = _analysis_similarity_distribution(max_sims, view, output_dir, mode)

        print("[2/3] Threshold sensitivity …")
        r3 = _analysis_threshold_sensitivity(max_sims, view, output_dir, mode)

        view_results = {
            "similarity_distribution": r1,
            "threshold_sensitivity": r3,
        }

        if has_citations:
            print("[3/3] Citation modularity …")
            r6 = _analysis_citation_modularity(
                max_sims, best_wiki_idx, patent_ids, citation_graph, wiki_df, view, output_dir, threshold,
                mode=mode,
            )
            view_results["citation_modularity"] = r6
        else:
            print("[3/3] Skipped (no citation data)")

        summary["analyses"][view] = view_results

        # Free per-view arrays
        del patent_emb, patent_ids, max_sims, best_wiki_idx
        gc.collect()

    # Write summary JSON
    json_path = output_dir / "coverage_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] Summary → {json_path}")
    print(f"[done] All outputs → {output_dir}")

    return summary


# ===========================================================================
# LLM Verification Functions  (from analysis_cluster_llm_verify.py)
# ===========================================================================

# ── LLM verification defaults ─────────────────────────────────────────────

LLM_VERIFY_DEFAULT_MODEL = "google/gemini-2.5-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_PAGES_PER_PROMPT = 80   # truncate cluster page list if larger
SAMPLE_LARGE = 80           # sample this many pages from large clusters
MAX_RETRIES = 3

LLM_VERIFY_SYSTEM_PROMPT = """\
You are a technology taxonomy expert. You evaluate whether the pages in a \
cluster form a coherent technology group.

Respond ONLY with valid JSON. No markdown, no explanation outside JSON."""

LLM_VERIFY_USER_PROMPT_TEMPLATE = """\
Below is a cluster from a hierarchical technology taxonomy.

Level: {level}
Cluster ID: {cluster_id}
Number of pages: {n_pages} {sample_note}

Pages:
{page_list}

Evaluate this cluster:

1. **coherence** (1-5): How well do these pages belong together as a technology group?
   - 5: Perfect — all pages clearly belong to the same technology subfield
   - 4: Good — mostly coherent, minor outliers
   - 3: Mixed — some coherent subgroup(s) but also unrelated pages
   - 2: Weak — pages share only a vague theme
   - 1: Incoherent — no meaningful grouping

2. **label**: A short label (1-4 words) describing the technology theme of this cluster.

3. **misplaced**: List of page titles (if any) that clearly do NOT belong in this cluster. \
Return an empty list if all pages fit.

4. **reasoning**: One sentence explaining your rating.

Respond with JSON:
{{"coherence": <int 1-5>, "label": "<str>", "misplaced": [<str>, ...], "reasoning": "<str>"}}"""


def _build_page_list(pages: list[str], descriptions: dict[str, str] | None,
                     max_pages: int, sample: int) -> tuple[str, str]:
    """Build the page list string, sampling if needed."""
    n = len(pages)
    sample_note = ""

    if n > max_pages:
        import random
        rng = random.Random(42)
        pages = rng.sample(pages, min(sample, n))
        sample_note = f"(showing {len(pages)} sampled from {n})"

    lines = []
    for p in sorted(pages):
        if descriptions and p in descriptions:
            desc = descriptions[p][:120]
            lines.append(f"- {p}: {desc}")
        else:
            lines.append(f"- {p}")

    return "\n".join(lines), sample_note


def _call_llm_reasoning(
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int = 1024,
    timeout_s: int = 120,
) -> str | None:
    """Call OpenRouter with reasoning mode enabled.

    Returns the assistant content string, or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "st4te-taxonomy-verify",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": int(max_tokens),
        "reasoning": {"effort": "high"},
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"].strip()
    # Clean quotes
    text = re.sub(r'^[\'"\s]+|[\'"\s]+$', "", text)
    return text


def _parse_llm_response(text: str) -> dict:
    """Parse the LLM JSON response, with fallback regex extraction."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"coherence": None, "label": "", "misplaced": [], "reasoning": text}


def verify_clusters(
    taxonomy_csv: str | Path,
    pages_csv: str | Path | None = None,
    out_dir: str | Path = ".",
    levels: str = "root,L1,L2",
    model: str = LLM_VERIFY_DEFAULT_MODEL,
    max_pages_per_prompt: int = MAX_PAGES_PER_PROMPT,
    sample_large: int = SAMPLE_LARGE,
    include_descriptions: bool = True,
) -> Path:
    """Run LLM verification on all clusters in the taxonomy.

    Parameters
    ----------
    taxonomy_csv : path to the taxonomy long CSV (with page, root, L1, L2 cols)
    pages_csv : optional path to filtered CSV with page_title + description
    out_dir : output directory
    levels : comma-separated level columns to verify
    model : LLM model to use via OpenRouter
    max_pages_per_prompt : max pages to include per prompt
    sample_large : how many to sample from clusters exceeding max_pages
    include_descriptions : whether to include page descriptions in prompts

    Returns
    -------
    Path to the output CSV
    """
    taxonomy_csv = Path(taxonomy_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    level_cols = [l.strip() for l in levels.split(",")]

    # Load taxonomy
    df_tax = pd.read_csv(taxonomy_csv, dtype=str)
    print(f"Loaded taxonomy: {len(df_tax)} pages, levels: {level_cols}", flush=True)

    # Load descriptions if available
    descriptions = None
    if include_descriptions and pages_csv:
        pages_csv = Path(pages_csv)
        df_pages = pd.read_csv(pages_csv, usecols=["page_title", "description"], dtype=str)
        descriptions = dict(zip(df_pages["page_title"], df_pages["description"].fillna("")))
        print(f"Loaded descriptions for {len(descriptions)} pages", flush=True)

    # API key
    api_key = read_openrouter_key()

    # Checkpoint: resume from existing results
    out_csv = out_dir / "llm_cluster_verification.csv"
    done_keys = set()
    results = []
    if out_csv.exists():
        df_done = pd.read_csv(out_csv, dtype=str)
        results = df_done.to_dict("records")
        done_keys = {(r["level"], r["cluster_id"]) for r in results}
        print(f"Resuming: {len(done_keys)} clusters already verified", flush=True)

    # Count total clusters
    total_clusters = 0
    for lvl in level_cols:
        if lvl in df_tax.columns:
            total_clusters += df_tax[lvl].nunique()

    i = 0
    for lvl in level_cols:
        if lvl not in df_tax.columns:
            print(f"WARNING: level '{lvl}' not in taxonomy CSV, skipping", flush=True)
            continue

        clusters = df_tax.groupby(lvl)["page"].apply(list).to_dict()
        print(f"\n--- {lvl}: {len(clusters)} clusters ---", flush=True)

        for cid, pages in sorted(clusters.items(), key=lambda x: -len(x[1])):
            i += 1
            if (lvl, cid) in done_keys:
                continue

            page_list_str, sample_note = _build_page_list(
                pages, descriptions, max_pages_per_prompt, sample_large
            )

            prompt = LLM_VERIFY_USER_PROMPT_TEMPLATE.format(
                level=lvl,
                cluster_id=cid,
                n_pages=len(pages),
                sample_note=sample_note,
                page_list=page_list_str,
            )

            # Call LLM with reasoning mode + retries
            response = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = _call_llm_reasoning(
                        api_key=api_key,
                        model=model,
                        prompt=prompt,
                        system_prompt=LLM_VERIFY_SYSTEM_PROMPT,
                        max_tokens=1024,
                        timeout_s=120,
                    )
                    break
                except Exception as e:
                    print(f"  Retry {attempt+1}/{MAX_RETRIES}: {e}", flush=True)
                    time.sleep(2 ** attempt)

            if response is None:
                parsed = {"coherence": None, "label": "", "misplaced": [], "reasoning": "API error"}
            else:
                parsed = _parse_llm_response(response)

            row = {
                "level": lvl,
                "cluster_id": cid,
                "n_pages": len(pages),
                "coherence": parsed.get("coherence"),
                "label": parsed.get("label", ""),
                "n_misplaced": len(parsed.get("misplaced", [])),
                "misplaced": json.dumps(parsed.get("misplaced", [])),
                "reasoning": parsed.get("reasoning", ""),
            }
            results.append(row)

            status = f"coherence={row['coherence']}  misplaced={row['n_misplaced']}"
            print(
                f"  [{i}/{total_clusters}] {lvl} | {cid[:60]:60s} | n={len(pages):4d} | {status}",
                flush=True,
            )

            # Checkpoint every 10 clusters
            if len(results) % 10 == 0:
                pd.DataFrame(results).to_csv(out_csv, index=False)

    # Final save
    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv, index=False)

    # Summary
    print(f"\n{'='*80}", flush=True)
    print("VERIFICATION SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    for lvl in level_cols:
        sub = df_out[df_out["level"] == lvl]
        if len(sub) == 0:
            continue
        scores = pd.to_numeric(sub["coherence"], errors="coerce")
        n_misplaced = pd.to_numeric(sub["n_misplaced"], errors="coerce")
        print(f"\n{lvl} ({len(sub)} clusters):", flush=True)
        print(f"  Mean coherence: {scores.mean():.2f}", flush=True)
        print(f"  Distribution: {dict(scores.value_counts().sort_index())}", flush=True)
        print(f"  Clusters with misplaced pages: {(n_misplaced > 0).sum()}", flush=True)
        print(f"  Total misplaced pages flagged: {n_misplaced.sum():.0f}", flush=True)

    # Write summary JSON
    summary = {}
    for lvl in level_cols:
        sub = df_out[df_out["level"] == lvl]
        if len(sub) == 0:
            continue
        scores = pd.to_numeric(sub["coherence"], errors="coerce")
        summary[lvl] = {
            "n_clusters": len(sub),
            "mean_coherence": round(float(scores.mean()), 2),
            "coherence_distribution": {
                str(k): int(v) for k, v in scores.value_counts().sort_index().items()
            },
            "n_with_misplaced": int((pd.to_numeric(sub["n_misplaced"], errors="coerce") > 0).sum()),
            "pct_good": round(float((scores >= 4).mean() * 100), 1),
        }
    summary_path = out_dir / "llm_cluster_verification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nResults → {out_csv}", flush=True)
    print(f"Summary → {summary_path}", flush=True)
    return out_csv


def summarize_llm_verification(
    out_dir: Path,
    model: str = LLM_VERIFY_DEFAULT_MODEL,
) -> Path | None:
    """Generate an LLM-written quality summary from verification results.

    Reads ``llm_cluster_verification_summary.json`` and the raw CSV, then
    asks the LLM for a 2-3 sentence overall assessment of taxonomy quality.
    Saves to ``llm_quality_summary.txt``.
    """
    out_dir = Path(out_dir)
    summary_path = out_dir / "llm_cluster_verification_summary.json"
    csv_path = out_dir / "llm_cluster_verification.csv"

    if not summary_path.exists():
        print("[llm-summary] No verification summary found, skipping", flush=True)
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # Build context about worst clusters
    worst_info = ""
    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str)
        df["coherence"] = pd.to_numeric(df["coherence"], errors="coerce")
        worst = df.nsmallest(5, "coherence")
        if len(worst):
            lines = []
            for _, r in worst.iterrows():
                lines.append(
                    f"  - {r.get('level','?')} | {str(r.get('cluster_id',''))[:50]} | "
                    f"coherence={r.get('coherence','?')} | n_pages={r.get('n_pages','?')} | "
                    f"misplaced={r.get('n_misplaced','0')}"
                )
            worst_info = "\n\nWorst 5 clusters:\n" + "\n".join(lines)

    # Format summary stats
    stats_lines = []
    for lvl, data in summary.items():
        stats_lines.append(
            f"  {lvl}: {data['n_clusters']} clusters, "
            f"mean coherence = {data['mean_coherence']}, "
            f"% good (>=4) = {data['pct_good']}%, "
            f"clusters with misplaced pages = {data['n_with_misplaced']}"
        )

    prompt = (
        "You are evaluating a technology taxonomy built from Wikipedia pages.\n"
        "Below are the LLM verification results for each hierarchy level.\n"
        "Coherence is rated 1-5 (5=best). 'Good' means coherence >= 4.\n\n"
        "Per-level statistics:\n" + "\n".join(stats_lines) +
        worst_info +
        "\n\nWrite a 2-3 sentence summary assessing the overall quality of this taxonomy. "
        "Comment on strengths and weaknesses. Be specific about which levels are "
        "well-structured and which have issues."
    )

    api_key = read_openrouter_key()
    try:
        response = _call_llm_reasoning(
            api_key=api_key,
            model=model,
            prompt=prompt,
            system_prompt="You are a research assistant evaluating taxonomy quality.",
            max_tokens=512,
            timeout_s=60,
        )
    except Exception as e:
        print(f"[llm-summary] LLM call failed: {e}", flush=True)
        return None

    if response:
        txt_path = out_dir / "llm_quality_summary.txt"
        txt_path.write_text(response, encoding="utf-8")
        print(f"[llm-summary] Quality summary → {txt_path}", flush=True)
        return txt_path
    return None


# ---------------------------------------------------------------------------
# Logical consistency validation (LLM-based)
# ---------------------------------------------------------------------------

# JSON schemas for structured LLM output in each check
_SCHEMA_TECH_VERIFY = {
    "name": "TechVerify",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string"},
            "corrected_technologies": {
                "type": "array", "items": {"type": "string"},
            },
            "reason": {"type": "string"},
        },
        "required": ["verdict", "corrected_technologies", "reason"],
        "additionalProperties": False,
    },
}

_SCHEMA_SUBSUMPTION = {
    "name": "Subsumption",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "valid": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["valid", "reason"],
        "additionalProperties": False,
    },
}

_SCHEMA_SIBLING = {
    "name": "SiblingExclusivity",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "overlaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "b": {"type": "string"},
                        "overlapping_technologies": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["a", "b", "overlapping_technologies", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["overlaps"],
        "additionalProperties": False,
    },
}

_SCHEMA_REMEDIATION = {
    "name": "Remediation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "move_to": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["action", "move_to", "reason"],
        "additionalProperties": False,
    },
}

_SCHEMA_PERIPHERAL = {
    "name": "PeripheralReview",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "assign_to": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["action", "assign_to", "reason"],
        "additionalProperties": False,
    },
}


def _build_hierarchy(df_assign: pd.DataFrame) -> dict:
    """Build a hierarchy dict from page assignments.

    Handles collapsed branches (where ``L1 == L2`` after
    ``collapse_single_child``).  When a branch is collapsed the node is
    registered at the *coarsest* level it appears in (L1) and is marked
    as a leaf — no self-referential L2 child is created.

    Returns
    -------
    dict with keys:
      - "nodes": {cluster_id: {"level": str, "parent": str|None,
        "pages": list[str], "children": list[str], "is_leaf": bool}}
      - "levels": {"root": [...], "L1": [...], "L2": [...]}
    """
    nodes: Dict[str, dict] = {}
    level_cols = ["root", "L1", "L2"]

    # Identify collapsed branches: rows where L1 == L2
    collapsed_ids: Set[str] = set()
    core = df_assign[df_assign["root"] != "Peripheral"].copy()
    for _, row in core.iterrows():
        l1, l2 = str(row["L1"]), str(row["L2"])
        if l1 == l2 and "Peripheral" not in l1:
            collapsed_ids.add(l1)

    # Collect pages per cluster at each level
    for lvl in level_cols:
        for cid, grp in df_assign.groupby(lvl):
            cid = str(cid)
            if cid == "Peripheral" or cid == "nan":
                continue
            if "Peripheral" in cid:
                continue
            # Skip L2 registration for collapsed branches (already at L1)
            if lvl == "L2" and cid in collapsed_ids:
                continue
            if cid not in nodes:
                nodes[cid] = {
                    "level": lvl, "parent": None,
                    "pages": [], "children": [], "is_leaf": False,
                }
            nodes[cid]["pages"] = grp["page"].tolist()

    # Determine parent-child relationships
    for _, row in core.drop_duplicates(subset=["L1"]).iterrows():
        l1, root = str(row["L1"]), str(row["root"])
        if l1 in nodes and root in nodes:
            nodes[l1]["parent"] = root
            if l1 not in nodes[root]["children"]:
                nodes[root]["children"].append(l1)

    for _, row in core.drop_duplicates(subset=["L2"]).iterrows():
        l2, l1 = str(row["L2"]), str(row["L1"])
        # Skip collapsed branches — l2 == l1, no real child
        if l2 == l1:
            continue
        if l2 in nodes and l1 in nodes:
            nodes[l2]["parent"] = l1
            if l2 not in nodes[l1]["children"]:
                nodes[l1]["children"].append(l2)

    # Mark leaf nodes (no children)
    for cid, node in nodes.items():
        node["is_leaf"] = len(node["children"]) == 0

    levels = {
        lvl: [cid for cid, n in nodes.items() if n["level"] == lvl]
        for lvl in level_cols
    }
    return {"nodes": nodes, "levels": levels}


def _load_labels(run_dir: Path, label_suffix: str = "") -> Dict[str, dict]:
    """Load cluster labels from per-level label CSVs.

    Looks for ``labels__{level}__{suffix}.csv`` files produced by
    :func:`label_taxonomy_contextual`.

    Parameters
    ----------
    run_dir : Path
        Directory containing the label CSV files.
    label_suffix : str
        Backend suffix, e.g. ``"claude"`` or ``"openrouter"``.

    Returns
    -------
    {cluster_id: {"label": str, "technologies": list[str], "description": str}}
    """
    labels: Dict[str, dict] = {}

    for lvl in ["root", "L1", "L2"]:
        label_file = run_dir / f"labels__{lvl}__{label_suffix}.csv"
        if not label_file.exists():
            continue
        df = pd.read_csv(label_file, dtype=str).fillna("")
        for _, row in df.iterrows():
            cid = str(row["cluster_id"])
            lbl = str(row.get("label", ""))
            if cid == "nan" or "Peripheral" in cid:
                continue
            # Parse technologies (pipe-separated in CSV)
            techs_raw = str(row.get("technologies", ""))
            technologies = [t.strip() for t in techs_raw.split("|") if t.strip()] if techs_raw else []
            description = str(row.get("description", ""))
            labels[cid] = {
                "label": lbl,
                "technologies": technologies,
                "description": description,
            }

    return labels


def _label_cluster_llm(
    chat_fn: ChatFn, pages: List[str], max_sample: int = 30,
) -> dict:
    """Ask LLM to label a cluster from its page titles.

    Returns {"label": str, "scope": str}.
    """
    import random
    rng = random.Random(42)
    sample = pages if len(pages) <= max_sample else rng.sample(pages, max_sample)
    page_list = "\n".join(f"- {p}" for p in sorted(sample))

    prompt = (
        f"These are Wikipedia pages in one technology cluster ({len(pages)} total, "
        f"showing {len(sample)}):\n\n{page_list}\n\n"
        "Respond with JSON: {\"label\": \"<1-4 word thematic label>\", "
        "\"scope\": \"<one sentence defining what this cluster covers>\"}"
    )
    system = "You are a technology taxonomy expert. Be concise and precise."

    try:
        text = chat_fn(prompt, system, 256)
    except Exception:
        return {"label": "UNKNOWN", "scope": ""}

    parsed = _parse_llm_response(text)
    return {
        "label": parsed.get("label", "UNKNOWN"),
        "scope": parsed.get("scope", ""),
    }


# ------------------------------------------------------------------
# Intra-check checkpoint utility
# ------------------------------------------------------------------

def _iter_with_checkpoints(items, ckpt_dict, ckpt_key, ckpt_path, save_every=30):
    """Wrap a list to auto-skip completed items and save progress.

    The consuming loop MUST append its result to ``ckpt_dict[ckpt_key]``
    after each yield.  Only suitable for loops with a 1:1 mapping between
    items and results (Checks 0, 1, 4).
    """
    if ckpt_key not in ckpt_dict:
        ckpt_dict[ckpt_key] = []

    already_done = len(ckpt_dict[ckpt_key])
    items_to_do = items[already_done:]

    for i, item in enumerate(items_to_do, start=1):
        yield item

        if i % save_every == 0:
            ckpt_path.write_text(
                json.dumps(ckpt_dict, ensure_ascii=False), encoding="utf-8",
            )
            print(f"\n  [AUTO-SAVE] {ckpt_key}: Checkpointed at "
                  f"{already_done + i} items.", flush=True)

    if items_to_do:
        ckpt_path.write_text(
            json.dumps(ckpt_dict, ensure_ascii=False), encoding="utf-8",
        )


# ------------------------------------------------------------------
# Check 0: Technology verification
# ------------------------------------------------------------------

def _verify_technologies(
    chat_fn: ChatFn,
    hierarchy: dict,
    labels: Dict[str, dict],
    page_distances: Optional[pd.DataFrame] = None,
    max_pages: int = 50,
    ckpt: Optional[dict] = None,
    ckpt_path: Optional[Path] = None,
) -> List[dict]:
    """Verify that LLM-extracted technologies per cluster are accurate.

    Two paths depending on node type:

    - **Leaf nodes** (L2, or collapsed L1): sends page titles ranked by
      distance to cluster centroid (most prototypical first, up to
      *max_pages*).
    - **Internal nodes** (L1/root with children): sends children's
      labels + technologies and asks whether the parent accurately
      summarizes them.

    Parameters
    ----------
    page_distances : DataFrame, optional
        From ``page_distances_digital.csv`` with columns like
        ``page, L2, L2_dist_to_own, ...``.  Used to rank pages by
        proximity to centroid.  Falls back to arbitrary ordering if None.
    max_pages : int
        Maximum page titles to include for leaf nodes (default 50).
    ckpt : dict, optional
        Checkpoint dictionary for intra-check resuming.
    ckpt_path : Path, optional
        Path to save checkpoint JSON.

    Returns list of per-cluster verdicts.
    """
    nodes = hierarchy["nodes"]
    ckpt_key = "tech_verification"

    # Pre-build page-distance lookup: {(level, cluster_id)} -> sorted page list
    _ranked_pages: Dict[tuple, List[str]] = {}
    if page_distances is not None:
        for lvl in ["root", "L1", "L2"]:
            dist_col = f"{lvl}_dist_to_own"
            if dist_col not in page_distances.columns or lvl not in page_distances.columns:
                continue
            df_lvl = page_distances.dropna(subset=[dist_col]).copy()
            df_lvl[dist_col] = pd.to_numeric(df_lvl[dist_col], errors="coerce")
            for cid, grp in df_lvl.groupby(lvl):
                ranked = grp.sort_values(dist_col)["page"].tolist()
                _ranked_pages[(lvl, str(cid))] = ranked

    items = [
        (cid, node) for cid, node in nodes.items()
        if cid in labels and labels[cid].get("technologies")
    ]

    already_done = len(ckpt.get(ckpt_key, [])) if ckpt is not None else 0
    print(f"  [tech-verify] Checking {len(items)} clusters "
          f"({already_done} already done)...", flush=True)

    if ckpt is not None and ckpt_path is not None:
        iterator = _iter_with_checkpoints(items, ckpt, ckpt_key, ckpt_path)
        pbar = tqdm(iterator, desc="TechVerify", unit="cluster", ncols=100,
                    total=len(items), initial=already_done)
    else:
        ckpt = {ckpt_key: []}
        pbar = tqdm(items, desc="TechVerify", unit="cluster", ncols=100)

    n_fail = 0
    n_attempted = 0
    for cid, node in pbar:
        n_attempted += 1
        lbl = labels[cid]
        techs = lbl["technologies"]
        desc = lbl["description"]

        if node["is_leaf"]:
            # --- Leaf: verify technologies against page titles ---
            ranked = _ranked_pages.get((node["level"], cid))
            if ranked:
                sample_pages = ranked[:max_pages]
            else:
                sample_pages = node["pages"][:max_pages]
            page_list = ", ".join(sample_pages)

            content_block = (
                f"Sample pages ({len(sample_pages)} of {len(node['pages'])}, "
                f"ranked by centrality): {page_list}"
            )
            question = (
                "Do the extracted technologies accurately represent this "
                "cluster's content as shown by the pages?"
            )
            examples = (
                "Examples:\n"
                "- Cluster \"Optical Character Recognition\", technologies "
                "[\"OCR\", \"Document processing\", \"Intelligent character recognition\"], "
                "pages: Forms processing, GOCR, Magnetic ink character recognition, "
                "Optical character recognition, ReCAPTCHA\n"
                "  -> {\"verdict\": \"correct\", \"corrected_technologies\": [], "
                "\"reason\": \"All technologies directly match the page content.\"}\n\n"
                "- Cluster \"Semiconductor Manufacturing & Devices\", technologies "
                "[\"Photolithography\", \"Wafer fabrication\", \"Field-effect transistors\"], "
                "pages: Chemical vapor deposition, Wafer bonding, Anodic bonding, "
                "Virtual metrology, Substrate mapping\n"
                "  -> {\"verdict\": \"partial\", \"corrected_technologies\": "
                "[\"Photolithography\", \"Wafer fabrication\", \"FET devices\", "
                "\"Semiconductor bonding & metrology\"], "
                "\"reason\": \"Technologies cover main themes but miss fabrication "
                "process steps like bonding and metrology.\"}\n\n"
                "- Cluster with technologies [\"Audio compression\", \"Signal encoding\"], "
                "pages: DNA computing, Chemical computer, Biological computing, "
                "Peptide computing\n"
                "  -> {\"verdict\": \"incorrect\", \"corrected_technologies\": "
                "[\"DNA computing\", \"Biological computing\", \"Unconventional computing\"], "
                "\"reason\": \"Technologies describe a completely different domain "
                "than the actual pages.\"}"
            )
        else:
            # --- Internal: verify technologies against children ---
            child_lines = []
            for child_id in node["children"]:
                if child_id in labels:
                    c = labels[child_id]
                    c_techs = ", ".join(c.get("technologies", [])) or "N/A"
                    child_lines.append(
                        f"- \"{c['label']}\": [{c_techs}]"
                    )
            content_block = (
                f"Children ({len(child_lines)}):\n" + "\n".join(child_lines)
            )
            question = (
                "Do the parent's technologies accurately summarize the scope "
                "of its children's technologies?"
            )
            examples = (
                "Examples:\n"
                "- Parent \"AI & Machine Learning\" [Machine learning, Computer vision], "
                "children: \"OCR\" [OCR, Document processing], "
                "\"Speech Processing\" [ASR, Text-to-speech], "
                "\"Biometric Recognition\" [Face recognition, Fingerprint ID]\n"
                "  -> {\"verdict\": \"correct\", \"corrected_technologies\": [], "
                "\"reason\": \"Parent technologies cover the shared ML/vision domain "
                "of all children.\"}\n\n"
                "- Parent \"Signal Processing\" [Audio codecs, Video compression], "
                "children: \"Speech Recognition\" [ASR, NLU], "
                "\"Computer Vision\" [Image segmentation, Object detection]\n"
                "  -> {\"verdict\": \"incorrect\", \"corrected_technologies\": "
                "[\"AI & Machine Learning\", \"Pattern recognition\"], "
                "\"reason\": \"Children are ML subfields, not signal processing — "
                "parent label mischaracterizes the domain.\"}"
            )

        prompt = (
            f"Cluster: \"{lbl['label']}\"\n"
            f"Extracted technologies: {techs}\n"
            f"Description: {desc}\n"
            f"{content_block}\n\n"
            f"{question}\n\n"
            f"{examples}\n\n"
            "Respond with JSON:\n"
            "{\n"
            "  \"verdict\": \"correct\" | \"partial\" | \"incorrect\",\n"
            "  \"corrected_technologies\": [\"<corrected technology 1>\", ...],\n"
            "  \"reason\": \"<brief explanation>\"\n"
            "}\n"
            "Use \"correct\" if technologies match well, \"partial\" if some are "
            "right but incomplete/imprecise, \"incorrect\" if they misrepresent "
            "the cluster. For \"correct\", leave corrected_technologies empty."
        )
        system = (
            "You are validating a technology taxonomy. Each cluster has "
            "LLM-extracted technologies and a description. Check whether "
            "these accurately capture the cluster's actual content. "
            "Be strict — technologies should be specific "
            "and representative. Return ONLY valid JSON."
        )

        try:
            text = chat_fn(prompt, system, 256, _SCHEMA_TECH_VERIFY)
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"  [WARN] {ckpt_key} call failed "
                      f"({n_fail}/{n_attempted}): {type(e).__name__}: {e}", flush=True)
            if n_attempted >= 10 and n_fail / n_attempted > 0.25:
                if ckpt_path is not None:
                    ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")
                raise RuntimeError(
                    f"[{ckpt_key}] failure rate {n_fail}/{n_attempted} > 25% — "
                    f"aborting. Checkpoint preserved at {ckpt_path}."
                )
            continue

        parsed = _parse_llm_response(text)
        ckpt[ckpt_key].append({
            "cluster_id": cid,
            "level": node["level"],
            "is_leaf": node["is_leaf"],
            "label": lbl["label"],
            "technologies": "|".join(techs),
            "description": desc,
            "verdict": parsed.get("verdict", "unknown"),
            "corrected_technologies": "|".join(parsed.get("corrected_technologies", [])),
            "reason": parsed.get("reason", ""),
        })

    verdicts = Counter(r["verdict"] for r in ckpt[ckpt_key])
    print(f"  [tech-verify] Results: {dict(verdicts)}", flush=True)
    return ckpt[ckpt_key]


def _check_subsumption(
    chat_fn: ChatFn,
    hierarchy: dict,
    labels: Dict[str, dict],
    ckpt: Optional[dict] = None,
    ckpt_path: Optional[Path] = None,
) -> List[dict]:
    """Check parent-child subsumption for all non-collapsed pairs.

    Uses technologies + description from bottom-up labeling.
    Skips collapsed branches (leaf nodes at L1 where L1==L2).

    Returns list of violations.
    """
    nodes = hierarchy["nodes"]
    ckpt_key = "subsumption"
    pairs = []

    for cid, node in nodes.items():
        parent_id = node["parent"]
        if parent_id is None:
            continue
        # Skip if either side is unlabeled
        if parent_id not in labels or cid not in labels:
            continue
        # Skip collapsed branches (parent == child after collapse)
        if parent_id == cid:
            continue
        pairs.append((parent_id, cid))

    already_done = len(ckpt.get(ckpt_key, [])) if ckpt is not None else 0
    print(f"  [subsumption] Checking {len(pairs)} parent-child pairs "
          f"({already_done} already done)...", flush=True)

    if ckpt is not None and ckpt_path is not None:
        iterator = _iter_with_checkpoints(pairs, ckpt, ckpt_key, ckpt_path)
        pbar = tqdm(iterator, desc="Subsumption", unit="pair", ncols=100,
                    total=len(pairs), initial=already_done)
    else:
        ckpt = {ckpt_key: []}
        pbar = tqdm(pairs, desc="Subsumption", unit="pair", ncols=100)

    n_fail = 0
    n_attempted = 0
    for parent_id, child_id in pbar:
        n_attempted += 1
        p_lbl = labels[parent_id]
        c_lbl = labels[child_id]
        p_techs = ", ".join(p_lbl.get("technologies", [])) or "N/A"
        c_techs = ", ".join(c_lbl.get("technologies", [])) or "N/A"
        p_desc = p_lbl.get("description", "")
        c_desc = c_lbl.get("description", "")

        prompt = (
            f"Parent cluster: \"{p_lbl['label']}\"\n"
            f"  Technologies: [{p_techs}]\n"
            f"  Description: {p_desc}\n\n"
            f"Child cluster: \"{c_lbl['label']}\"\n"
            f"  Technologies: [{c_techs}]\n"
            f"  Description: {c_desc}\n\n"
            "Are the child's technologies a logical subset of the parent's domain?\n\n"
            "Examples:\n"
            "- Parent: \"AI & Machine Learning\" [Machine learning, Speech recognition, Computer vision]\n"
            "  Child: \"Optical Character Recognition\" [OCR, Document processing, Intelligent character recognition]\n"
            "  → {\"valid\": true, \"reason\": \"OCR is an application of machine learning and computer vision.\"}\n\n"
            "- Parent: \"Audio/Video Compression & Signal Processing\" [Video codecs, Audio compression, Signal processing]\n"
            "  Child: \"Unconventional Computing\" [DNA computing, Biological computing, Chemical computing]\n"
            "  → {\"valid\": false, \"reason\": \"Unconventional computing paradigms are not a subset of "
            "compression or signal processing — they belong under a computing paradigms branch.\"}\n\n"
            "Respond with JSON: {\"valid\": true/false, \"reason\": \"<brief explanation>\"}"
        )
        system = (
            "You are validating a technology taxonomy. A child should be a narrower "
            "specialization of its parent — its technologies should logically fall "
            "within the parent's broader domain. If the child topic does not "
            "logically belong under the parent, mark it as invalid."
        )

        try:
            text = chat_fn(prompt, system, 256, _SCHEMA_SUBSUMPTION)
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"  [WARN] {ckpt_key} call failed "
                      f"({n_fail}/{n_attempted}): {type(e).__name__}: {e}", flush=True)
            if n_attempted >= 10 and n_fail / n_attempted > 0.25:
                if ckpt_path is not None:
                    ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")
                raise RuntimeError(
                    f"[{ckpt_key}] failure rate {n_fail}/{n_attempted} > 25% — "
                    f"aborting. Checkpoint preserved at {ckpt_path}."
                )
            continue

        parsed = _parse_llm_response(text)
        is_valid = parsed.get("valid", True)
        ckpt[ckpt_key].append({
            "parent_id": parent_id,
            "parent_label": p_lbl["label"],
            "parent_technologies": "|".join(p_lbl.get("technologies", [])),
            "child_id": child_id,
            "child_label": c_lbl["label"],
            "child_technologies": "|".join(c_lbl.get("technologies", [])),
            "valid": is_valid,
            "reason": parsed.get("reason", ""),
        })

    results = ckpt[ckpt_key]
    n_invalid = sum(1 for v in results if not v["valid"])
    print(f"  [subsumption] {n_invalid} violations found out of {len(results)} pairs", flush=True)
    return results


def _check_sibling_exclusivity(
    chat_fn: ChatFn,
    hierarchy: dict,
    labels: Dict[str, dict],
) -> List[dict]:
    """Check sibling exclusivity for all parent nodes.

    Uses technologies + description from bottom-up labeling to detect
    overlapping technology domains between siblings.

    Returns list of overlaps.
    """
    nodes = hierarchy["nodes"]
    overlaps = []

    parents_with_children = [
        (cid, node) for cid, node in nodes.items()
        if len(node["children"]) > 1
    ]

    print(f"  [siblings] Checking {len(parents_with_children)} parent nodes...", flush=True)
    pbar = tqdm(parents_with_children, desc="Siblings", unit="parent", ncols=100)

    n_fail = 0
    n_attempted = 0
    for parent_id, parent_node in pbar:
        children = [c for c in parent_node["children"] if c in labels]
        if len(children) < 2:
            continue
        n_attempted += 1

        sibling_lines = []
        for c in children:
            lbl = labels[c]
            n_pages = len(nodes[c]["pages"]) if c in nodes else 0
            techs = ", ".join(lbl.get("technologies", [])) or "N/A"
            desc = lbl.get("description", "")
            sibling_lines.append(
                f"- \"{lbl['label']}\" ({n_pages} pages)\n"
                f"  Technologies: [{techs}]\n"
                f"  Description: {desc}"
            )

        parent_label = labels.get(parent_id, {}).get("label", parent_id)
        prompt = (
            f"Parent cluster: \"{parent_label}\"\n"
            f"Sibling clusters under this parent:\n"
            + "\n".join(sibling_lines) + "\n\n"
            "Which pairs of siblings have overlapping technology domains?\n\n"
            "Examples:\n"
            "- Under parent \"AI & Machine Learning\":\n"
            "  Sibling A: \"Computer Vision & Biometric Recognition\" [Face recognition, Fingerprint identification, Image processing]\n"
            "  Sibling B: \"Optical Character Recognition\" [OCR, Document processing, Intelligent character recognition]\n"
            "  → {\"overlaps\": []} (both involve image analysis but address distinct tasks — "
            "biometric identification vs text extraction)\n\n"
            "- Under parent \"AI & Machine Learning\":\n"
            "  Sibling A: \"Computer Vision & Biometric Recognition\" [Face recognition, Fingerprint identification, Image processing]\n"
            "  Sibling B: \"Face & Gesture Recognition\" [Facial recognition, Gesture detection, Emotion recognition]\n"
            "  → {\"overlaps\": [{\"a\": \"Computer Vision & Biometric Recognition\", "
            "\"b\": \"Face & Gesture Recognition\", "
            "\"overlapping_technologies\": [\"Facial recognition\"], "
            "\"reason\": \"Face recognition appears in both clusters — one should absorb the other.\"}]}\n\n"
            "Respond with JSON: {\"overlaps\": [{\"a\": \"<label a>\", "
            "\"b\": \"<label b>\", \"overlapping_technologies\": [\"<tech>\", ...], "
            "\"reason\": \"<why they overlap>\"}]} — use an empty list if none overlap."
        )
        system = (
            "You are validating a technology taxonomy. Sibling categories should be "
            "mutually exclusive — they should not cover the same technologies or concepts. "
            "Only flag genuine technology overlaps, not loose thematic connections."
        )

        try:
            text = chat_fn(prompt, system, 512, _SCHEMA_SIBLING)
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"  [WARN] sibling_overlaps call failed "
                      f"({n_fail}/{n_attempted}): {type(e).__name__}: {e}", flush=True)
            if n_attempted >= 10 and n_fail / n_attempted > 0.25:
                raise RuntimeError(
                    f"[sibling_overlaps] failure rate {n_fail}/{n_attempted} > 25% — "
                    f"aborting. (Atomic check — partial progress is not saved.)"
                )
            continue

        parsed = _parse_llm_response(text)
        for ov in parsed.get("overlaps", []):
            # Resolve labels back to cluster IDs
            a_label = ov.get("a", "")
            b_label = ov.get("b", "")
            a_id, b_id = "", ""
            a_techs, b_techs = [], []
            for c in children:
                if labels[c]["label"] == a_label:
                    a_id = c
                    a_techs = labels[c].get("technologies", [])
                if labels[c]["label"] == b_label:
                    b_id = c
                    b_techs = labels[c].get("technologies", [])

            overlaps.append({
                "parent_id": parent_id,
                "parent_label": parent_label,
                "sibling_a_id": a_id,
                "sibling_a_label": a_label,
                "sibling_a_technologies": "|".join(a_techs),
                "sibling_b_id": b_id,
                "sibling_b_label": b_label,
                "sibling_b_technologies": "|".join(b_techs),
                "overlap": True,
                "overlapping_technologies": "|".join(ov.get("overlapping_technologies", [])),
                "reason": ov.get("reason", ""),
            })

    print(f"  [siblings] {len(overlaps)} overlapping pairs found", flush=True)
    return overlaps


def _review_inconsistencies(
    chat_fn: ChatFn,
    hierarchy: dict,
    labels: Dict[str, dict],
    subsumption_violations: List[dict],
    sibling_overlaps: List[dict],
    ckpt: Optional[dict] = None,
    ckpt_path: Optional[Path] = None,
) -> List[dict]:
    """For each inconsistency, ask LLM: remove, merge, or move? Where?

    Uses technology lists in root-branch context for better recommendations.

    Returns a list of remediation recommendations.
    """
    nodes = hierarchy["nodes"]
    recommendations = []

    # Only review violations (not valid pairs)
    items = []
    for v in subsumption_violations:
        if not v.get("valid", True):
            items.append(("subsumption", v))
    for v in sibling_overlaps:
        items.append(("sibling_overlap", v))

    if not items:
        return []

    print(f"  [remediation] Reviewing {len(items)} inconsistencies...", flush=True)
    pbar = tqdm(items, desc="Remediation", unit="issue", ncols=100)

    # Build taxonomy summary with technologies for context
    root_lines = []
    for cid in hierarchy["levels"].get("root", []):
        lbl = labels.get(cid, {})
        label = lbl.get("label", cid)
        techs = ", ".join(lbl.get("technologies", [])) or "N/A"
        root_lines.append(f"  - \"{label}\": [{techs}]")
    taxonomy_context = "Root branches:\n" + "\n".join(root_lines)

    n_fail = 0
    n_attempted = 0
    for issue_type, detail in pbar:
        n_attempted += 1
        if issue_type == "subsumption":
            p_techs = detail.get("parent_technologies", "N/A")
            c_techs = detail.get("child_technologies", "N/A")
            cluster_id = detail.get("child_id", "")
            cluster_label = detail.get("child_label", "")
            desc = (
                f"SUBSUMPTION VIOLATION: Child \"{detail['child_label']}\" "
                f"(technologies: [{c_techs}]) does not fit under parent "
                f"\"{detail['parent_label']}\" (technologies: [{p_techs}]). "
                f"Reason: {detail.get('reason', 'N/A')}"
            )

        elif issue_type == "sibling_overlap":
            a_techs = detail.get("sibling_a_technologies", "N/A")
            b_techs = detail.get("sibling_b_technologies", "N/A")
            overlap_techs = detail.get("overlapping_technologies", "N/A")
            cluster_id = detail.get("sibling_a_id", "")
            cluster_label = detail.get("sibling_a_label", "")
            desc = (
                f"SIBLING OVERLAP: Under parent \"{detail['parent_label']}\", "
                f"siblings \"{detail['sibling_a_label']}\" (technologies: [{a_techs}]) "
                f"and \"{detail['sibling_b_label']}\" (technologies: [{b_techs}]) overlap. "
                f"Overlapping technologies: [{overlap_techs}]. "
                f"Reason: {detail.get('reason', 'N/A')}"
            )

        prompt = (
            f"{taxonomy_context}\n\n"
            f"Issue: {desc}\n\n"
            "What should be done to fix this? Respond with JSON:\n"
            "{\n"
            "  \"action\": \"remove\" | \"merge\" | \"move\" | \"keep\",\n"
            "  \"move_to\": \"<destination cluster/branch if action=move, else empty string>\",\n"
            "  \"reason\": \"<brief explanation>\"\n"
            "}"
        )
        system = (
            "You are a technology taxonomy expert reviewing inconsistencies. "
            "Recommend the minimum change needed to fix each issue. "
            "Prefer 'move' over 'remove' when possible — only recommend 'remove' "
            "if the cluster genuinely doesn't belong in the taxonomy at all. "
            "Return ONLY valid JSON."
        )

        try:
            text = chat_fn(prompt, system, 256, _SCHEMA_REMEDIATION)
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"  [WARN] remediation call failed "
                      f"({n_fail}/{n_attempted}): {type(e).__name__}: {e}", flush=True)
            if n_attempted >= 10 and n_fail / n_attempted > 0.25:
                if ckpt is not None and ckpt_path is not None:
                    ckpt["remediation"] = recommendations
                    ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")
                raise RuntimeError(
                    f"[remediation] failure rate {n_fail}/{n_attempted} > 25% — "
                    f"aborting. Checkpoint preserved at {ckpt_path}."
                )
            continue

        parsed = _parse_llm_response(text)
        recommendations.append({
            "issue_type": issue_type,
            "source_check": "check1_subsumption" if issue_type == "subsumption" else "check2_sibling",
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "action": parsed.get("action", "unknown"),
            "move_to": parsed.get("move_to", ""),
            "reason": parsed.get("reason", ""),
        })

    print(f"  [remediation] {len(recommendations)} recommendations generated", flush=True)
    return recommendations


def _build_taxonomy_context(
    nodes: dict,
    labels: Dict[str, dict],
    root_id: Optional[str] = None,
) -> str:
    """Build taxonomy context showing siblings + their children.

    If *root_id* is None (global peripheral), shows the full taxonomy
    (all roots → L1s → L2s).  If *root_id* is given (local peripheral),
    shows all L1 siblings under that root and their L2 children.
    """
    lines: List[str] = []

    if root_id is None:
        # Full taxonomy: iterate all roots
        roots = [cid for cid, n in nodes.items()
                 if n.get("level") == "root" and "Peripheral" not in cid
                 and cid in labels]
    else:
        # Scoped to one root: its L1 children
        if root_id in nodes:
            roots = [root_id]
        else:
            return ""

    for rid in roots:
        if rid not in labels:
            continue
        rlbl = labels[rid]
        rtechs = ", ".join(rlbl.get("technologies", [])) or "N/A"
        lines.append(f"  {rlbl['label']} -- technologies: [{rtechs}]")

        children_l1 = [c for c in nodes.get(rid, {}).get("children", [])
                       if c in labels and "Peripheral" not in c]
        for l1id in children_l1:
            l1lbl = labels[l1id]
            l1techs = ", ".join(l1lbl.get("technologies", [])) or "N/A"
            lines.append(
                f"    {l1lbl['label']} -- technologies: [{l1techs}]"
            )
            children_l2 = [c for c in nodes.get(l1id, {}).get("children", [])
                           if c in labels and "Peripheral" not in c]
            for l2id in children_l2:
                l2lbl = labels[l2id]
                l2techs = ", ".join(l2lbl.get("technologies", [])) or "N/A"
                lines.append(
                    f"      {l2lbl['label']} -- technologies: [{l2techs}]"
                )

    return "\n".join(lines)


def _review_peripherals(
    chat_fn: ChatFn,
    df_assign: pd.DataFrame,
    hierarchy: dict,
    labels: Dict[str, dict],
    ckpt: Optional[dict] = None,
    ckpt_path: Optional[Path] = None,
) -> List[dict]:
    """Review peripheral pages: assign to a core cluster, keep peripheral, or remove?

    Handles two types of peripheral pages:
    - **Global**: rows where the deepest level is "Peripheral" or "Peripheral_*"
      (from ``attach_peripheral_to_clusters``).  Gets the **full taxonomy**
      as context (all roots → L1s → L2s).
    - **Local**: rows where a level contains "|Peripheral"
      (from ``posthoc_resplit`` singleton→peripheral conversion).  Gets
      **all siblings + their children** under the same root.

    Returns a list of peripheral reviews.
    """
    ckpt_key = "peripherals"
    level_cols = ["root", "L1", "L2"]
    nodes = hierarchy["nodes"]

    # Collect peripheral pages with their context
    periph_pages = []
    for _, row in df_assign.iterrows():
        page = str(row["page"])
        periph_type = None
        parent_cluster = None
        root_cluster = None

        # Check for global peripheral (L2 == "Peripheral" or starts with "Peripheral_")
        l2_val = str(row.get("L2", ""))
        if l2_val == "Peripheral" or l2_val.startswith("Peripheral_"):
            periph_type = "global"
            l1_val = str(row.get("L1", ""))
            if "Peripheral" not in l1_val:
                parent_cluster = l1_val
            root_cluster = str(row.get("root", ""))

        # Check for local peripheral ("|Peripheral" in any level)
        if periph_type is None:
            for lvl in level_cols:
                val = str(row.get(lvl, ""))
                if "|Peripheral" in val:
                    periph_type = "local"
                    parent_cluster = val.split("|Peripheral")[0]
                    root_cluster = str(row.get("root", ""))
                    break

        if periph_type is not None:
            periph_pages.append({
                "page": page,
                "peripheral_type": periph_type,
                "parent_cluster": parent_cluster,
                "root_cluster": root_cluster,
            })

    if not periph_pages:
        return []

    already_done = len(ckpt.get(ckpt_key, [])) if ckpt is not None else 0
    print(f"  [peripheral] Reviewing {len(periph_pages)} peripheral pages "
          f"(global={sum(1 for p in periph_pages if p['peripheral_type']=='global')}, "
          f"local={sum(1 for p in periph_pages if p['peripheral_type']=='local')})"
          f" ({already_done} already done)...",
          flush=True)

    if ckpt is not None and ckpt_path is not None:
        iterator = _iter_with_checkpoints(periph_pages, ckpt, ckpt_key, ckpt_path)
        pbar = tqdm(iterator, desc="Peripheral", unit="page", ncols=100,
                    total=len(periph_pages), initial=already_done)
    else:
        ckpt = {ckpt_key: []}
        pbar = tqdm(periph_pages, desc="Peripheral", unit="page", ncols=100)

    # Pre-build taxonomy context (full for global, per-root for local)
    _full_ctx: Optional[str] = None
    _root_ctx_cache: Dict[str, str] = {}

    n_fail = 0
    n_attempted = 0
    for item in pbar:
        n_attempted += 1
        page = item["page"]
        parent_id = item["parent_cluster"]
        root_id = item["root_cluster"]
        parent_label = labels.get(parent_id, {}).get("label", parent_id) if parent_id else "N/A"

        if item["peripheral_type"] == "global":
            # Full taxonomy context
            if _full_ctx is None:
                _full_ctx = _build_taxonomy_context(nodes, labels, root_id=None)
            taxonomy_context = f"Full taxonomy:\n{_full_ctx}"
        else:
            # Root-scoped context (siblings + children)
            if root_id not in _root_ctx_cache:
                _root_ctx_cache[root_id] = _build_taxonomy_context(
                    nodes, labels, root_id=root_id,
                )
            taxonomy_context = (
                f"Clusters under root \"{labels.get(root_id, {}).get('label', root_id)}\":\n"
                + _root_ctx_cache[root_id]
            )

        prompt = (
            f"Peripheral page: \"{page}\" ({item['peripheral_type']} peripheral)\n"
            f"Parent cluster: \"{parent_label}\"\n\n"
            f"{taxonomy_context}\n\n"
            "This Wikipedia page is currently peripheral (not assigned to any core "
            "cluster). Should it be:\n"
            "- **assign** to one of the clusters above (specify which label)\n"
            "- **keep_peripheral** (it doesn't fit any cluster well)\n"
            "- **remove** from the taxonomy (not a real technology)\n\n"
            "Respond with JSON:\n"
            "{\n"
            "  \"action\": \"assign\" | \"keep_peripheral\" | \"remove\",\n"
            "  \"assign_to\": \"<cluster label if action=assign, else null>\",\n"
            "  \"reason\": \"<brief explanation>\"\n"
            "}"
        )
        system = (
            "You are a technology taxonomy expert. Peripheral pages are Wikipedia "
            "technology articles that could not be cleanly assigned to a cluster "
            "during taxonomy construction. "
            "If the page clearly fits an existing cluster, recommend assign. "
            "If it's not a real technology, recommend remove. "
            "If it's a valid technology that doesn't fit the existing clusters, "
            "recommend keep_peripheral. Return ONLY valid JSON."
        )

        try:
            text = chat_fn(prompt, system, 256, _SCHEMA_PERIPHERAL)
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"  [WARN] {ckpt_key} call failed "
                      f"({n_fail}/{n_attempted}): {type(e).__name__}: {e}", flush=True)
            if n_attempted >= 10 and n_fail / n_attempted > 0.25:
                if ckpt_path is not None:
                    ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")
                raise RuntimeError(
                    f"[{ckpt_key}] failure rate {n_fail}/{n_attempted} > 25% — "
                    f"aborting. Checkpoint preserved at {ckpt_path}."
                )
            continue

        parsed = _parse_llm_response(text)
        # Resolve assign_to label → cluster ID (search all nodes)
        assign_to_label = parsed.get("assign_to", "")
        nearest_id = ""
        nearest_label = ""
        if assign_to_label:
            nearest_label = assign_to_label
            for cid, lbl in labels.items():
                if lbl.get("label") == assign_to_label and "Peripheral" not in cid:
                    nearest_id = cid
                    break

        ckpt[ckpt_key].append({
            "page": page,
            "peripheral_type": item["peripheral_type"],
            "current_parent": parent_id or "",
            "nearest_cluster_id": nearest_id,
            "nearest_cluster_label": nearest_label,
            "action": parsed.get("action", "unknown"),
            "assign_to": assign_to_label,
            "reason": parsed.get("reason", ""),
        })

    results = ckpt[ckpt_key]
    actions = Counter(r["action"] for r in results)
    print(f"  [peripheral] Results: {dict(actions)}", flush=True)
    return results


def validate_logical_consistency(
    run_dir: Path,
    pages_csv: Path,
    *,
    output_dir: Path | None = None,
    llm_model: str = "google/gemini-3-flash-preview",
    backend: str = "openrouter",
    claude_model: str = "opus",
    label_suffix: str = "claude",
    suffix: str = "",
) -> dict:
    """Run logical consistency validation on a taxonomy.

    Five checks (in order):
    0. Technology verification — are extracted technologies per cluster correct?
    1. Parent-child subsumption — is each child a logical subset of its parent?
    2. Sibling exclusivity — are siblings mutually exclusive?
    3. Remediation — for each issue from 1+2, recommend remove/merge/move/keep
    4. Peripheral review — for each peripheral page, assign/keep/remove?

    Handles collapsed branches (L1==L2 after ``collapse_single_child``).
    Uses technologies + description from bottom-up labeling.

    Supports two backends:
    - ``backend="openrouter"`` — calls OpenRouter API (needs API key)
    - ``backend="claude-code"`` — calls ``claude`` CLI (no API key needed)

    Requires labels to exist (from a prior labeling step). If labels are missing,
    generates them via LLM first.

    Returns a summary dict (also saved as logical_consistency.json + .xlsx).
    """
    run_dir = Path(run_dir)
    pages_csv = Path(pages_csv)

    if output_dir is None:
        output_dir = run_dir / "validation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sfx = f"_{suffix}" if suffix else ""
    out_path = output_dir / f"logical_consistency{sfx}.json"

    print(f"\n{'='*60}")
    print(f"  LOGICAL CONSISTENCY VALIDATION")
    print(f"  run_dir    = {run_dir}")
    print(f"  output_dir = {output_dir}")
    print(f"  backend    = {backend}")
    print(f"  llm_model  = {llm_model}")
    print(f"{'='*60}\n")

    # Build chat function for the chosen backend
    api_key = None
    if backend == "openrouter":
        api_key = read_openrouter_key()
    chat_fn = make_chat_fn(
        backend=backend,
        api_key=api_key,
        model=llm_model,
        user_agent="st4te-taxonomy-logical",
        claude_model=claude_model,
    )

    # ---- Load page assignments ----
    assign_files = list(run_dir.glob("page_assignments_L2*.csv"))
    if not assign_files:
        assign_files = list(run_dir.glob("page_assignments_*.csv"))
    if not assign_files:
        raise SystemExit(f"No page_assignments*.csv found in {run_dir}")
    df_assign = pd.read_csv(assign_files[0], dtype=str)
    df_assign["page"] = df_assign["page"].astype(str)
    print(f"  assignments = {assign_files[0].name} ({len(df_assign)} pages)")

    # ---- Build hierarchy (collapsed-branch aware) ----
    hierarchy = _build_hierarchy(df_assign)
    n_nodes = len(hierarchy["nodes"])
    n_root = len(hierarchy["levels"]["root"])
    n_l1 = len(hierarchy["levels"]["L1"])
    n_l2 = len(hierarchy["levels"]["L2"])
    n_leaf = sum(1 for n in hierarchy["nodes"].values() if n["is_leaf"])
    print(f"  hierarchy: {n_root} root, {n_l1} L1, {n_l2} L2 "
          f"({n_nodes} nodes, {n_leaf} leaves)")

    # ---- Load or generate labels ----
    labels = _load_labels(run_dir, label_suffix=label_suffix)
    missing = [cid for cid in hierarchy["nodes"] if cid not in labels]

    if missing:
        print(f"  Labeling {len(missing)} unlabeled clusters via LLM...", flush=True)
        pbar = tqdm(missing, desc="Labeling", unit="cluster", ncols=100)
        for cid in pbar:
            pages = hierarchy["nodes"][cid]["pages"]
            if not pages:
                labels[cid] = {"label": "EMPTY", "technologies": [], "description": ""}
                continue
            result = _label_cluster_llm(chat_fn, pages)
            labels[cid] = {
                "label": result.get("label", "UNKNOWN"),
                "technologies": [],
                "description": result.get("scope", ""),
            }

        # Save labels for reuse
        label_out = output_dir / f"cluster_labels{sfx}.json"
        label_out.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Labels → {label_out}")

    print(f"  {len(labels)} clusters labeled\n")

    # Checkpoint file for partial progress
    ckpt_path = output_dir / f".checkpoint_lc{sfx}.json"
    ckpt = {}
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        print(f"  [CHECKPOINT] resumed from {ckpt_path.name} "
              f"(completed: {list(ckpt.keys())})", flush=True)

    # ---- Check 0: Technology verification ----
    # Load page distances for ranking pages by centroid proximity
    page_dist_files = list(output_dir.glob("page_distances_*.csv"))
    if not page_dist_files:
        page_dist_files = list((run_dir / "validation").glob("page_distances_*.csv"))
    df_page_dist = None
    if page_dist_files:
        df_page_dist = pd.read_csv(page_dist_files[0], dtype={"page": str})
        print(f"  page distances = {page_dist_files[0].name} "
              f"({len(df_page_dist)} pages)")
    else:
        print("  [WARN] No page_distances CSV found — using arbitrary page order")

    print(f"\n{'='*60}")
    print(f"  CHECK 0: Technology verification")
    print(f"{'='*60}\n")
    tech_verification = _verify_technologies(
        chat_fn, hierarchy, labels, page_distances=df_page_dist,
        ckpt=ckpt, ckpt_path=ckpt_path,
    )

    # ---- Check 1: Parent-child subsumption ----
    print(f"\n{'='*60}")
    print(f"  CHECK 1: Parent-child subsumption")
    print(f"{'='*60}\n")
    subsumption_violations = _check_subsumption(
        chat_fn, hierarchy, labels,
        ckpt=ckpt, ckpt_path=ckpt_path,
    )

    # ---- Check 2: Sibling exclusivity ----
    if "sibling_overlaps" in ckpt:
        sibling_overlaps = ckpt["sibling_overlaps"]
        print(f"  [SKIP] siblings — loaded {len(sibling_overlaps)} overlaps from checkpoint", flush=True)
    else:
        print(f"\n{'='*60}")
        print(f"  CHECK 2: Sibling exclusivity")
        print(f"{'='*60}\n")
        sibling_overlaps = _check_sibling_exclusivity(chat_fn, hierarchy, labels)
        ckpt["sibling_overlaps"] = sibling_overlaps
        ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")
        print(f"  [CHECKPOINT] saved after siblings", flush=True)

    # ---- Check 3: Remediation recommendations ----
    if "remediation" in ckpt:
        remediation = ckpt["remediation"]
        print(f"  [SKIP] remediation — loaded {len(remediation)} from checkpoint", flush=True)
    else:
        print(f"\n{'='*60}")
        print(f"  CHECK 3: Remediation recommendations")
        print(f"{'='*60}\n")
        remediation = _review_inconsistencies(
            chat_fn, hierarchy, labels,
            subsumption_violations, sibling_overlaps,
            ckpt=ckpt, ckpt_path=ckpt_path,
        )
        ckpt["remediation"] = remediation
        ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")
        print(f"  [CHECKPOINT] saved after remediation", flush=True)

    # ---- Check 4: Peripheral review ----
    print(f"\n{'='*60}")
    print(f"  CHECK 4: Peripheral review")
    print(f"{'='*60}\n")
    peripheral_reviews = _review_peripherals(
        chat_fn, df_assign, hierarchy, labels,
        ckpt=ckpt, ckpt_path=ckpt_path,
    )

    # ---- Build report ----
    total_pairs = len(subsumption_violations)
    n_subsumption_invalid = sum(1 for v in subsumption_violations if not v["valid"])
    total_parents = sum(
        1 for n in hierarchy["nodes"].values()
        if len(n["children"]) > 1
    )
    tech_verdicts = Counter(r["verdict"] for r in tech_verification)
    peripheral_actions = Counter(r["action"] for r in peripheral_reviews)

    report = {
        "tech_verification": {
            "total_checked": len(tech_verification),
            "correct": tech_verdicts.get("correct", 0),
            "partial": tech_verdicts.get("partial", 0),
            "incorrect": tech_verdicts.get("incorrect", 0),
            "verification_rate": round(
                (tech_verdicts.get("correct", 0) + tech_verdicts.get("partial", 0))
                / max(len(tech_verification), 1) * 100, 1),
            "details": tech_verification,
        },
        "subsumption": {
            "total_pairs_checked": total_pairs,
            "violations": n_subsumption_invalid,
            "violation_rate": round(n_subsumption_invalid / max(total_pairs, 1) * 100, 1),
            "details": subsumption_violations,
        },
        "sibling_exclusivity": {
            "total_parents_checked": total_parents,
            "overlapping_pairs": len(sibling_overlaps),
            "details": sibling_overlaps,
        },
        "remediation": {
            "total_issues_reviewed": len(remediation),
            "actions": dict(Counter(r["action"] for r in remediation)),
            "details": remediation,
        },
        "peripherals": {
            "total_peripheral": len(peripheral_reviews),
            "actions": dict(peripheral_actions),
            "details": peripheral_reviews,
        },
        "summary": {
            "tech_verification_rate": round(
                (tech_verdicts.get("correct", 0) + tech_verdicts.get("partial", 0))
                / max(len(tech_verification), 1) * 100, 1),
            "subsumption_violations": n_subsumption_invalid,
            "sibling_overlaps": len(sibling_overlaps),
            "total_issues": n_subsumption_invalid + len(sibling_overlaps),
            "peripheral_total": len(peripheral_reviews),
            "peripheral_assign": peripheral_actions.get("assign", 0),
            "peripheral_keep": peripheral_actions.get("keep_peripheral", 0),
            "peripheral_remove": peripheral_actions.get("remove", 0),
        },
        "config": {
            "llm_model": llm_model,
            "run_dir": str(run_dir),
            "backend": backend,
            "label_suffix": label_suffix,
        },
    }

    # ---- Save JSON (checkpoint kept on disk so partial/complete state is always recoverable) ----
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Save Excel (one sheet per check + summary) ----
    xlsx_path = output_dir / f"logical_consistency{sfx}.xlsx"
    _EMPTY = lambda msg: pd.DataFrame({"result": [msg]})

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Sheet 0 — tech_verification
        if tech_verification:
            pd.DataFrame(tech_verification)[
                ["cluster_id", "level", "is_leaf", "label", "technologies",
                 "description", "verdict", "corrected_technologies", "reason"]
            ].to_excel(writer, sheet_name="tech_verification", index=False)
        else:
            _EMPTY("No clusters to verify").to_excel(
                writer, sheet_name="tech_verification", index=False)

        # Sheet 1 — subsumption
        if subsumption_violations:
            pd.DataFrame(subsumption_violations)[
                ["parent_id", "parent_label", "parent_technologies",
                 "child_id", "child_label", "child_technologies",
                 "valid", "reason"]
            ].to_excel(writer, sheet_name="subsumption", index=False)
        else:
            _EMPTY("No pairs to check").to_excel(
                writer, sheet_name="subsumption", index=False)

        # Sheet 2 — sibling_exclusivity
        if sibling_overlaps:
            pd.DataFrame(sibling_overlaps)[
                ["parent_id", "parent_label",
                 "sibling_a_id", "sibling_a_label", "sibling_a_technologies",
                 "sibling_b_id", "sibling_b_label", "sibling_b_technologies",
                 "overlap", "overlapping_technologies", "reason"]
            ].to_excel(writer, sheet_name="sibling_exclusivity", index=False)
        else:
            _EMPTY("No sibling overlaps found").to_excel(
                writer, sheet_name="sibling_exclusivity", index=False)

        # Sheet 3 — remediation
        if remediation:
            pd.DataFrame(remediation)[
                ["issue_type", "source_check", "cluster_id", "cluster_label",
                 "action", "move_to", "reason"]
            ].to_excel(writer, sheet_name="remediation", index=False)
        else:
            _EMPTY("No remediation needed").to_excel(
                writer, sheet_name="remediation", index=False)

        # Sheet 4 — peripherals
        if peripheral_reviews:
            pd.DataFrame(peripheral_reviews)[
                ["page", "peripheral_type", "current_parent",
                 "nearest_cluster_id", "nearest_cluster_label",
                 "action", "assign_to", "reason"]
            ].to_excel(writer, sheet_name="peripherals", index=False)
        else:
            _EMPTY("No peripheral pages found").to_excel(
                writer, sheet_name="peripherals", index=False)

        # Sheet 5 — summary
        summary_rows = [
            {"metric": "Tech verification rate (%)",
             "value": report["summary"]["tech_verification_rate"],
             "total": len(tech_verification)},
            {"metric": "Subsumption violations",
             "value": n_subsumption_invalid,
             "total": total_pairs},
            {"metric": "Sibling overlaps",
             "value": len(sibling_overlaps),
             "total": total_parents},
            {"metric": "Total issues",
             "value": report["summary"]["total_issues"],
             "total": ""},
            {"metric": "Peripheral (total)",
             "value": len(peripheral_reviews), "total": ""},
            {"metric": "Peripheral: assign",
             "value": peripheral_actions.get("assign", 0), "total": ""},
            {"metric": "Peripheral: keep",
             "value": peripheral_actions.get("keep_peripheral", 0), "total": ""},
            {"metric": "Peripheral: remove",
             "value": peripheral_actions.get("remove", 0), "total": ""},
        ]
        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name="summary", index=False)

    print(f"\n{'='*60}")
    print(f"  LOGICAL CONSISTENCY SUMMARY")
    print(f"{'='*60}")
    print(f"  Tech verification:      {tech_verdicts.get('correct', 0)}/{len(tech_verification)} correct "
          f"({report['summary']['tech_verification_rate']}%)")
    print(f"  Subsumption violations: {n_subsumption_invalid} / {total_pairs}")
    print(f"  Sibling overlaps:       {len(sibling_overlaps)}")
    print(f"  Total issues:           {report['summary']['total_issues']}")
    print(f"  Peripherals:            {len(peripheral_reviews)} "
          f"(assign={peripheral_actions.get('assign', 0)}, "
          f"keep={peripheral_actions.get('keep_peripheral', 0)}, "
          f"remove={peripheral_actions.get('remove', 0)})")
    print(f"  Report → {out_path}")
    print(f"  Excel  → {xlsx_path}")

    return report


# ---------------------------------------------------------------------------
# Unified validation orchestrator
# ---------------------------------------------------------------------------

def run_unified_validation(
    run_dir: Path,
    pages_csv: Path,
    mode: str,
    *,
    output_dir: Path | None = None,
    embed_model: str = "AI-Growth-Lab/PatentSBERTa",
    # Tier flags
    internal: bool = True,
    external: bool = True,
    llm_verify: bool = False,
    logical: bool = False,
    backend: str = "openrouter",
    claude_model: str = "opus",
    # External options
    threshold: float = 0.5,
    citation_csv: Path | None = None,
    cpc_digits: str = "2,3,4",
    skip_wiki_category: bool = False,
    skip_modularity: bool = False,
    skip_coverage: bool = False,
    min_component_size: int = 10,
    expert_csv: Path | None = None,
    llm_model: str = "google/gemini-3-flash-preview",
) -> dict:
    """Run unified post-cluster validation.

    Three tiers, each toggleable:

    **Internal** (``internal=True``):
      - Page distances: per-page margin (own centroid vs closest-other)
      - Per-cluster margins: within-sim vs nearest-other-sim
      - Per-cluster coherence: mean cos(page, centroid)

    **External** (``external=True``):
      - CPC HHI + top-k shares (3-digit, 4-digit)
      - Citation intra-rate + modularity Q
      - Wiki category HHI
      - RCA / Balassa index
      - Coverage: patent-wiki matching (7 analyses)
      - Expert agreement (if ``expert_csv`` provided)

    **LLM** (``llm_verify=True``):
      - LLM coherence scoring per cluster

    All tiers generate plots and summary JSONs.

    Returns
    -------
    dict : unified metrics dict (also written to validation_summary_{mode}.json)
    """
    run_dir = Path(run_dir)
    pages_csv = Path(pages_csv)

    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    if not pages_csv.exists():
        raise SystemExit(f"Pages CSV not found: {pages_csv}")

    if output_dir is None:
        output_dir = run_dir / "validation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  UNIFIED VALIDATION")
    print(f"  run_dir    = {run_dir}")
    print(f"  pages_csv  = {pages_csv}")
    print(f"  mode       = {mode}")
    print(f"  output_dir = {output_dir}")
    print(f"  tiers      = internal={internal}, external={external}, llm={llm_verify}")
    print(f"{'='*60}\n")

    # ---- Auto-detect taxonomy and page-assignment CSVs ----
    taxonomy_files = list(run_dir.glob("taxonomy_long_labeled__*.csv"))
    if not taxonomy_files:
        taxonomy_files = list(run_dir.glob("taxonomy_long_*.csv"))
    if not taxonomy_files:
        raise SystemExit(f"No taxonomy_long*.csv found in {run_dir}")
    taxonomy_csv = taxonomy_files[0]

    assign_files = list(run_dir.glob("page_assignments_L2*.csv"))
    if not assign_files:
        assign_files = list(run_dir.glob("page_assignments_*.csv"))
    if not assign_files:
        raise SystemExit(f"No page_assignments*.csv found in {run_dir}")
    assign_csv = assign_files[0]

    print(f"  taxonomy   = {taxonomy_csv.name}")
    print(f"  assignments= {assign_csv.name}\n")

    level_cols = ["root", "L1", "L2"]
    unified: dict = {"mode": mode, "run_dir": str(run_dir), "levels": {}}
    margins_by_level: dict = {}
    cpc_by_level: dict = {}
    citation_by_level: dict = {}
    wikicat_by_level: dict = {}

    # ==================================================================
    # INTERNAL VALIDATION
    # ==================================================================
    if internal:
        print(f"{'='*60}")
        print(f"  INTERNAL VALIDATION")
        print(f"{'='*60}\n")

        # ---- Part 1: Page distances ----
        print(f"  Part 1: Page distances")
        page_dist_out = output_dir / f"page_distances_{mode}.csv"
        run_page_distances(
            taxonomy_csv=taxonomy_csv,
            pages_csv=pages_csv,
            taxonomy_page_col="page",
            taxonomy_cols="root,L1,L2",
            pages_title_col="page_title",
            pages_desc_col="description",
            embed_model=embed_model,
            exclude_peripheral=True,
            out=str(page_dist_out),
            mode=mode,
        )
        print(f"  Page distances → {page_dist_out}\n")

        # Read page distances for summary
        df_page_dist = pd.read_csv(page_dist_out)
        for lvl in level_cols:
            margin_col = f"{lvl}_margin"
            if margin_col in df_page_dist.columns:
                vals = pd.to_numeric(df_page_dist[margin_col], errors="coerce").dropna()
                if len(vals) > 0:
                    unified.setdefault("levels", {}).setdefault(lvl, {})["page_distance"] = {
                        "mean_margin": round(float(vals.mean()), 4),
                        "median_margin": round(float(vals.median()), 4),
                        "pct_positive_margin": round(float((vals > 0).mean() * 100), 1),
                    }

        # ---- Part 2: Cluster margins + coherence ----
        print(f"  Part 2: Cluster margins + coherence")
        df_assign = pd.read_csv(assign_csv)
        df_assign["page"] = df_assign["page"].astype(str)
        df_input = pd.read_csv(pages_csv)
        df_input["page_title"] = df_input["page_title"].astype(str)

        desc_texts = [
            clean_description_text(x)
            for x in df_input["description"].fillna("").astype(str).tolist()
        ]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        emb_all = load_or_build_embeddings(
            desc_texts=desc_texts,
            input_path=pages_csv,
            out_dir=CACHE_ROOT,
            embed_model=embed_model,
            device=device,
            desc_col="description",
        )

        title_to_idx = {t: i for i, t in enumerate(df_input["page_title"].tolist())}
        pages = df_assign["page"].tolist()
        valid_mask = np.array([p in title_to_idx for p in pages])
        emb_indices = np.array([title_to_idx.get(p, 0) for p in pages])
        emb_pages = emb_all[emb_indices]

        n_valid = int(valid_mask.sum())
        print(f"  {n_valid}/{len(pages)} pages matched to embeddings")

        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("CLUSTER MARGIN ANALYSIS (global)")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Run:   {run_dir}")
        summary_lines.append(f"Pages: {n_valid} matched / {len(pages)} total")
        summary_lines.append("")

        for level in level_cols:
            if level not in df_assign.columns:
                continue
            labels = df_assign[level].astype(str).values
            mask = (
                valid_mask
                & (labels != "Peripheral")
                & (labels != "nan")
                & (labels != "None")
            )
            if mask.sum() < 2:
                continue

            labels_f = labels[mask]
            emb_f = emb_pages[mask]
            unique = np.unique(labels_f)
            if len(unique) < 2:
                continue

            margins_df = compute_cluster_margins(emb_f, labels_f, None)
            margins_df = margins_df.sort_values("margin")
            margins_by_level[level] = margins_df
            csv_path = output_dir / f"cluster_margins_{mode}_{level}.csv"
            margins_df.to_csv(csv_path, index=False)

            # Semantic coherence
            coh_df = _per_cluster_coherence(emb_f, labels_f)
            coh_csv = output_dir / f"cluster_coherence_{mode}_{level}.csv"
            coh_df.to_csv(coh_csv, index=False)
            print(f"  {level} coherence: mean={coh_df['coherence'].mean():.4f}")
            n_with_margin = int(margins_df["margin"].notna().sum())
            n_neg = int((margins_df["margin"] < 0).sum())
            median_m = float(margins_df["margin"].median())
            mean_m = float(margins_df["margin"].mean())

            summary_lines.append(f"--- {level} ({len(unique)} clusters) ---")
            summary_lines.append(f"  Negative margin: {n_neg}/{n_with_margin} ({100*n_neg/max(n_with_margin,1):.1f}%)")
            summary_lines.append(f"  Mean margin:  {mean_m:.4f}")
            summary_lines.append(f"  Median margin: {median_m:.4f}")
            summary_lines.append(f"  Worst 5:")
            for _, r in margins_df.dropna(subset=["margin"]).head(5).iterrows():
                cid = str(r["cluster_id"])[:50]
                summary_lines.append(
                    f"    {cid:<52} margin={r['margin']:+.4f}  "
                    f"within={r['within_sim']:.4f}  nearest={r['nearest_other_sim']:.4f}"
                )
            summary_lines.append("")

            # Summary dict
            vals = margins_df["margin"].dropna()
            unified.setdefault("levels", {}).setdefault(level, {})["margin"] = {
                "mean": round(float(vals.mean()), 4) if len(vals) else None,
                "median": round(float(vals.median()), 4) if len(vals) else None,
                "pct_negative": round(100 * float((vals < 0).sum()) / max(len(vals), 1), 1),
            }
            unified["levels"][level]["coherence"] = {
                "mean": round(float(coh_df["coherence"].mean()), 4),
                "median": round(float(coh_df["coherence"].median()), 4),
            }
            unified["levels"][level]["n_clusters"] = len(unique)

        summary_path = output_dir / f"cluster_margins_{mode}_summary.txt"
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        print(f"  Margin summary → {summary_path}")

        # Margin bar chart
        if margins_by_level:
            plot_cluster_quality(
                margins_by_level, mode, output_dir,
                cpc_by_level=cpc_by_level,
                citation_by_level=citation_by_level,
                wikicat_by_level=wikicat_by_level,
            )

    # ==================================================================
    # EXTERNAL VALIDATION
    # ==================================================================
    if external:
        print(f"\n{'='*60}")
        print(f"  EXTERNAL VALIDATION")
        print(f"{'='*60}\n")

        # Build the citation graph ONCE and share it across Part 3 (external) and
        # Part 4 (coverage): both previously rebuilt it from (mode, CORPUS_ROOT,
        # citation_csv) — a heavy full-corpus + full-CSV scan done twice per run.
        _cit_path = Path(citation_csv) if citation_csv else None
        _shared_cit_graph = None
        if _cit_path is not None and _cit_path.exists():
            _fam, _a2f = load_family_to_applns(mode, CORPUS_ROOT)
            _shared_cit_graph = load_citation_graph(_cit_path, set(_a2f.keys()), _a2f, _fam)
            del _fam, _a2f
            gc.collect()

        # ---- Part 3: CPC + Citation + Wiki Categories ----
        print(f"  Part 3: CPC + Citation + Wiki Categories")
        _, cpc_by_level, citation_by_level, wikicat_by_level = (
            run_cluster_external_validation(
                run_dir=run_dir,
                pages_csv=pages_csv,
                mode=mode,
                output_dir=output_dir,
                embed_model=embed_model,
                threshold=threshold,
                citation_csv=Path(citation_csv) if citation_csv else None,
                cpc_digits=cpc_digits,
                skip_wiki_category=skip_wiki_category,
                skip_modularity=skip_modularity,
                citation_graph=_shared_cit_graph,
            )
        )

        # Read external validation summary into unified dict
        ext_json_path = output_dir / f"cluster_external_validation_{mode}_summary.json"
        if ext_json_path.exists():
            ext = json.loads(ext_json_path.read_text(encoding="utf-8"))
            for lvl_name, ext_level in ext.get("levels", {}).items():
                for key in ["cpc_2digit", "cpc_3digit", "cpc_4digit", "citation", "wiki_category"]:
                    if key in ext_level:
                        unified.setdefault("levels", {}).setdefault(lvl_name, {})[key] = ext_level[key]

        # Re-plot with full data if internal was also run
        if internal and margins_by_level:
            plot_cluster_quality(
                margins_by_level, mode, output_dir,
                cpc_by_level=cpc_by_level,
                citation_by_level=citation_by_level,
                wikicat_by_level=wikicat_by_level,
            )

        # ---- Part 4: Coverage (core pages only) ----
        if not skip_coverage:
            print(f"\n  Part 4: Post-cluster coverage (core pages only)")

            # Load assignments if not already loaded
            if "df_assign" not in dir():
                df_assign = pd.read_csv(
                    list(run_dir.glob("page_assignments_L2*.csv"))[0]
                    if list(run_dir.glob("page_assignments_L2*.csv"))
                    else list(run_dir.glob("page_assignments_*.csv"))[0]
                )
                df_assign["page"] = df_assign["page"].astype(str)
            if "df_input" not in dir():
                df_input = pd.read_csv(pages_csv)
                df_input["page_title"] = df_input["page_title"].astype(str)

            l2_col = "L2" if "L2" in df_assign.columns else None
            if l2_col is None:
                for col in reversed(level_cols):
                    if col in df_assign.columns:
                        l2_col = col
                        break

            if l2_col:
                l2_labels = df_assign[l2_col].astype(str)
                is_peripheral = (
                    l2_labels.str.contains("Peripheral", case=False, na=False)
                    | (l2_labels == "nan")
                    | (l2_labels == "None")
                )
                core_pages = set(df_assign.loc[~is_peripheral, "page"].tolist())
                df_core = df_input[df_input["page_title"].isin(core_pages)].copy()
                print(f"  {len(core_pages)} core pages, {int(is_peripheral.sum())} peripheral excluded")

                core_csv = output_dir / f"core_pages_filtered_{mode}.csv"
                df_core.to_csv(core_csv, index=False)

                run_coverage_analysis(
                    wiki_csv=core_csv,
                    mode=mode,
                    output_dir=output_dir,
                    embed_model=embed_model,
                    threshold=threshold,
                    citation_csv=Path(citation_csv) if citation_csv else None,
                    min_component_size=min_component_size,
                    citation_graph=_shared_cit_graph,
                )
                print(f"  Coverage output → {output_dir}")

                # Read coverage summary
                cov_json = output_dir / "coverage_summary.json"
                if cov_json.exists():
                    cov = json.loads(cov_json.read_text(encoding="utf-8"))
                    for lvl in level_cols:
                        unified.setdefault("levels", {}).setdefault(lvl, {})["coverage"] = {
                            k: cov[k] for k in ["patent_coverage_pct", "mean_similarity"]
                            if k in cov
                        }
        else:
            print("  Skipping coverage (--skip-coverage)")

        # ---- Part 5: Expert agreement ----
        if expert_csv and Path(expert_csv).exists():
            print(f"\n  Part 5: Expert gold-standard agreement")
            _expert_csv = Path(expert_csv)

            df_expert = pd.read_csv(_expert_csv)
            df_tax = pd.read_csv(taxonomy_csv)

            tax_page_col = "page" if "page" in df_tax.columns else "page_title"
            tax_map = {}
            for _, row in df_tax.iterrows():
                p = str(row[tax_page_col])
                tax_map[p] = {
                    "assigned_root": str(row.get("root", "")),
                    "assigned_L1": str(row.get("L1", "")),
                    "assigned_L2": str(row.get("L2", "")),
                }

            for col in ["assigned_root", "assigned_L1", "assigned_L2"]:
                df_expert[col] = pd.array([""] * len(df_expert), dtype="string")

            matched = 0
            for idx, row in df_expert.iterrows():
                p = str(row["page_title"])
                if p in tax_map:
                    matched += 1
                    for col in ["assigned_root", "assigned_L1", "assigned_L2"]:
                        df_expert.at[idx, col] = tax_map[p][col]

            print(f"  {matched}/{len(df_expert)} validation pages found in taxonomy")

            expert_summary = []
            expert_summary.append("=" * 80)
            expert_summary.append("EXPERT GOLD-STANDARD AGREEMENT")
            expert_summary.append("=" * 80)

            for level, assigned_col, expert_col in [
                ("root", "assigned_root", "expert_root_label"),
                ("L1", "assigned_L1", "expert_L1_label"),
                ("L2", "assigned_L2", "expert_L2_label"),
            ]:
                if expert_col not in df_expert.columns or assigned_col not in df_expert.columns:
                    continue

                has_both = (
                    df_expert[expert_col].notna()
                    & df_expert[assigned_col].notna()
                    & (df_expert[assigned_col] != "")
                    & (df_expert[assigned_col] != "nan")
                )
                df_eval = df_expert[has_both].copy()

                assigned = df_eval[assigned_col].astype(str)
                expert_labels = df_eval[expert_col].astype(str)

                cluster_to_expert = {}
                for cid in assigned.unique():
                    m = assigned == cid
                    el = expert_labels[m]
                    if len(el) == 0:
                        continue
                    majority = el.value_counts().idxmax()
                    majority_count = el.value_counts().iloc[0]
                    cluster_to_expert[cid] = {
                        "majority_label": majority,
                        "majority_pct": round(100 * majority_count / len(el), 1),
                        "n_pages": len(el),
                    }

                correct = total = 0
                for _, r in df_eval.iterrows():
                    cid = str(r[assigned_col])
                    exp = str(r[expert_col])
                    if cid in cluster_to_expert:
                        total += 1
                        if exp == cluster_to_expert[cid]["majority_label"]:
                            correct += 1

                agreement_pct = round(100 * correct / max(total, 1), 1)
                total_pages = sum(v["n_pages"] for v in cluster_to_expert.values())
                purity = round(
                    sum(v["majority_pct"] * v["n_pages"] / 100 for v in cluster_to_expert.values())
                    / max(total_pages, 1) * 100, 1
                )

                expert_summary.append(f"--- {level} ---")
                expert_summary.append(f"  Agreement: {correct}/{total} ({agreement_pct}%)")
                expert_summary.append(f"  Purity: {purity}%")
                expert_summary.append("")

                unified.setdefault("levels", {}).setdefault(level, {})["expert"] = {
                    "agreement_pct": agreement_pct,
                    "purity": purity,
                }

            expert_out = output_dir / f"expert_agreement_{mode}.txt"
            expert_out.write_text("\n".join(expert_summary), encoding="utf-8")
            df_expert.to_csv(output_dir / f"expert_validation_joined_{mode}.csv", index=False)

            # Expert confusion heatmaps: removed (plot helper deleted)
        else:
            if expert_csv:
                print(f"  Expert CSV not found: {expert_csv}, skipping")

    # ==================================================================
    # LLM VALIDATION
    # ==================================================================
    if llm_verify:
        print(f"\n{'='*60}")
        print(f"  LLM CLUSTER VERIFICATION")
        print(f"{'='*60}\n")

        llm_out_csv = verify_clusters(
            taxonomy_csv=taxonomy_csv,
            pages_csv=pages_csv,
            out_dir=output_dir,
            levels="root,L1,L2",
            model=llm_model,
        )

        llm_df = pd.read_csv(llm_out_csv, dtype=str)
        for level in level_cols:
            pass
        # Read LLM summary
        llm_summary_path = output_dir / "llm_cluster_verification_summary.json"
        if llm_summary_path.exists():
            unified["llm"] = json.loads(llm_summary_path.read_text(encoding="utf-8"))

        # Generate LLM quality summary text
        summarize_llm_verification(output_dir, model=llm_model)

    # ==================================================================
    # LOGICAL CONSISTENCY
    # ==================================================================
    if logical:
        print(f"\n{'='*60}")
        print(f"  LOGICAL CONSISTENCY VALIDATION")
        print(f"{'='*60}\n")

        lc_report = validate_logical_consistency(
            run_dir=run_dir,
            pages_csv=pages_csv,
            output_dir=output_dir,
            embed_model=embed_model,
            llm_model=llm_model,
            backend=backend,
            claude_model=claude_model,
            suffix=backend,
        )
        unified["logical_consistency"] = lc_report.get("summary", {})

    # ==================================================================
    # Consolidated dashboard
    # ==================================================================
    _plot_validation_summary_dashboard(unified, mode, output_dir)

    # ==================================================================
    # Write unified summary JSON
    # ==================================================================
    unified_path = output_dir / f"validation_summary_{mode}.json"
    unified_path.write_text(json.dumps(unified, indent=2), encoding="utf-8")
    print(f"\n[validate] Unified summary → {unified_path}")
    print(f"[validate] All done → {output_dir}")

    return unified


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m p2_taxonomy.analysis_cluster_quality",
        description="Per-cluster margin analysis + post-clustering coverage for a completed clustering run",
    )
    ap.add_argument(
        "--run-dir", required=True,
        help="Directory containing page_assignments CSV from a clustering run",
    )
    ap.add_argument(
        "--input", default=str(DEFAULT_INPUT),
        help="Input CSV used for clustering (for descriptions + embeddings)",
    )
    ap.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <run-dir>/cluster_quality)",
    )
    ap.add_argument(
        "--embed-model", default="AI-Growth-Lab/PatentSBERTa",
        help="SentenceTransformer model (must match clustering run)",
    )
    ap.add_argument(
        "--mode", choices=["digital", "green"], default=None,
        help="Patent subset for post-clustering coverage analysis (required unless --skip-coverage)",
    )
    ap.add_argument(
        "--skip-coverage", action="store_true",
        help="Skip post-clustering patent-wiki coverage analysis",
    )
    args = ap.parse_args()

    run_cluster_quality(
        run_dir=Path(args.run_dir),
        input_csv=Path(args.input),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        embed_model=args.embed_model,
        mode=args.mode,
        skip_coverage=args.skip_coverage,
    )


if __name__ == "__main__":
    main()
