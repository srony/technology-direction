#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
taxonomy_adaptive_pipeline.py

Adaptive multilevel Wikipedia taxonomy builder (top-down Leiden):

Core pipeline
  - Build an undirected wiki-link graph from `linked_wiki_titles_json`
  - Compute sentence-transformer embeddings from `description` ONLY
    (embeddings are used only for semantic edge reweighting + diagnostics, NOT for clustering)
  - Reweight existing wiki-link edges with a hybrid of:
      * semantic similarity (cosine of description embeddings)
      * observed link strength (edge-count weight)
    then prune edges below `tau_edge`
  - Core/peripheral split:
      * isolates + nodes in components smaller than `min_component_size` are treated as peripheral
      * clustering is performed on the core graph only
  - Top-down hierarchical clustering (tree by construction):
      root -> L1 -> L2
    using Leiden (RBConfigurationVertexPartition) with adaptive gamma search per parent group:
      * targets K in [Kmin, Kmax]
      * respects min_size and split_if_gt rules (do not split small groups)
      * stability check via ARI across multiple random seeds for the chosen gamma
  - Diagnostics per group and per level:
      * ARI stability
      * semantic coherence (mean cos(item, cluster centroid))
      * within/between centroid similarity and within-minus-between gap

Peripheral handling
  - After taxonomy is built on the core graph, peripheral nodes are attached to the deepest level
    cluster via weighted connections to core nodes (or marked Peripheral_* if no strong core links)

Labeling (optional)
  - Optional OpenRouter cluster labeling using parent-context prompts
  - Writes per-level label maps and labeled taxonomy outputs

Plotting (optional, Plotly HTML)
  - Cluster size distributions by level
  - Heatmap: given root_technology vs discovered root (if `root_technology` exists)
  - Interactive network plots per level (core-only; hover shows endpoints + weight)
  - Cluster-level L2 network (nodes = L2 clusters; size = #pages; color = root)

Outputs
  - page assignments with raw cluster IDs:
      page_assignments__L2_adaptive_description.csv
      taxonomy_long__L2_adaptive_description.csv
      taxonomy_grouped_pages__L2_adaptive_description.csv
  - (optional) labeled variants:
      taxonomy_long_labeled__L2_adaptive_description.csv
      page_assignments_labeled__L2_adaptive_description.csv
      labels__<level>__adaptive.csv
  - run diagnostics:
      run_diagnostics_adaptive_description__<run_id>.json
      run_summary_by_level_adaptive_description.csv

Dependencies
  pip install pandas numpy tqdm plotly scikit-learn python-igraph leidenalg requests torch sentence-transformers
"""


import argparse
import json
import math
import os
import re
import sys
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Ensure project root is on path when run as a standalone script
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

import igraph as ig
import leidenalg

from sklearn.metrics import adjusted_rand_score

import torch
from sentence_transformers import SentenceTransformer

from p2_taxonomy.llm import label_clusters_from_df
from p2_taxonomy.visualisations import (
    viz_cluster_size_distributions,
    viz_heatmap_given_vs_discovered,
    viz_network_by_level,
    viz_L2_nodes_colored_by_root,
    viz_taxonomy_sunburst_cluster,
)
from shared.utils import ensure_dir, parse_csv_ints, parse_links_json
from shared.llm import read_openrouter_key
# Canonical whitespace-normaliser; re-exported so the ~7 modules that do
# `from p2_taxonomy.analysis_cluster import clean_description_text` keep working.
from shared.cleaning import clean_description_text

CACHE_ROOT = Path(r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\taxonomy\wiki\cache\wiki_embedding_cache")
# ----------------------------
# tqdm helper (Windows-friendly)
# ----------------------------
def tqdm_force(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("mininterval", 0.5)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("ascii", True)
    return tqdm(*args, **kwargs)


# ----------------------------
# Paths / defaults
# ----------------------------
def default_out_dir_from_input(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}"


def get_visualisations_dir() -> Path:
    home = Path.home()
    return home / "Dropbox" / "Work" / "UNU-MERIT" / "ST4TE" / "Task2_1" / "visualisations" / "taxonomy"


# ----------------------------
# Embeddings + semantic edge reweighting
# ----------------------------


def embed_texts(
    texts: List[str],
    model_name: str,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)

def _emb_cache_paths(out_dir: Path, embed_model: str, cache_tag: str = "") -> Tuple[Path, Path]:
    # Use short model name: "AI-Growth-Lab/PatentSBERTa" -> "PatentSBERTa"
    model_short = embed_model.split("/")[-1]
    tag = f"_{cache_tag}" if cache_tag else ""
    emb_path = out_dir / f"wiki{tag}_{model_short}.npz"
    meta_path = out_dir / f"wiki{tag}_{model_short}.meta.json"
    return emb_path, meta_path


def load_or_build_embeddings(
    *,
    desc_texts: List[str],
    input_path: Path,
    out_dir: Path,
    embed_model: str,
    device: str,
    desc_col: str,
    cache_tag: str = "",
) -> np.ndarray:
    emb_path, meta_path = _emb_cache_paths(out_dir, embed_model, cache_tag=cache_tag)
    n = len(desc_texts)

    if emb_path.exists():
        try:
            arr = np.load(emb_path)["emb"]
            if arr.shape[0] == n:
                print(f"[EMBED] Loaded cached embeddings: {emb_path} ({n} texts)", flush=True)
                return arr.astype(np.float32)
            else:
                print(f"[EMBED] Cache length mismatch ({arr.shape[0]} vs {n}), recomputing...", flush=True)
        except Exception as e:
            print(f"[EMBED] Failed to load cache ({e}), recomputing...", flush=True)

    # Build + save
    print(f"[EMBED] Computing embeddings for {n} texts...", flush=True)
    emb = embed_texts(desc_texts, model_name=embed_model, device=device)

    ensure_dir(out_dir)
    np.savez_compressed(emb_path, emb=emb.astype(np.float32))

    meta = {
        "embed_model": embed_model,
        "desc_col": desc_col,
        "n_texts": n,
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[EMBED] Saved embeddings cache: {emb_path}", flush=True)
    return emb



# ----------------------------
# Graph building from wiki links
# ----------------------------
def build_graph_from_wikilinks(
    df: pd.DataFrame,
    title_col: str,
    links_col: str,
    max_links_per_page: int = 500,
    symmetrize: bool = True,
) -> Tuple[ig.Graph, Dict[str, int], List[str]]:
    titles_list = df[title_col].astype(str).tolist()
    title_to_idx = {t: i for i, t in enumerate(titles_list)}
    title_set = set(title_to_idx.keys())

    edge_w: Dict[Tuple[int, int], float] = {}

    for _, row in tqdm_force(df.iterrows(), total=len(df), desc="Building wiki-link edges", unit="page"):
        src = str(row[title_col])
        src_i = title_to_idx.get(src)
        if src_i is None:
            continue

        links = parse_links_json(row.get(links_col, ""))
        if max_links_per_page and len(links) > max_links_per_page:
            links = links[:max_links_per_page]

        for dst in links:
            if dst not in title_set:
                continue
            dst_i = title_to_idx.get(dst)
            if dst_i is None or dst_i == src_i:
                continue

            u, v = src_i, dst_i
            if symmetrize:
                u, v = (u, v) if u < v else (v, u)

            edge_w[(u, v)] = edge_w.get((u, v), 0.0) + 1.0

    edges = list(edge_w.keys())
    weights = list(edge_w.values())

    g = ig.Graph(n=len(titles_list), edges=edges, directed=False)
    g.es["edge_count_weight"] = weights
    g.es["weight"] = weights  # default initial weight
    return g, title_to_idx, titles_list


def reweight_edges_semantic(
    g: ig.Graph,
    desc_emb: np.ndarray,
    semantic_weight: float,
    edge_weight: float,
    tau_edge: float,
    hub_penalty: bool = True,
    prune_pct: float = 0.0,
) -> Dict[str, Any]:
    """
    For each existing edge (u,v):
      sim = cosine(desc_u, desc_v) = dot since normalized
      w_sem = max(0, sim)
      w_edge = normalized edge count weight
      w = semantic_weight*w_sem + edge_weight*w_edge
      optionally divide by hub penalty
    Then prune edges with w < tau_edge
    """
    assert g.vcount() == desc_emb.shape[0]

    ecw = np.array(g.es["edge_count_weight"], dtype=float) if g.ecount() else np.array([])
    if len(ecw) == 0:
        return {"n_edges_before": 0, "n_edges_after": 0, "tau_edge": tau_edge}

    ecw_norm = ecw / (ecw.max() if ecw.max() > 0 else 1.0)

    deg = np.array(g.degree(), dtype=float)
    hub_div = np.ones(g.ecount(), dtype=float)

    if hub_penalty:
        for i, e in enumerate(g.es):
            u, v = e.tuple
            hub_div[i] = (math.log1p(deg[u]) * math.log1p(deg[v]))
            if hub_div[i] <= 0:
                hub_div[i] = 1.0

    weights_new = np.zeros(g.ecount(), dtype=float)

    for i, e in enumerate(tqdm_force(g.es, total=g.ecount(), desc="Reweighting edges (semantic)", unit="edge")):
        u, v = e.tuple
        sim = float(np.dot(desc_emb[u], desc_emb[v]))
        w_sem = max(0.0, sim)
        w_edge = float(ecw_norm[i])
        w = float(semantic_weight) * w_sem + float(edge_weight) * w_edge
        w = w / hub_div[i]
        weights_new[i] = w

    g.es["weight"] = weights_new.tolist()

    edges_before = g.ecount()
    if prune_pct > 0:
        threshold = float(np.percentile(weights_new, prune_pct))
    else:
        threshold = float(tau_edge)
    keep_mask = weights_new >= threshold
    drop_eids = np.where(~keep_mask)[0].tolist()
    if drop_eids:
        g.delete_edges(drop_eids)

    stats = {
        "n_edges_before": int(edges_before),
        "n_edges_after": int(g.ecount()),
        "tau_edge": float(tau_edge),
        "prune_pct": float(prune_pct),
        "effective_threshold": float(threshold),
        "semantic_weight": float(semantic_weight),
        "edge_weight": float(edge_weight),
        "hub_penalty": bool(hub_penalty),
        "edge_weight_stats_after": {
            "mean": float(np.mean(g.es["weight"])) if g.ecount() else 0.0,
            "median": float(np.median(g.es["weight"])) if g.ecount() else 0.0,
            "p90": float(np.percentile(g.es["weight"], 90)) if g.ecount() else 0.0,
            "max": float(np.max(g.es["weight"])) if g.ecount() else 0.0,
        },
    }
    return stats


# ----------------------------
# Core/peripheral pruning
# ----------------------------
def connected_components_vids(g: ig.Graph) -> List[List[int]]:
    comps = g.components(mode="WEAK")
    return [list(c) for c in comps]


def split_core_peripheral(
    g: ig.Graph,
    min_component_size: int,
    remove_isolates: bool = True,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    n = g.vcount()
    deg = np.array(g.degree(), dtype=int)

    isolates = set(np.where(deg == 0)[0].tolist())
    comps = connected_components_vids(g)

    tiny = set()
    for comp in comps:
        if len(comp) < int(min_component_size):
            tiny.update(comp)

    peripheral = set()
    if remove_isolates:
        peripheral.update(isolates)
    peripheral.update(tiny)

    core = [i for i in range(n) if i not in peripheral]
    peripheral = sorted(list(peripheral))

    stats = {
        "n_nodes": int(n),
        "n_edges": int(g.ecount()),
        "n_isolates": int(len(isolates)),
        "n_tiny_component_nodes": int(len(tiny)),
        "n_core": int(len(core)),
        "n_peripheral": int(len(peripheral)),
        "min_component_size": int(min_component_size),
    }
    return core, peripheral, stats


# ----------------------------
# Leiden helpers
# ----------------------------
def leiden_membership(g: ig.Graph, gamma: float, iterations: int, seed: int) -> List[int]:
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"] if "weight" in g.es.attributes() else None,
        resolution_parameter=float(gamma),
        n_iterations=int(iterations),
        seed=int(seed),
    )
    return list(part.membership)


def cluster_sizes(mem: List[int]) -> np.ndarray:
    _, counts = np.unique(np.array(mem, dtype=int), return_counts=True)
    return counts


def stability_ari(g: ig.Graph, gamma: float, iterations: int, seeds: List[int]) -> float:
    mems = []
    for s in seeds:
        mems.append(leiden_membership(g, gamma, iterations, s))
    aris = []
    for i in range(len(mems)):
        for j in range(i + 1, len(mems)):
            aris.append(adjusted_rand_score(mems[i], mems[j]))
    return float(np.mean(aris)) if aris else 1.0


@dataclass
class GammaSearchResult:
    gamma: float
    k: int
    small_frac: float
    min_size: int
    sizes: List[int]
    ari_stability: float
    margin: float = 0.0


def adaptive_gamma_search(
    g: ig.Graph,
    kmin: int,
    kmax: int,
    min_size: int,
    iterations: int,
    seed: int,
    emb: Optional[np.ndarray] = None,
    margin_weight: float = 10.0,
    small_frac_weight: float = 1.0,
    gamma_low: float = 1e-4,
    gamma_high: float = 10.0,
    max_steps: int = 30,
    n_explore: int = 15,
    stability_seeds: Optional[List[int]] = None,
    verbose_prefix: str = "",
) -> GammaSearchResult:
    """Two-phase gamma search.

    Phase 1: Binary search to find the valid gamma range where K in [kmin, kmax].
    Phase 2: Sample *n_explore* gammas uniformly across that range, evaluate
             margin for each, pick the best by:
             score = (0 if in_range else 1) + small_frac_weight * small_frac - margin_weight * margin.
    """
    if stability_seeds is None:
        stability_seeds = [seed, seed + 1, seed + 2]

    # ------------------------------------------------------------------
    # Phase 1: Find gamma_lo_valid and gamma_hi_valid via binary search
    # ------------------------------------------------------------------
    # In Leiden: higher gamma → MORE clusters (higher resolution)
    # gamma_lo_valid = lowest gamma where K >= kmin  (below → too few clusters)
    # gamma_hi_valid = highest gamma where K <= kmax (above → too many clusters)

    def _leiden_k(gamma: float) -> int:
        mem = leiden_membership(g, gamma, iterations, seed)
        return int(len(cluster_sizes(mem)))

    # Find gamma_lo_valid: lowest gamma where K first reaches kmin
    lo_search, hi_search = float(gamma_low), float(gamma_high)
    for _ in range(max_steps):
        mid = (lo_search + hi_search) / 2.0
        k = _leiden_k(mid)
        if k < kmin:
            lo_search = mid   # need higher gamma for more clusters
        else:
            hi_search = mid   # K >= kmin, try lower to find boundary
    gamma_lo_valid = hi_search

    # Find gamma_hi_valid: highest gamma where K still <= kmax
    lo_search, hi_search = float(gamma_low), float(gamma_high)
    for _ in range(max_steps):
        mid = (lo_search + hi_search) / 2.0
        k = _leiden_k(mid)
        if k > kmax:
            hi_search = mid   # need lower gamma for fewer clusters
        else:
            lo_search = mid   # K <= kmax, try higher to find boundary
    gamma_hi_valid = lo_search

    # ------------------------------------------------------------------
    # Phase 2: Sample n_explore gammas in the valid range, pick best
    # ------------------------------------------------------------------
    if gamma_hi_valid - gamma_lo_valid < 1e-8:
        # Degenerate range — single gamma
        candidates = [gamma_lo_valid]
    else:
        candidates = np.linspace(gamma_lo_valid, gamma_hi_valid, n_explore).tolist()

    best = None
    for gamma_cand in candidates:
        mem = leiden_membership(g, gamma_cand, iterations, seed)
        sizes = cluster_sizes(mem)
        k = int(len(sizes))
        small_frac = float(np.mean(sizes < int(min_size))) if len(sizes) else 1.0
        in_range = (kmin <= k <= kmax)

        margin = 0.0
        if emb is not None and k >= 2:
            w_in, w_bt = within_between_similarity(
                emb=emb,
                labels=[str(m) for m in mem],
                sample_per_cluster=200,
                rng_seed=seed,
            )
            margin = w_in - w_bt

        score = (0 if in_range else 1) + small_frac_weight * small_frac - margin_weight * margin

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "gamma": gamma_cand,
                "k": k,
                "small_frac": small_frac,
                "sizes": sizes.tolist(),
                "margin": margin,
            }

    gamma_best = float(best["gamma"])
    ari = stability_ari(g, gamma_best, iterations, stability_seeds)

    print(
        f"{verbose_prefix}[GAMMA] chosen gamma={gamma_best:.6f}  "
        f"K={best['k']}  small(<{min_size})={100*best['small_frac']:.1f}%  "
        f"margin={best['margin']:+.3f}  ARI(stability)={ari:.3f}",
        flush=True,
    )

    return GammaSearchResult(
        gamma=gamma_best,
        k=int(best["k"]),
        small_frac=float(best["small_frac"]),
        min_size=int(min_size),
        sizes=best["sizes"],
        ari_stability=float(ari),
        margin=float(best["margin"]),
    )


# ----------------------------
# Coherence + within/between diagnostics
# ----------------------------
def cluster_text_coherence(emb: np.ndarray, membership: List[int]) -> float:
    mem = np.array(membership, dtype=int)
    vals = np.unique(mem)
    coherences = []
    for cid in vals:
        idx = np.where(mem == cid)[0]
        if len(idx) < 2:
            continue
        vecs = emb[idx]
        centroid = vecs.mean(axis=0)
        nrm = np.linalg.norm(centroid)
        if nrm > 0:
            centroid = centroid / nrm
        sims = vecs @ centroid
        coherences.append(float(np.mean(sims)))
    return float(np.mean(coherences)) if coherences else 0.0


def within_between_similarity(
    emb: np.ndarray,
    labels: List[str],
    sample_per_cluster: int = 200,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """
    Returns (within_mean, between_mean) cosine similarity using sampling.
    emb assumed normalized.
    """
    rng = np.random.default_rng(rng_seed)
    labels = np.array(labels, dtype=object)

    # within
    within_sims = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        if len(idx) > sample_per_cluster:
            idx = rng.choice(idx, size=sample_per_cluster, replace=False)
        vecs = emb[idx]
        centroid = vecs.mean(axis=0)
        nrm = np.linalg.norm(centroid)
        if nrm > 0:
            centroid = centroid / nrm
        within_sims.append(float(np.mean(vecs @ centroid)))

    within_mean = float(np.mean(within_sims)) if within_sims else 0.0

    # between: centroids
    centroids = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        vecs = emb[idx]
        centroid = vecs.mean(axis=0)
        nrm = np.linalg.norm(centroid)
        if nrm > 0:
            centroid = centroid / nrm
        centroids.append(centroid)

    if len(centroids) < 2:
        return within_mean, 0.0

    C = np.vstack(centroids)
    sim_mat = C @ C.T
    n = sim_mat.shape[0]
    between_mean = float((sim_mat.sum() - np.trace(sim_mat)) / (n * (n - 1)))

    return within_mean, between_mean


# ----------------------------
# Peripheral attach (FULL graph indices)
# ----------------------------
def attach_peripheral_to_clusters(
    g_full: ig.Graph,
    core_vids_full: List[int],
    core_assign_df: pd.DataFrame,
    leaf_col: str,
    peripheral_vids_full: List[int],
    min_score_to_attach: float = 0.0,
) -> pd.DataFrame:
    if "vid_full" not in core_assign_df.columns:
        raise ValueError("core_assign_df must contain 'vid_full'")

    core_set = set(int(v) for v in core_vids_full)
    vid_to_cluster = dict(
        zip(core_assign_df["vid_full"].astype(int).tolist(), core_assign_df[leaf_col].astype(str).tolist())
    )

    weights = np.array(g_full.es["weight"], dtype=float) if "weight" in g_full.es.attributes() else np.ones(g_full.ecount(), dtype=float)

    out_rows = []
    for n in tqdm_force(peripheral_vids_full, desc="Attaching peripheral nodes", unit="node"):
        n = int(n)
        score_by_cluster: Dict[str, float] = {}

        for e_id in g_full.incident(n):
            e = g_full.es[e_id]
            u, v = e.tuple
            other = v if u == n else u
            if other in core_set:
                c = vid_to_cluster.get(other)
                if c is None:
                    continue
                score_by_cluster[c] = score_by_cluster.get(c, 0.0) + float(weights[e_id])

        if not score_by_cluster:
            out_rows.append({"vid_full": n, "attached_cluster": "Peripheral_NoCoreLinks", "attach_score": 0.0})
            continue

        best_cluster, best_score = max(score_by_cluster.items(), key=lambda kv: kv[1])
        if best_score <= float(min_score_to_attach):
            out_rows.append({"vid_full": n, "attached_cluster": "Peripheral_Unassigned", "attach_score": float(best_score)})
        else:
            out_rows.append({"vid_full": n, "attached_cluster": str(best_cluster), "attach_score": float(best_score)})

    return pd.DataFrame(out_rows)


# ----------------------------
# Recursive clustering engine + diagnostics
# ----------------------------
@dataclass
class LevelConfig:
    kmin: int
    kmax: int
    min_size: int
    split_if_gt: int


def pad_list(vals: List[int], needed: int) -> List[int]:
    if len(vals) >= needed:
        return vals[:needed]
    return vals + [vals[-1]] * (needed - len(vals))


def recursive_split(
    g_core: ig.Graph,
    emb_core: np.ndarray,
    core_vids_full: List[int],
    titles_full: List[str],
    level_cols: List[str],
    cfgs: List[LevelConfig],
    iterations: int,
    seed: int,
    margin_weight: float = 10.0,
    small_frac_weight: float = 1.0,
    mega_threshold: int = 0,
    local_tau: float = 0.0,
    prune_pct: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      - assign DataFrame with columns: vid_full, page, root, L1..Ln
      - run_diag dict with per-group and per-level metrics (machine readable)
    """
    assert g_core.vcount() == emb_core.shape[0]

    run_diag: Dict[str, Any] = {
        "levels": {},   # rollups per level
        "groups": [],   # per parent-group decisions
    }

    assign = pd.DataFrame({
        "vid_full": core_vids_full,
        "page": [titles_full[v] for v in core_vids_full],
    })

    core_pos = {vid: i for i, vid in enumerate(core_vids_full)}

    for lvl_i, lvl_name in enumerate(level_cols):
        cfg = cfgs[lvl_i]
        parent_col = None if lvl_i == 0 else level_cols[lvl_i - 1]

        print("\n" + "=" * 98, flush=True)
        print(f"[LEVEL {lvl_name}] Adaptive clustering", flush=True)
        print(f"Targets: K in [{cfg.kmin},{cfg.kmax}], min_size={cfg.min_size}, split_if_size_gt={cfg.split_if_gt}", flush=True)
        print("=" * 98, flush=True)

        if parent_col is None:
            assign["__ALL__"] = "__ALL__"
            parent_col = "__ALL__"
            groups = ["__ALL__"]
        else:
            groups = assign[parent_col].astype(str).unique().tolist()

        assign[lvl_name] = None

        for grp in tqdm_force(groups, desc=f"Splitting {parent_col} -> {lvl_name}", unit="group"):
            rows = assign.index[assign[parent_col].astype(str) == str(grp)].tolist()
            n_nodes = len(rows)

            # If too small, do not split
            if n_nodes <= int(cfg.split_if_gt) or n_nodes < 2 * int(cfg.min_size):
                for idx in rows:
                    assign.at[idx, lvl_name] = f"{grp}|{lvl_name}_0"
                # record a diag row for “no split”
                run_diag["groups"].append({
                    "level": lvl_name,
                    "parent_group": str(grp),
                    "n_nodes": int(n_nodes),
                    "did_split": False,
                    "gamma": None,
                    "K": 1,
                    "small_frac": None,
                    "ari_stability": None,
                    "coherence": None,
                    "within_mean": None,
                    "between_mean": None,
                    "within_minus_between": None,
                })
                continue

            # Build subgraph for this group (core-indexed)
            sub_nodes = [core_pos[int(assign.at[idx, "vid_full"])] for idx in rows]
            subg = g_core.subgraph(sub_nodes)
            sub_emb = emb_core[sub_nodes]

            # Percentile-based adaptive pruning (fires for ALL subgraphs)
            if prune_pct > 0 and subg.ecount() >= 5:
                weights_arr = np.array(subg.es["weight"], dtype=float)
                threshold = float(np.percentile(weights_arr, prune_pct))
                drop = np.where(weights_arr < threshold)[0].tolist()
                # Safeguard: keep at least max(n_nodes//4, 5) edges
                if drop and (subg.ecount() - len(drop)) >= max(n_nodes // 4, 5):
                    subg.delete_edges(drop)
                    n_isolated = sum(1 for v in range(subg.vcount()) if subg.degree(v) == 0)
                    print(
                        f"[{lvl_name} grp={str(grp)[:40]}] ADAPTIVE_TAU pct={prune_pct}%: "
                        f"threshold={threshold:.4f}, pruned {len(drop)}/{len(weights_arr)} edges, "
                        f"{n_isolated}/{subg.vcount()} isolated",
                        flush=True,
                    )
            # Legacy: absolute local tau for mega-clusters
            elif mega_threshold > 0 and local_tau > 0 and n_nodes > mega_threshold:
                weights = subg.es["weight"] if "weight" in subg.es.attributes() else [1.0] * subg.ecount()
                drop = [i for i, w in enumerate(weights) if w < local_tau]
                if drop:
                    subg.delete_edges(drop)
                    n_isolated = sum(1 for v in range(subg.vcount()) if subg.degree(v) == 0)
                    print(
                        f"[{lvl_name} grp={str(grp)[:40]}] LOCAL TAU={local_tau}: "
                        f"pruned {len(drop)} edges, {n_isolated}/{subg.vcount()} isolated",
                        flush=True,
                    )

            # Choose gamma adaptively
            res = adaptive_gamma_search(
                subg,
                kmin=cfg.kmin,
                kmax=cfg.kmax,
                min_size=cfg.min_size,
                iterations=iterations,
                seed=seed,
                emb=sub_emb,
                margin_weight=margin_weight,
                small_frac_weight=small_frac_weight,
                verbose_prefix=f"[{lvl_name} grp={str(grp)[:40]}] ",
            )

            # Run Leiden with chosen gamma
            mem = leiden_membership(subg, res.gamma, iterations, seed)

            # Diagnostics: coherence + within/between semantic similarity
            coh = cluster_text_coherence(sub_emb, mem)
            w_in, w_bt = within_between_similarity(
                emb=sub_emb,
                labels=[str(m) for m in mem],
                sample_per_cluster=200,
                rng_seed=seed,
            )

            print(
                f"[{lvl_name} grp={str(grp)[:40]}] coherence={coh:.3f} within={w_in:.3f} between={w_bt:.3f}",
                flush=True,
            )

            # Record group-level diag
            run_diag["groups"].append({
                "level": lvl_name,
                "parent_group": str(grp),
                "n_nodes": int(n_nodes),
                "did_split": True,
                "gamma": float(res.gamma),
                "K": int(res.k),
                "small_frac": float(res.small_frac),
                "ari_stability": float(res.ari_stability),
                "coherence": float(coh),
                "within_mean": float(w_in),
                "between_mean": float(w_bt),
                "within_minus_between": float(w_in - w_bt),
            })

            # Assign cluster ids (tree path)
            for local_i, idx in enumerate(rows):
                assign.at[idx, lvl_name] = f"{grp}|{lvl_name}_{mem[local_i]}"

        # Level rollups
        sizes = assign.groupby(lvl_name)["page"].size().values
        if len(sizes):
            print(
                f"[{lvl_name}] clusters={len(sizes)}  min={sizes.min()}  median={int(np.median(sizes))}  "
                f"p90={int(np.percentile(sizes,90))}  max={sizes.max()}",
                flush=True,
            )

        grows = [g for g in run_diag["groups"] if g["level"] == lvl_name and g.get("did_split")]

        lvl_summary: Dict[str, Any] = {
            "n_clusters": int(len(sizes)),
            "min_cluster": int(sizes.min()) if len(sizes) else 0,
            "median_cluster": float(np.median(sizes)) if len(sizes) else 0.0,
            "p90_cluster": float(np.percentile(sizes, 90)) if len(sizes) else 0.0,
            "max_cluster": int(sizes.max()) if len(sizes) else 0,
            "n_groups_total": int(len(groups)),
            "n_groups_split": int(len(grows)),
        }

        if grows:
            lvl_summary.update({
                "gamma_mean": float(np.mean([g["gamma"] for g in grows if g["gamma"] is not None])),
                "ari_mean": float(np.mean([g["ari_stability"] for g in grows if g["ari_stability"] is not None])),
                "coherence_mean": float(np.mean([g["coherence"] for g in grows if g["coherence"] is not None])),
                "within_mean": float(np.mean([g["within_mean"] for g in grows if g["within_mean"] is not None])),
                "between_mean": float(np.mean([g["between_mean"] for g in grows if g["between_mean"] is not None])),
                "within_minus_between": float(np.mean([g["within_minus_between"] for g in grows if g["within_minus_between"] is not None])),
            })
        else:
            lvl_summary.update({
                "gamma_mean": None,
                "ari_mean": None,
                "coherence_mean": None,
                "within_mean": None,
                "between_mean": None,
                "within_minus_between": None,
            })

        run_diag["levels"][lvl_name] = lvl_summary

    if "__ALL__" in assign.columns:
        assign.drop(columns=["__ALL__"], inplace=True)

    return assign, run_diag


# ----------------------------
# Post-hoc resplitting of problem clusters
# ----------------------------

def _compute_page_margins(
    emb: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Per-page margin: sim_to_own_centroid − sim_to_nearest_other_centroid.

    Returns (N,) array of per-page margins.  Positive = well-placed.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.zeros(len(labels))

    # Build L2-normalised centroids
    centroids = {}
    for cid in unique_labels:
        mask = labels == cid
        c = emb[mask].mean(axis=0)
        nrm = np.linalg.norm(c)
        if nrm > 0:
            c /= nrm
        centroids[cid] = c

    cid_list = list(centroids.keys())
    centroid_mat = np.vstack([centroids[c] for c in cid_list])   # (K, D)
    cid_to_idx = {c: i for i, c in enumerate(cid_list)}

    sims = emb @ centroid_mat.T  # (N, K) — cosine similarities (assumes normalised emb)
    margins = np.empty(len(labels))
    for i in range(len(labels)):
        own_idx = cid_to_idx[labels[i]]
        own_sim = sims[i, own_idx]
        other_sims = np.concatenate([sims[i, :own_idx], sims[i, own_idx + 1:]])
        margins[i] = own_sim - other_sims.max()

    return margins


def _flag_problem_clusters(
    assign: pd.DataFrame,
    emb_core: np.ndarray,
    core_pos: Dict[int, int],
    level_col: str,
    parent_col: Optional[str],
    *,
    mega_frac: float = 0.9,
    mega_min_size: int = 10,
    weak_margin: float = 0.0,
    weak_min_size: int = 10,
) -> List[Dict]:
    """Identify MEGA and WEAK clusters at a given level.

    Returns list of dicts with keys: cluster_id, flags, size, parent_size,
    fraction, mean_margin, pct_well_placed.
    """
    results = []
    non_periph = assign[assign[level_col].astype(str) != "Peripheral"]
    all_labels = non_periph[level_col].astype(str).values

    # Get embeddings in assignment order
    emb_indices = [core_pos[int(v)] for v in non_periph["vid_full"]]
    emb_sub = emb_core[emb_indices]

    # Per-page margins at this level
    page_margins = _compute_page_margins(emb_sub, all_labels)

    for cid, grp in non_periph.groupby(level_col):
        cid_str = str(cid)
        sz = len(grp)
        mask = all_labels == cid_str

        # Parent fraction
        if parent_col is not None:
            parent_id = grp[parent_col].iloc[0]
            parent_sz = int((non_periph[parent_col] == parent_id).sum())
        else:
            parent_sz = len(non_periph)
        frac = sz / max(parent_sz, 1)

        margin_vals = page_margins[mask]
        mean_m = float(margin_vals.mean())
        pct_wp = float((margin_vals > 0).mean() * 100)

        flags = []
        if frac > mega_frac and sz > mega_min_size:
            flags.append("MEGA")
        if mean_m < weak_margin and sz > weak_min_size:
            flags.append("WEAK")

        if flags:
            results.append({
                "cluster_id": cid_str,
                "flags": flags,
                "size": sz,
                "parent_size": parent_sz,
                "fraction": round(frac, 4),
                "mean_margin": round(mean_m, 4),
                "pct_well_placed": round(pct_wp, 1),
            })

    return results


def collapse_single_child(
    assign: pd.DataFrame,
    level_cols: List[str],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Collapse single-child branches in the taxonomy hierarchy.

    For each level (bottom-up: L2 first, then L1), find parent clusters
    with exactly 1 non-peripheral child.  For those, set the child column
    equal to the parent column value — the child level is absorbed.

    After collapse, ``L1 == L2`` for collapsed branches, which is
    detectable and works with downstream ``groupby`` operations.

    Parameters
    ----------
    assign : pd.DataFrame
        Core page assignments with columns ``[*level_cols, vid_full, page]``.
    level_cols : list[str]
        Hierarchy levels coarsest→finest, e.g. ``["root", "L1", "L2"]``.

    Returns
    -------
    (assign, report) : tuple
        *assign* is modified in-place and also returned.
        *report* is a list of dicts documenting each collapsed branch.
    """
    report: List[Dict[str, Any]] = []

    # Process bottom-up: deepest child first
    for lvl_i in reversed(range(1, len(level_cols))):
        child_col = level_cols[lvl_i]        # e.g. "L2"
        parent_col = level_cols[lvl_i - 1]   # e.g. "L1"

        # Exclude rows where child is already peripheral
        non_periph = assign[
            ~assign[child_col].astype(str).str.contains("Peripheral", na=False)
        ]

        for parent_id, grp in non_periph.groupby(parent_col):
            child_ids = grp[child_col].astype(str).unique()
            # Filter out any peripheral-tagged children
            real_children = [c for c in child_ids if "Peripheral" not in c]

            if len(real_children) == 1:
                # Collapse: set child column = parent column for ALL rows
                # (including peripheral rows under this parent)
                rows = assign.index[assign[parent_col].astype(str) == str(parent_id)]
                old_child = real_children[0]
                assign.loc[rows, child_col] = str(parent_id)

                report.append({
                    "parent_col": parent_col,
                    "child_col": child_col,
                    "parent_id": str(parent_id),
                    "collapsed_child_id": old_child,
                    "n_pages": len(rows),
                })

    if report:
        print(f"  [COLLAPSE] {len(report)} single-child branches collapsed", flush=True)
    return assign, report


def posthoc_resplit(
    assign: pd.DataFrame,
    g_core: ig.Graph,
    emb_core: np.ndarray,
    core_vids_full: List[int],
    level_cols: List[str],
    cfgs: List[LevelConfig],
    iterations: int,
    seed: int,
    margin_weight: float = 10.0,
    small_frac_weight: float = 1.0,
    local_tau: float = 0.0,
    run_diag: Optional[Dict[str, Any]] = None,
    *,
    mega_frac: float = 0.9,
    mega_min_size: int = 10,
    weak_margin: float = 0.0,
    weak_min_size: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Post-hoc resplitting of MEGA / WEAK flagged clusters.

    For each flagged cluster at each level:
      1. Extract induced subgraph
      2. Optionally prune edges below local_tau
      3. Run adaptive_gamma_search + Leiden with relaxed kmax (2× default)
      4. Singleton sub-clusters → labelled as Peripheral
      5. Replace assignments with new sub-clusters if improvement

    Parameters
    ----------
    assign : DataFrame from recursive_split() with vid_full, page, root, L1, L2
    g_core : core graph (igraph)
    emb_core : (N, D) normalised embeddings for core nodes
    core_vids_full : mapping from position to full graph vertex id
    level_cols : e.g. ["root", "L1", "L2"]
    cfgs : LevelConfig per level
    iterations, seed : Leiden params
    margin_weight, small_frac_weight : gamma search params
    local_tau : edge threshold applied within flagged clusters (0 = no pruning)
    run_diag : existing run diagnostics dict (updated in-place)
    mega_frac : MEGA flag fraction threshold (default 0.9)
    mega_min_size : MEGA flag minimum cluster size (default 10)
    weak_margin : WEAK flag margin threshold (default 0.0)
    weak_min_size : WEAK flag minimum cluster size (default 10)

    Returns
    -------
    (assign_updated, resplit_report) where resplit_report has per-cluster details.
    """
    if run_diag is None:
        run_diag = {"levels": {}, "groups": []}

    # Pre-resplit collapse: absorb single-child branches
    assign, pre_collapse = collapse_single_child(assign, level_cols)
    run_diag["pre_collapse"] = pre_collapse
    print(f"[PRE-COLLAPSE] {len(pre_collapse)} single-child branches collapsed", flush=True)

    core_pos = {vid: i for i, vid in enumerate(core_vids_full)}
    assign = assign.copy()
    report = {"resplits": [], "summary": {}}

    total_resplit = 0
    total_flagged = 0

    for lvl_i, lvl_name in enumerate(level_cols):
        cfg = cfgs[lvl_i]
        parent_col = level_cols[lvl_i - 1] if lvl_i > 0 else None

        print(f"\n{'=' * 98}", flush=True)
        print(f"[POST-HOC] Checking {lvl_name} for MEGA/WEAK clusters", flush=True)
        print(f"{'=' * 98}", flush=True)

        # Skip root — too coarse to resplit
        if lvl_i == 0:
            print(f"  Skipping {lvl_name} — root too coarse to resplit", flush=True)
            continue

        # Tiered local tau: L1 uses local_tau, deeper levels add +0.05 per level
        level_tau = local_tau + 0.05 * (lvl_i - 1) if local_tau > 0 else 0.0

        flagged = _flag_problem_clusters(
            assign, emb_core, core_pos, lvl_name, parent_col,
            mega_frac=mega_frac, mega_min_size=mega_min_size,
            weak_margin=weak_margin, weak_min_size=weak_min_size,
        )

        if not flagged:
            print(f"  No problem clusters at {lvl_name}", flush=True)
            continue

        total_flagged += len(flagged)
        print(f"  Found {len(flagged)} problem cluster(s):", flush=True)
        for f in flagged:
            print(
                f"    {f['cluster_id'][:60]:60s}  size={f['size']:5d}  "
                f"frac={f['fraction']:.2f}  margin={f['mean_margin']:.4f}  "
                f"wp={f['pct_well_placed']:.1f}%  [{', '.join(f['flags'])}]",
                flush=True,
            )

        for finfo in flagged:
            cid = finfo["cluster_id"]
            rows = assign.index[assign[lvl_name].astype(str) == cid].tolist()
            n_nodes = len(rows)

            if n_nodes < 2 * cfg.min_size or n_nodes < 4:
                print(f"  Skipping {cid} — too small to resplit ({n_nodes} pages)", flush=True)
                continue

            # Extract subgraph
            sub_nodes = [core_pos[int(assign.at[idx, "vid_full"])] for idx in rows]
            subg = g_core.subgraph(sub_nodes)
            sub_emb = emb_core[sub_nodes]

            # Apply tiered local tau pruning
            n_pruned = 0
            if level_tau > 0 and subg.ecount() > 0:
                weights = subg.es["weight"] if "weight" in subg.es.attributes() else [1.0] * subg.ecount()
                drop = [i for i, w in enumerate(weights) if w < level_tau]
                if drop:
                    subg.delete_edges(drop)
                    n_pruned = len(drop)
                    n_isolated = sum(1 for v in range(subg.vcount()) if subg.degree(v) == 0)
                    print(
                        f"  [{cid[:50]}] level_tau={level_tau:.3f}: pruned {n_pruned} edges, "
                        f"{n_isolated} isolated nodes",
                        flush=True,
                    )

            if subg.ecount() == 0:
                print(f"  Skipping {cid} — no edges after pruning", flush=True)
                continue

            # Relaxed kmax: 2× default or at least 12
            relaxed_kmax = max(cfg.kmax * 2, 12)

            # Run adaptive gamma search with relaxed constraints
            res = adaptive_gamma_search(
                subg,
                kmin=max(2, cfg.kmin),
                kmax=relaxed_kmax,
                min_size=max(cfg.min_size, 3),
                iterations=iterations,
                seed=seed,
                emb=sub_emb,
                margin_weight=margin_weight,
                small_frac_weight=small_frac_weight,
                verbose_prefix=f"  [resplit {cid[:40]}] ",
            )

            if res.k <= 1:
                print(f"  Skipping {cid} — Leiden found only 1 cluster", flush=True)
                continue

            mem = leiden_membership(subg, res.gamma, iterations, seed)

            # Label sub-clusters; singletons → Peripheral
            PERIPH_LABEL = f"{cid}|Peripheral"
            mem_arr = np.array(mem)
            unique_c, counts_c = np.unique(mem_arr, return_counts=True)
            singleton_clusters = set(unique_c[counts_c == 1])

            new_labels = np.array([
                PERIPH_LABEL if mem[i] in singleton_clusters else f"{cid}|re_{mem[i]}"
                for i in range(len(mem))
            ])
            n_periph = int(sum(1 for m in mem if m in singleton_clusters))

            new_k = len(unique_c) - len(singleton_clusters)
            if new_k < 2:
                print(f"  Skipping {cid} — only {new_k} non-singleton sub-cluster(s)", flush=True)
                continue

            new_sizes = cluster_sizes(mem)
            new_margins = _compute_page_margins(sub_emb, new_labels)
            new_pct_wp = float((new_margins > 0).mean() * 100)
            new_mean_margin = float(new_margins.mean())

            print(
                f"  [{cid[:50]}] resplit: K={new_k} (+{n_periph} periph), "
                f"sizes={sorted(new_sizes, reverse=True)[:5]}, "
                f"new_margin={new_mean_margin:.4f}, new_wp={new_pct_wp:.1f}%",
                flush=True,
            )

            # Accept resplit if it produces a positive improvement
            if new_mean_margin > finfo["mean_margin"] or new_pct_wp > finfo["pct_well_placed"]:
                for local_i, idx in enumerate(rows):
                    assign.at[idx, lvl_name] = new_labels[local_i]

                total_resplit += 1
                status = "ACCEPTED"
            else:
                status = "REJECTED (no improvement)"

            report["resplits"].append({
                "level": lvl_name,
                "cluster_id": cid,
                "flags": finfo["flags"],
                "original_size": n_nodes,
                "original_margin": finfo["mean_margin"],
                "original_pct_wp": finfo["pct_well_placed"],
                "new_k": new_k,
                "n_peripheral": n_periph,
                "new_margin": round(new_mean_margin, 4),
                "new_pct_wp": round(new_pct_wp, 1),
                "edges_pruned": n_pruned,
                "gamma": round(res.gamma, 4),
                "status": status,
            })

            print(f"  → {status}", flush=True)

        # If any resplits accepted at this level, recluster child levels
        if any(r["status"] == "ACCEPTED" and r["level"] == lvl_name for r in report["resplits"]):
            # Check if there are child levels with None values that need reclustering
            for child_i in range(lvl_i + 1, len(level_cols)):
                child_lvl = level_cols[child_i]
                child_cfg = cfgs[child_i]
                needs_recluster = assign[assign[child_lvl].isna()]

                if len(needs_recluster) == 0:
                    continue

                print(f"\n  [RECLUSTER] {len(needs_recluster)} pages need {child_lvl} reassignment", flush=True)

                # Group by parent (current level)
                parent_lvl_for_child = level_cols[child_i - 1]
                new_parents = needs_recluster[parent_lvl_for_child].unique()

                for new_parent in new_parents:
                    parent_rows = assign.index[assign[parent_lvl_for_child].astype(str) == str(new_parent)].tolist()
                    n_child = len(parent_rows)

                    if n_child <= int(child_cfg.split_if_gt) or n_child < 2 * int(child_cfg.min_size):
                        for idx in parent_rows:
                            assign.at[idx, child_lvl] = f"{new_parent}|{child_lvl}_0"
                        continue

                    child_sub_nodes = [core_pos[int(assign.at[idx, "vid_full"])] for idx in parent_rows]
                    child_subg = g_core.subgraph(child_sub_nodes)
                    child_emb = emb_core[child_sub_nodes]

                    if child_subg.ecount() == 0:
                        for idx in parent_rows:
                            assign.at[idx, child_lvl] = f"{new_parent}|{child_lvl}_0"
                        continue

                    child_res = adaptive_gamma_search(
                        child_subg,
                        kmin=child_cfg.kmin,
                        kmax=child_cfg.kmax,
                        min_size=child_cfg.min_size,
                        iterations=iterations,
                        seed=seed,
                        emb=child_emb,
                        margin_weight=margin_weight,
                        small_frac_weight=small_frac_weight,
                        verbose_prefix=f"  [recluster {child_lvl} parent={str(new_parent)[:30]}] ",
                    )

                    child_mem = leiden_membership(child_subg, child_res.gamma, iterations, seed)

                    for local_i, idx in enumerate(parent_rows):
                        assign.at[idx, child_lvl] = f"{new_parent}|{child_lvl}_{child_mem[local_i]}"

                    print(
                        f"    {str(new_parent)[:50]} → K={len(set(child_mem))}, "
                        f"sizes={sorted(cluster_sizes(child_mem), reverse=True)[:5]}",
                        flush=True,
                    )

    report["summary"] = {
        "total_flagged": total_flagged,
        "total_resplit": total_resplit,
        "total_rejected": total_flagged - total_resplit,
    }

    # Store in run_diag
    run_diag["posthoc_resplit"] = report

    print(f"\n{'=' * 98}", flush=True)
    print(
        f"[POST-HOC DONE] Flagged: {total_flagged}, "
        f"Resplit: {total_resplit}, Rejected: {total_flagged - total_resplit}",
        flush=True,
    )
    print(f"{'=' * 98}", flush=True)

    # Populate run_diag["levels"] from final assignments
    _populate_level_summaries(assign, level_cols, emb_core, core_pos, run_diag)

    return assign, report


def _populate_level_summaries(
    assign: pd.DataFrame,
    level_cols: List[str],
    emb_core: np.ndarray,
    core_pos: Dict[int, int],
    run_diag: Dict[str, Any],
) -> None:
    """Compute per-level summary stats and store in run_diag['levels'].

    Works for any code path (recursive_split, posthoc_resplit, notebook).
    """
    for lvl_name in level_cols:
        non_periph = assign[~assign[lvl_name].astype(str).str.contains("Peripheral", na=False)]
        sizes = non_periph.groupby(lvl_name)["page"].size().values

        if len(sizes) == 0:
            continue

        # Coherence + margin per cluster
        coherences, within_means, between_means, margins = [], [], [], []
        for cid, grp in non_periph.groupby(lvl_name):
            vids = grp["vid_full"].astype(int).tolist()
            positions = [core_pos[v] for v in vids if v in core_pos]
            if len(positions) < 2:
                continue
            emb_sub = emb_core[positions]
            centroid = emb_sub.mean(axis=0)
            nrm = np.linalg.norm(centroid)
            if nrm > 0:
                centroid /= nrm
            within = float(np.mean(emb_sub @ centroid))
            coherences.append(within)
            within_means.append(within)

        # Between: mean similarity to nearest other cluster centroid
        cluster_ids = list(non_periph[lvl_name].unique())
        if len(cluster_ids) >= 2:
            centroids = {}
            for cid in cluster_ids:
                grp = non_periph[non_periph[lvl_name] == cid]
                vids = grp["vid_full"].astype(int).tolist()
                positions = [core_pos[v] for v in vids if v in core_pos]
                if positions:
                    c = emb_core[positions].mean(axis=0)
                    nrm = np.linalg.norm(c)
                    if nrm > 0:
                        c /= nrm
                    centroids[cid] = c

            cid_list = list(centroids.keys())
            if len(cid_list) >= 2:
                cent_mat = np.array([centroids[c] for c in cid_list])
                sim_mat = cent_mat @ cent_mat.T
                np.fill_diagonal(sim_mat, -np.inf)
                for i, cid in enumerate(cid_list):
                    nearest = float(sim_mat[i].max())
                    between_means.append(nearest)
                    if i < len(within_means):
                        margins.append(within_means[i] - nearest)

        lvl_summary = {
            "n_clusters": int(len(sizes)),
            "min_cluster": int(sizes.min()),
            "median_cluster": float(np.median(sizes)),
            "p90_cluster": float(np.percentile(sizes, 90)),
            "max_cluster": int(sizes.max()),
            "coherence_mean": float(np.mean(coherences)) if coherences else None,
            "within_mean": float(np.mean(within_means)) if within_means else None,
            "between_mean": float(np.mean(between_means)) if between_means else None,
            "within_minus_between": float(np.mean(margins)) if margins else None,
            "gamma_mean": None,
            "ari_mean": None,
        }
        run_diag["levels"][lvl_name] = lvl_summary

    parts = [f"{k}={v['n_clusters']}" for k, v in run_diag["levels"].items()]
    print(f"[DIAG] Level summaries: {', '.join(parts)}", flush=True)


# ----------------------------
# Shared helpers for new strategies
# ----------------------------

def semantic_split(
    emb: np.ndarray,
    seed: int = 42,
    s_min: int = 2,
    s_max: int = 12,
    min_bucket_size: int = 10,
) -> List[np.ndarray]:
    """Split nodes into semantic buckets using KMeans on embeddings.

    Adaptively increases S (number of buckets) from s_min up to s_max,
    picking the S that maximises mean within-centroid similarity.
    Buckets smaller than min_bucket_size are merged into the nearest
    larger bucket.

    Returns list of index arrays (one per bucket).
    """
    from sklearn.cluster import KMeans

    n = emb.shape[0]
    if n < 2 * min_bucket_size or n < 2 * s_min:
        return [np.arange(n)]

    max_s = min(s_max, n // max(min_bucket_size, 1))
    if max_s < s_min:
        return [np.arange(n)]

    best_score = -np.inf
    best_labels = np.zeros(n, dtype=int)

    for s in range(s_min, max_s + 1):
        km = KMeans(n_clusters=s, random_state=seed, n_init=3, max_iter=100)
        labels = km.fit_predict(emb)

        # Score: size-weighted mean within-centroid cosine similarity
        score = 0.0
        for c in range(s):
            idx = np.where(labels == c)[0]
            if len(idx) < 2:
                continue
            centroid = emb[idx].mean(axis=0)
            nrm = np.linalg.norm(centroid)
            if nrm > 0:
                centroid /= nrm
            score += float(np.mean(emb[idx] @ centroid)) * len(idx)
        score /= n

        if score > best_score:
            best_score = score
            best_labels = labels.copy()
        elif score < best_score - 0.01:
            break  # diminishing returns

    # Enforce min_bucket_size: merge small buckets into nearest larger one
    labels = best_labels.copy()
    centroids: Dict[int, Tuple[np.ndarray, int]] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        centroid = emb[idx].mean(axis=0)
        nrm = np.linalg.norm(centroid)
        if nrm > 0:
            centroid /= nrm
        centroids[c] = (centroid, len(idx))

    changed = True
    while changed:
        changed = False
        for c in list(centroids.keys()):
            if c not in centroids:
                continue
            if centroids[c][1] < min_bucket_size:
                best_target, best_sim = None, -np.inf
                for other in centroids:
                    if other == c or centroids[other][1] < min_bucket_size:
                        continue
                    sim = float(centroids[c][0] @ centroids[other][0])
                    if sim > best_sim:
                        best_sim = sim
                        best_target = other
                if best_target is not None:
                    labels[labels == c] = best_target
                    new_idx = np.where(labels == best_target)[0]
                    new_centroid = emb[new_idx].mean(axis=0)
                    nrm = np.linalg.norm(new_centroid)
                    if nrm > 0:
                        new_centroid /= nrm
                    centroids[best_target] = (new_centroid, len(new_idx))
                    del centroids[c]
                    changed = True

    buckets = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) > 0:
            buckets.append(idx)

    return buckets if buckets else [np.arange(n)]


def compute_per_cluster_margins_inline(
    emb: np.ndarray,
    membership: List[int],
) -> pd.DataFrame:
    """Thin wrapper: compute per-cluster margins from embeddings + integer membership.

    Lazily imports compute_cluster_margins to avoid circular import.
    """
    from p2_taxonomy.analysis_cluster_quality import compute_cluster_margins
    labels = np.array([str(m) for m in membership], dtype=object)
    return compute_cluster_margins(emb, labels)


def detect_mega_clusters(
    membership: List[int],
    parent_size: int,
    pct_threshold: float = 0.9,
) -> List[int]:
    """Return cluster IDs where size > pct_threshold * parent_size."""
    mem_arr = np.array(membership, dtype=int)
    unique, counts = np.unique(mem_arr, return_counts=True)
    mega = []
    for cid, cnt in zip(unique, counts):
        if cnt > pct_threshold * parent_size:
            mega.append(int(cid))
    return mega


# ----------------------------
# Strategy 1: Simple Leiden Sweep
# ----------------------------

def strategy_simple_sweep(
    g_core: ig.Graph,
    emb_core: np.ndarray,
    core_vids_full: List[int],
    titles_full: List[str],
    level_cols: List[str],
    cfgs: List[LevelConfig],
    iterations: int,
    seed: int,
    margin_weight: float = 10.0,
    small_frac_weight: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Simple Leiden sweep with total cluster count targets and constraints.

    Targets total cluster counts per level:
      Root: 10-12, L1: 45+, L2: 150+
    Constraints: all per-cluster margins > 0, no mega-clusters (>90% of parent).
    Falls back to best compromise if no candidate is fully valid.

    Returns (assign DataFrame, run_diag dict) — same interface as recursive_split().
    """
    total_targets = [(10, 12), (45, 80), (150, 250)]

    assert g_core.vcount() == emb_core.shape[0]

    run_diag: Dict[str, Any] = {"levels": {}, "groups": []}

    assign = pd.DataFrame({
        "vid_full": core_vids_full,
        "page": [titles_full[v] for v in core_vids_full],
    })

    core_pos = {vid: i for i, vid in enumerate(core_vids_full)}

    for lvl_i, lvl_name in enumerate(level_cols):
        cfg = cfgs[lvl_i]
        target_min, target_max = total_targets[lvl_i] if lvl_i < len(total_targets) else (cfg.kmin, cfg.kmax)
        parent_col = None if lvl_i == 0 else level_cols[lvl_i - 1]

        print(f"\n{'=' * 98}", flush=True)
        print(f"[LEVEL {lvl_name}] Simple sweep (target total: {target_min}-{target_max})", flush=True)
        print(f"{'=' * 98}", flush=True)

        if parent_col is None:
            assign["__ALL__"] = "__ALL__"
            parent_col = "__ALL__"
            groups = ["__ALL__"]
        else:
            groups = assign[parent_col].astype(str).unique().tolist()

        assign[lvl_name] = None
        total_k = 0

        # Sort groups largest-first so big groups get budget priority
        grp_sizes = {}
        for grp in groups:
            grp_sizes[grp] = int((assign[parent_col].astype(str) == str(grp)).sum())
        groups_sorted = sorted(groups, key=lambda g: grp_sizes[g], reverse=True)

        # Count groups that won't be split (too small) — they each contribute 1
        n_unsplittable = sum(
            1 for g in groups_sorted
            if grp_sizes[g] <= int(cfg.split_if_gt) or grp_sizes[g] < 2 * int(cfg.min_size)
        )

        for grp in tqdm_force(groups_sorted, desc=f"Splitting {parent_col} -> {lvl_name}", unit="group"):
            rows = assign.index[assign[parent_col].astype(str) == str(grp)].tolist()
            n_nodes = len(rows)

            if n_nodes <= int(cfg.split_if_gt) or n_nodes < 2 * int(cfg.min_size):
                for idx in rows:
                    assign.at[idx, lvl_name] = f"{grp}|{lvl_name}_0"
                total_k += 1
                run_diag["groups"].append({
                    "level": lvl_name, "parent_group": str(grp),
                    "n_nodes": int(n_nodes), "did_split": False,
                    "gamma": None, "K": 1,
                })
                continue

            # Budget: how many clusters this group can contribute before total exceeds target_max
            remaining_budget = max(2, target_max - total_k)

            # If total already at/past target_max, don't split this group
            if total_k >= target_max:
                for idx in rows:
                    assign.at[idx, lvl_name] = f"{grp}|{lvl_name}_0"
                total_k += 1
                run_diag["groups"].append({
                    "level": lvl_name, "parent_group": str(grp),
                    "n_nodes": int(n_nodes), "did_split": False,
                    "gamma": None, "K": 1,
                    "constraints_failed": ["budget_exhausted"],
                })
                continue

            sub_nodes = [core_pos[int(assign.at[idx, "vid_full"])] for idx in rows]
            subg = g_core.subgraph(sub_nodes)
            sub_emb = emb_core[sub_nodes]

            # Gamma sweep with constraint checking
            gammas = np.logspace(-4, 1, 40)
            best_cand = None
            best_score = np.inf

            for gamma_cand in gammas:
                mem = leiden_membership(subg, float(gamma_cand), iterations, seed)
                sizes = cluster_sizes(mem)
                k = int(len(sizes))
                if k < 2:
                    continue
                # Skip candidates that exceed this group's budget
                if k > remaining_budget:
                    continue

                max_frac = float(sizes.max()) / n_nodes
                has_mega = max_frac > 0.9

                # Per-cluster margin check
                if k >= 2:
                    margins_df = compute_per_cluster_margins_inline(sub_emb, mem)
                    min_margin = float(margins_df["margin"].min()) if len(margins_df) > 1 else 0.0
                    mean_margin = float(margins_df["margin"].mean()) if len(margins_df) > 1 else 0.0
                else:
                    min_margin, mean_margin = 0.0, 0.0

                all_positive = min_margin > 0
                small_frac = float(np.mean(sizes < int(cfg.min_size))) if len(sizes) else 1.0

                # Constraint check
                constraints_ok = (not has_mega) and all_positive

                # Score: lower is better
                penalty = 0.0
                if has_mega:
                    penalty += 10.0
                if not all_positive:
                    penalty += 5.0 * max(0.0, -min_margin)
                score = penalty - margin_weight * mean_margin + small_frac_weight * small_frac

                if best_cand is None or \
                   (constraints_ok and not best_cand.get("constraints_ok") ) or \
                   (constraints_ok == best_cand.get("constraints_ok") and score < best_score):
                    best_cand = {
                        "gamma": float(gamma_cand), "k": k, "mem": mem,
                        "sizes": sizes.tolist(), "max_frac": max_frac,
                        "min_margin": min_margin, "mean_margin": mean_margin,
                        "small_frac": small_frac, "constraints_ok": constraints_ok,
                    }
                    best_score = score

            if best_cand is None:
                for idx in rows:
                    assign.at[idx, lvl_name] = f"{grp}|{lvl_name}_0"
                total_k += 1
                run_diag["groups"].append({
                    "level": lvl_name, "parent_group": str(grp),
                    "n_nodes": int(n_nodes), "did_split": False,
                    "gamma": None, "K": 1,
                    "constraints_failed": ["no_valid_candidate_within_budget"],
                })
                continue

            mem = best_cand["mem"]
            total_k += best_cand["k"]

            coh = cluster_text_coherence(sub_emb, mem)
            w_in, w_bt = within_between_similarity(sub_emb, [str(m) for m in mem], rng_seed=seed)

            diag_entry: Dict[str, Any] = {
                "level": lvl_name, "parent_group": str(grp),
                "n_nodes": int(n_nodes), "did_split": True,
                "gamma": best_cand["gamma"], "K": best_cand["k"],
                "small_frac": best_cand["small_frac"],
                "max_frac": best_cand["max_frac"],
                "min_margin": best_cand["min_margin"],
                "mean_margin": best_cand["mean_margin"],
                "constraints_ok": best_cand["constraints_ok"],
                "ari_stability": float(stability_ari(subg, best_cand["gamma"], iterations, [seed, seed + 1, seed + 2])),
                "coherence": float(coh),
                "within_mean": float(w_in), "between_mean": float(w_bt),
                "within_minus_between": float(w_in - w_bt),
            }
            if not best_cand["constraints_ok"]:
                failed = []
                if best_cand["max_frac"] > 0.9:
                    failed.append("mega_cluster")
                if best_cand["min_margin"] <= 0:
                    failed.append("negative_margin")
                diag_entry["constraints_failed"] = failed

            run_diag["groups"].append(diag_entry)

            print(
                f"[{lvl_name} grp={str(grp)[:40]}] gamma={best_cand['gamma']:.6f} K={best_cand['k']} "
                f"margin={best_cand['mean_margin']:+.3f} mega_frac={best_cand['max_frac']:.2f}",
                flush=True,
            )

            for local_i, idx in enumerate(rows):
                assign.at[idx, lvl_name] = f"{grp}|{lvl_name}_{mem[local_i]}"

        # Level rollup
        sizes = assign.groupby(lvl_name)["page"].size().values
        if len(sizes):
            print(
                f"[{lvl_name}] total_clusters={len(sizes)}  min={sizes.min()}  "
                f"median={int(np.median(sizes))}  max={sizes.max()}",
                flush=True,
            )

        grows = [g for g in run_diag["groups"] if g["level"] == lvl_name and g.get("did_split")]
        lvl_summary: Dict[str, Any] = {
            "n_clusters": int(len(sizes)),
            "total_target": f"{target_min}-{target_max}",
            "min_cluster": int(sizes.min()) if len(sizes) else 0,
            "median_cluster": float(np.median(sizes)) if len(sizes) else 0.0,
            "p90_cluster": float(np.percentile(sizes, 90)) if len(sizes) else 0.0,
            "max_cluster": int(sizes.max()) if len(sizes) else 0,
            "n_groups_total": int(len(groups)),
            "n_groups_split": int(len(grows)),
        }
        if grows:
            lvl_summary.update({
                "gamma_mean": float(np.mean([g["gamma"] for g in grows if g["gamma"] is not None])),
                "ari_mean": float(np.mean([g["ari_stability"] for g in grows if g.get("ari_stability") is not None])),
                "coherence_mean": float(np.mean([g["coherence"] for g in grows if g.get("coherence") is not None])),
            })
        run_diag["levels"][lvl_name] = lvl_summary

    if "__ALL__" in assign.columns:
        assign.drop(columns=["__ALL__"], inplace=True)

    return assign, run_diag


# ----------------------------
# Strategy 2: Margin-Threshold Guardrail with Semantic Pre-split
# ----------------------------

def strategy_margin_adaptive(
    g_core: ig.Graph,
    emb_core: np.ndarray,
    core_vids_full: List[int],
    titles_full: List[str],
    level_cols: List[str],
    cfgs: List[LevelConfig],
    iterations: int,
    seed: int,
    margin_weight: float = 10.0,
    small_frac_weight: float = 1.0,
    margin_threshold: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Margin-threshold guardrail: adaptive gamma search + semantic pre-split
    intervention at root/L1 if any per-cluster margin falls below margin_threshold.

    Returns (assign DataFrame, run_diag dict).
    """
    assert g_core.vcount() == emb_core.shape[0]

    run_diag: Dict[str, Any] = {"levels": {}, "groups": []}

    assign = pd.DataFrame({
        "vid_full": core_vids_full,
        "page": [titles_full[v] for v in core_vids_full],
    })

    core_pos = {vid: i for i, vid in enumerate(core_vids_full)}

    for lvl_i, lvl_name in enumerate(level_cols):
        cfg = cfgs[lvl_i]
        parent_col = None if lvl_i == 0 else level_cols[lvl_i - 1]

        print(f"\n{'=' * 98}", flush=True)
        print(f"[LEVEL {lvl_name}] Margin-adaptive (threshold={margin_threshold:.4f})", flush=True)
        print(f"{'=' * 98}", flush=True)

        if parent_col is None:
            assign["__ALL__"] = "__ALL__"
            parent_col = "__ALL__"
            groups = ["__ALL__"]
        else:
            groups = assign[parent_col].astype(str).unique().tolist()

        assign[lvl_name] = None

        for grp in tqdm_force(groups, desc=f"Splitting {parent_col} -> {lvl_name}", unit="group"):
            rows = assign.index[assign[parent_col].astype(str) == str(grp)].tolist()
            n_nodes = len(rows)

            if n_nodes <= int(cfg.split_if_gt) or n_nodes < 2 * int(cfg.min_size):
                for idx in rows:
                    assign.at[idx, lvl_name] = f"{grp}|{lvl_name}_0"
                run_diag["groups"].append({
                    "level": lvl_name, "parent_group": str(grp),
                    "n_nodes": int(n_nodes), "did_split": False,
                    "gamma": None, "K": 1,
                    "coherence": None, "within_mean": None, "between_mean": None,
                    "within_minus_between": None, "ari_stability": None,
                    "small_frac": None,
                })
                continue

            sub_nodes = [core_pos[int(assign.at[idx, "vid_full"])] for idx in rows]
            subg = g_core.subgraph(sub_nodes)
            sub_emb = emb_core[sub_nodes]

            # Phase 1: normal adaptive gamma search
            res = adaptive_gamma_search(
                subg, kmin=cfg.kmin, kmax=cfg.kmax, min_size=cfg.min_size,
                iterations=iterations, seed=seed, emb=sub_emb,
                margin_weight=margin_weight, small_frac_weight=small_frac_weight,
                verbose_prefix=f"[{lvl_name} grp={str(grp)[:40]}] ",
            )
            mem = leiden_membership(subg, res.gamma, iterations, seed)

            # Phase 2: check per-cluster margins — only re-split BAD clusters
            intervention_applied = False
            margin_before = None
            intervention_diag = {}

            if len(set(mem)) >= 2:
                margins_df = compute_per_cluster_margins_inline(sub_emb, mem)
                min_margin = float(margins_df["margin"].min())
                margin_before = float(margins_df["margin"].mean())

                needs_intervention = min_margin < margin_threshold
                # Only intervene at root and L1 levels
                is_deep_level = (lvl_i >= len(level_cols) - 1)

                if needs_intervention and not is_deep_level:
                    bad_cids = margins_df.loc[
                        margins_df["margin"] < margin_threshold, "cluster_id"
                    ].tolist()
                    n_bad = len(bad_cids)
                    n_good = len(set(mem)) - n_bad
                    print(
                        f"  [INTERVENTION] min_margin={min_margin:.4f} < threshold={margin_threshold:.4f}, "
                        f"re-splitting {n_bad} bad cluster(s) (keeping {n_good} good)",
                        flush=True,
                    )

                    mem_arr = np.array(mem, dtype=int)
                    bad_cid_set = set(int(c) for c in bad_cids)
                    # Relabel good clusters to contiguous starting at 0
                    new_mem = np.full(len(mem_arr), -1, dtype=int)
                    label_offset = 0
                    for cid in sorted(set(mem)):
                        if cid not in bad_cid_set:
                            new_mem[mem_arr == cid] = label_offset
                            label_offset += 1

                    # For each bad cluster, run Leiden on its induced subgraph
                    n_refined = 0
                    for bad_cid in sorted(bad_cid_set):
                        bad_mask = (mem_arr == int(bad_cid))
                        bad_indices = np.where(bad_mask)[0]

                        if len(bad_indices) < 2 * cfg.min_size:
                            # Too small — keep as single cluster
                            new_mem[bad_mask] = label_offset
                            label_offset += 1
                            continue

                        bad_subg = subg.subgraph(bad_indices.tolist())
                        bad_emb = sub_emb[bad_indices]

                        if bad_subg.ecount() == 0:
                            new_mem[bad_mask] = label_offset
                            label_offset += 1
                            continue

                        # Re-cluster just this bad cluster
                        b_res = adaptive_gamma_search(
                            bad_subg,
                            kmin=max(2, cfg.kmin),
                            kmax=cfg.kmax,
                            min_size=cfg.min_size,
                            iterations=iterations, seed=seed,
                            emb=bad_emb,
                            margin_weight=margin_weight,
                            small_frac_weight=small_frac_weight,
                            verbose_prefix=f"  [re-split cid={bad_cid}] ",
                        )
                        b_mem = leiden_membership(bad_subg, b_res.gamma, iterations, seed)
                        for local_i, global_i in enumerate(bad_indices):
                            new_mem[global_i] = b_mem[local_i] + label_offset
                        label_offset += int(max(b_mem)) + 1
                        n_refined += 1

                    mem = new_mem.tolist()
                    intervention_applied = True
                    intervention_diag = {
                        "targeted_resplit": True,
                        "n_bad_clusters": n_bad,
                        "n_good_kept": n_good,
                        "n_refined": n_refined,
                        "total_clusters_after": len(set(mem)),
                    }

                    # Recompute margins after intervention
                    if len(set(mem)) >= 2:
                        margins_df_after = compute_per_cluster_margins_inline(sub_emb, mem)
                        intervention_diag["margin_before"] = margin_before
                        intervention_diag["margin_after"] = float(margins_df_after["margin"].mean())

            # Diagnostics
            coh = cluster_text_coherence(sub_emb, mem)
            w_in, w_bt = within_between_similarity(sub_emb, [str(m) for m in mem], rng_seed=seed)

            diag_entry: Dict[str, Any] = {
                "level": lvl_name, "parent_group": str(grp),
                "n_nodes": int(n_nodes), "did_split": True,
                "gamma": float(res.gamma) if not intervention_applied else None,
                "K": int(len(set(mem))),
                "small_frac": float(res.small_frac),
                "ari_stability": float(res.ari_stability),
                "coherence": float(coh),
                "within_mean": float(w_in), "between_mean": float(w_bt),
                "within_minus_between": float(w_in - w_bt),
            }
            if intervention_applied:
                diag_entry["semantic_presplit"] = True
                diag_entry.update(intervention_diag)
            run_diag["groups"].append(diag_entry)

            print(
                f"[{lvl_name} grp={str(grp)[:40]}] K={len(set(mem))} coherence={coh:.3f} "
                f"within={w_in:.3f} between={w_bt:.3f}"
                + (f" (semantic pre-split)" if intervention_applied else ""),
                flush=True,
            )

            for local_i, idx in enumerate(rows):
                assign.at[idx, lvl_name] = f"{grp}|{lvl_name}_{mem[local_i]}"

        # Level rollup
        sizes = assign.groupby(lvl_name)["page"].size().values
        if len(sizes):
            print(
                f"[{lvl_name}] clusters={len(sizes)}  min={sizes.min()}  "
                f"median={int(np.median(sizes))}  max={sizes.max()}",
                flush=True,
            )

        grows = [g for g in run_diag["groups"] if g["level"] == lvl_name and g.get("did_split")]
        lvl_summary: Dict[str, Any] = {
            "n_clusters": int(len(sizes)),
            "min_cluster": int(sizes.min()) if len(sizes) else 0,
            "median_cluster": float(np.median(sizes)) if len(sizes) else 0.0,
            "p90_cluster": float(np.percentile(sizes, 90)) if len(sizes) else 0.0,
            "max_cluster": int(sizes.max()) if len(sizes) else 0,
            "n_groups_total": int(len(groups)),
            "n_groups_split": int(len(grows)),
        }
        if grows:
            lvl_summary.update({
                "gamma_mean": float(np.mean([g["gamma"] for g in grows if g.get("gamma") is not None])),
                "ari_mean": float(np.mean([g["ari_stability"] for g in grows if g.get("ari_stability") is not None])),
                "coherence_mean": float(np.mean([g["coherence"] for g in grows if g.get("coherence") is not None])),
                "n_interventions": int(sum(1 for g in grows if g.get("semantic_presplit"))),
            })
        run_diag["levels"][lvl_name] = lvl_summary

    if "__ALL__" in assign.columns:
        assign.drop(columns=["__ALL__"], inplace=True)

    return assign, run_diag


# ----------------------------
# Strategy 3: Mega-Cluster Guardrail with Semantic Split
# ----------------------------

def strategy_mega_adaptive(
    g_core: ig.Graph,
    emb_core: np.ndarray,
    core_vids_full: List[int],
    titles_full: List[str],
    level_cols: List[str],
    cfgs: List[LevelConfig],
    iterations: int,
    seed: int,
    margin_weight: float = 10.0,
    small_frac_weight: float = 1.0,
    mega_pct: float = 0.9,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Mega-cluster guardrail: normal adaptive partition + semantic refinement
    of any mega cluster (child holding > mega_pct of parent).

    At root/L1: mega clusters are semantically split then re-Leiden'd.
    At L2: semantic split is final (no re-Leiden).

    Returns (assign DataFrame, run_diag dict).
    """
    assert g_core.vcount() == emb_core.shape[0]

    run_diag: Dict[str, Any] = {"levels": {}, "groups": []}

    assign = pd.DataFrame({
        "vid_full": core_vids_full,
        "page": [titles_full[v] for v in core_vids_full],
    })

    core_pos = {vid: i for i, vid in enumerate(core_vids_full)}

    for lvl_i, lvl_name in enumerate(level_cols):
        cfg = cfgs[lvl_i]
        parent_col = None if lvl_i == 0 else level_cols[lvl_i - 1]

        print(f"\n{'=' * 98}", flush=True)
        print(f"[LEVEL {lvl_name}] Mega-adaptive (mega_pct={mega_pct:.2f})", flush=True)
        print(f"{'=' * 98}", flush=True)

        if parent_col is None:
            assign["__ALL__"] = "__ALL__"
            parent_col = "__ALL__"
            groups = ["__ALL__"]
        else:
            groups = assign[parent_col].astype(str).unique().tolist()

        assign[lvl_name] = None

        for grp in tqdm_force(groups, desc=f"Splitting {parent_col} -> {lvl_name}", unit="group"):
            rows = assign.index[assign[parent_col].astype(str) == str(grp)].tolist()
            n_nodes = len(rows)

            if n_nodes <= int(cfg.split_if_gt) or n_nodes < 2 * int(cfg.min_size):
                for idx in rows:
                    assign.at[idx, lvl_name] = f"{grp}|{lvl_name}_0"
                run_diag["groups"].append({
                    "level": lvl_name, "parent_group": str(grp),
                    "n_nodes": int(n_nodes), "did_split": False,
                    "gamma": None, "K": 1,
                    "coherence": None, "within_mean": None, "between_mean": None,
                    "within_minus_between": None, "ari_stability": None,
                    "small_frac": None,
                })
                continue

            sub_nodes = [core_pos[int(assign.at[idx, "vid_full"])] for idx in rows]
            subg = g_core.subgraph(sub_nodes)
            sub_emb = emb_core[sub_nodes]

            # Phase 1: normal adaptive gamma search
            res = adaptive_gamma_search(
                subg, kmin=cfg.kmin, kmax=cfg.kmax, min_size=cfg.min_size,
                iterations=iterations, seed=seed, emb=sub_emb,
                margin_weight=margin_weight, small_frac_weight=small_frac_weight,
                verbose_prefix=f"[{lvl_name} grp={str(grp)[:40]}] ",
            )
            mem = leiden_membership(subg, res.gamma, iterations, seed)

            # Phase 2: detect mega clusters
            mega_ids = detect_mega_clusters(mem, n_nodes, pct_threshold=mega_pct)
            mega_fix_applied = False
            mega_before = None
            n_mega_refined = 0
            semantic_buckets_total = 0

            if mega_ids:
                mem_arr = np.array(mem, dtype=int)
                mega_max_size = max(int(np.sum(mem_arr == mid)) for mid in mega_ids)
                mega_before = {"max_size": mega_max_size, "frac": mega_max_size / n_nodes}

                is_deep_level = (lvl_i >= len(level_cols) - 1)

                for mega_cid in mega_ids:
                    mega_mask = (mem_arr == mega_cid)
                    mega_indices = np.where(mega_mask)[0]
                    mega_emb = sub_emb[mega_indices]

                    # Semantic split of mega cluster
                    buckets = semantic_split(
                        mega_emb, seed=seed, s_min=2, s_max=8,
                        min_bucket_size=max(10, cfg.min_size),
                    )
                    semantic_buckets_total += len(buckets)
                    n_buckets_mega = len(buckets)

                    if n_buckets_mega <= 1:
                        continue  # can't split further

                    mega_fix_applied = True
                    n_mega_refined += 1

                    # Semantic split is final at all levels (no re-Leiden)
                    new_label_base = int(mem_arr.max()) + 1
                    for bi, bucket_idx in enumerate(buckets):
                        for local_i in bucket_idx:
                            global_i = mega_indices[local_i]
                            mem_arr[global_i] = new_label_base + bi

                # Relabel to contiguous
                unique_labels = np.unique(mem_arr)
                remap = {old: new for new, old in enumerate(unique_labels)}
                mem = [remap[m] for m in mem_arr]
            # end mega refinement

            # Diagnostics
            coh = cluster_text_coherence(sub_emb, mem)
            w_in, w_bt = within_between_similarity(sub_emb, [str(m) for m in mem], rng_seed=seed)

            diag_entry: Dict[str, Any] = {
                "level": lvl_name, "parent_group": str(grp),
                "n_nodes": int(n_nodes), "did_split": True,
                "gamma": float(res.gamma),
                "K": int(len(set(mem))),
                "small_frac": float(res.small_frac),
                "ari_stability": float(res.ari_stability),
                "coherence": float(coh),
                "within_mean": float(w_in), "between_mean": float(w_bt),
                "within_minus_between": float(w_in - w_bt),
            }
            if mega_fix_applied:
                mem_arr_final = np.array(mem, dtype=int)
                mega_max_after = int(np.max(np.bincount(mem_arr_final)))
                diag_entry["mega_fix_applied"] = True
                diag_entry["mega_before"] = mega_before
                diag_entry["mega_after"] = {"max_size": mega_max_after, "frac": mega_max_after / n_nodes}
                diag_entry["n_mega_clusters_refined"] = n_mega_refined
                diag_entry["semantic_buckets_total"] = semantic_buckets_total

            run_diag["groups"].append(diag_entry)

            print(
                f"[{lvl_name} grp={str(grp)[:40]}] K={len(set(mem))} coherence={coh:.3f} "
                f"within={w_in:.3f} between={w_bt:.3f}"
                + (f" (mega-fix: {n_mega_refined} refined)" if mega_fix_applied else ""),
                flush=True,
            )

            for local_i, idx in enumerate(rows):
                assign.at[idx, lvl_name] = f"{grp}|{lvl_name}_{mem[local_i]}"

        # Level rollup
        sizes = assign.groupby(lvl_name)["page"].size().values
        if len(sizes):
            print(
                f"[{lvl_name}] clusters={len(sizes)}  min={sizes.min()}  "
                f"median={int(np.median(sizes))}  max={sizes.max()}",
                flush=True,
            )

        grows = [g for g in run_diag["groups"] if g["level"] == lvl_name and g.get("did_split")]
        lvl_summary: Dict[str, Any] = {
            "n_clusters": int(len(sizes)),
            "min_cluster": int(sizes.min()) if len(sizes) else 0,
            "median_cluster": float(np.median(sizes)) if len(sizes) else 0.0,
            "p90_cluster": float(np.percentile(sizes, 90)) if len(sizes) else 0.0,
            "max_cluster": int(sizes.max()) if len(sizes) else 0,
            "n_groups_total": int(len(groups)),
            "n_groups_split": int(len(grows)),
        }
        if grows:
            lvl_summary.update({
                "gamma_mean": float(np.mean([g["gamma"] for g in grows if g.get("gamma") is not None])),
                "ari_mean": float(np.mean([g["ari_stability"] for g in grows if g.get("ari_stability") is not None])),
                "coherence_mean": float(np.mean([g["coherence"] for g in grows if g.get("coherence") is not None])),
                "n_mega_fixes": int(sum(1 for g in grows if g.get("mega_fix_applied"))),
            })
        run_diag["levels"][lvl_name] = lvl_summary

    if "__ALL__" in assign.columns:
        assign.drop(columns=["__ALL__"], inplace=True)

    return assign, run_diag


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True, help="Input CSV.")
    ap.add_argument("--out-dir", default=None, help="Output directory (default derived from input).")

    ap.add_argument("--title-col", default="page_title")
    ap.add_argument("--links-col", default="linked_wiki_titles_json")
    ap.add_argument("--desc-col", default="description")
    ap.add_argument("--given-root-col", default="root_technology")


    # preprocessing core graph
  
    ap.add_argument("--min-component-size", type=int, default=5,
                    help="Components smaller than this go peripheral. Default=5")

    # hybrid weights
    ap.add_argument("--semantic-weight", type=float, default=0.65)
    ap.add_argument("--edge-weight", type=float, default=0.35)
    ap.add_argument("--tau-edge", type=float, default=0.20)
    ap.add_argument("--hub-penalty", action="store_true", default=False)

    # embeddings
    ap.add_argument("--embed-model", default="AI-Growth-Lab/PatentSBERTa")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # hierarchy
    ap.add_argument("--n-levels", type=int, default=2)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iterations", type=int, default=50)

    # adaptive targets (root + L1..Ln)
    ap.add_argument("--kmins", default="8,3,3",
                    help="Comma list of Kmin per level (root,L1..). Padded if shorter.")
    ap.add_argument("--kmaxs", default="12,10,9",
                    help="Comma list of Kmax per level (root,L1..). Padded if shorter.")
    ap.add_argument("--min-sizes", default="2000,30,10",
                    help="Comma list of min_size per level (root,L1..). Padded if shorter.")
    ap.add_argument("--split-sizes", default="6000,200,100",
                    help="Comma list of split_if_gt per level (root,L1..). Padded if shorter.")

    # scoring weights
    ap.add_argument("--margin-weight", type=float, default=10.0,
                    help="Weight for semantic margin in gamma search (default: 10.0)")
    ap.add_argument("--small-frac-weight", type=float, default=1.0,
                    help="Weight for small fraction in gamma search (default: 1.0)")

    # strategy
    ap.add_argument("--strategy", default="adaptive",
                    choices=["adaptive", "simple", "margin_adaptive", "mega_adaptive"],
                    help="Clustering strategy (default: adaptive)")
    ap.add_argument("--margin-threshold", type=float, default=0.0,
                    help="Min per-cluster margin for margin_adaptive strategy (default: 0.0)")
    ap.add_argument("--mega-pct", type=float, default=0.9,
                    help="Mega-cluster percentage threshold for mega_adaptive (default: 0.9)")
    ap.add_argument("--mega-abs", type=int, default=200,
                    help="Mega-cluster absolute size threshold for mega_adaptive (default: 200)")

    # adaptive tau (percentile-based edge pruning)
    ap.add_argument("--prune-pct", type=float, default=0.0,
                    help="Percentile-based edge pruning (0=disabled). "
                         "Prune bottom X%% of edges by weight, globally and per-subgraph.")

    # plotting
    ap.add_argument("--plot", action="store_true")


    # labeling
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--openrouter-model", default="google/gemini-3-flash-preview")
    ap.add_argument("--sleep-s", type=float, default=1.5)
    ap.add_argument("--max-pages-in-prompt", type=int, default=40)
    ap.add_argument("--user-agent", default="wiki-link-extractor/1.0 @srony@356576@gmail.com (research use)")

    return ap.parse_args()



# ----------------------------
# Run diagnostics writers (JSON + flat CSV for grid runs)
# ----------------------------
def make_run_id(args: argparse.Namespace) -> str:
    payload = {
        "semantic_weight": args.semantic_weight,
        "edge_weight": args.edge_weight,
        "tau_edge": args.tau_edge,
        "hub_penalty": args.hub_penalty,
        "embed_model": args.embed_model,
        "kmins": args.kmins,
        "kmaxs": args.kmaxs,
        "min_sizes": args.min_sizes,
        "split_sizes": args.split_sizes,
        "seed": args.seed,
        "iterations": args.iterations,
        "n_levels": args.n_levels,
        "min_component_size": args.min_component_size,
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def write_run_diagnostics(out_dir: Path, run_diag: dict, args: argparse.Namespace, suffix: str) -> Path:
    ensure_dir(out_dir)

    run_id = run_diag.get("run_id") or make_run_id(args)
    ts = run_diag.get("timestamp_utc") or (datetime.utcnow().isoformat() + "Z")

    run_diag["run_id"] = run_id
    run_diag["timestamp_utc"] = ts

    path = out_dir / f"run_diagnostics{suffix}.json"
    path.write_text(json.dumps(run_diag, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def flatten_run_summary(run_diag: dict) -> pd.DataFrame:
    rows = []
    base = {
        "run_id": run_diag.get("run_id"),
        "timestamp_utc": run_diag.get("timestamp_utc"),
    }
    params = run_diag.get("params", {})
    graph = run_diag.get("graph", {})

    for lvl, metrics in run_diag.get("levels", {}).items():
        row = {}
        row.update(base)
        for k, v in params.items():
            row[f"param__{k}"] = v
        for k, v in graph.items():
            # keep only simple values in the flat table
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[f"graph__{k}"] = v
        row["level"] = lvl
        for k, v in metrics.items():
            row[k] = v
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------
# Infrastructure: setup + finalize (extracted from main)
# ----------------------------

def setup_clustering_infrastructure(args: argparse.Namespace) -> dict:
    """Load data, build graph, embed, reweight, core/peripheral split, build configs.

    Returns a context dict with everything needed by strategy dispatch + finalize.
    """
    strategy = getattr(args, "strategy", "adaptive")
    suffix = f"_{args.desc_col}_{strategy}"

    print("[INFO] Starting with arguments:", flush=True)
    for k, v in vars(args).items():
        print(f"  {k}: {v}", flush=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir_from_input(input_path)
    ensure_dir(out_dir)

    vis_dir = get_visualisations_dir()
    ensure_dir(vis_dir)

    print(f"[INFO] input: {input_path}", flush=True)
    print(f"[INFO] out_dir: {out_dir}", flush=True)
    print(f"[INFO] vis_dir: {vis_dir}", flush=True)
    print(f"[INFO] strategy: {strategy}", flush=True)

    # Load data
    df = pd.read_csv(input_path)
    n_input_rows = len(df)   # original row count, before the denoise filter below (for run_diag)
    df[args.title_col] = df[args.title_col].astype(str)
    if args.desc_col not in df.columns:
        df[args.desc_col] = ""
    print(f"[INFO] Loaded {len(df):,} pages", flush=True)

    # Denoise: remove known noise pages before clustering (mode-aware)
    mode = getattr(args, "mode", "digital")
    noise_filename = "noise_pages_green_to_remove.txt" if mode == "green" else "noise_pages_all.txt"
    noise_path = input_path.parent / noise_filename
    if noise_path.exists():
        noise_pages = set(
            line.strip() for line in noise_path.read_text().splitlines() if line.strip()
        )
        before = len(df)
        df = df[~df[args.title_col].isin(noise_pages)].reset_index(drop=True)
        removed = before - len(df)
        print(f"[DENOISE] Removed {removed} noise pages ({len(noise_pages)} in list, {len(df)} remain)", flush=True)

    # Build wiki-link graph (FULL)
    g_full, title_to_idx, titles_list = build_graph_from_wikilinks(
        df=df,
        title_col=args.title_col,
        links_col="linked_wiki_titles_json",
        max_links_per_page=500,
        symmetrize=True,
    )
    print(f"\n[GRAPH] full graph nodes={g_full.vcount():,} edges={g_full.ecount():,}", flush=True)

    # Embeddings
    desc_texts = [clean_description_text(x) for x in df[args.desc_col].fillna("").astype(str).tolist()]
    print(f"\n[EMBED] model={args.embed_model} device={args.device}", flush=True)
    emb_full = load_or_build_embeddings(
        desc_texts=desc_texts,
        input_path=input_path,
        out_dir=CACHE_ROOT,
        embed_model=args.embed_model,
        device=args.device,
        desc_col=args.desc_col,
    )

    # Optional isotropy correction
    if getattr(args, "whiten", False):
        print("[WHITEN] PCA whitening: demean + scale by 1/sqrt(eigenvalue) + re-normalize", flush=True)
        mean_vec = emb_full.mean(axis=0, keepdims=True)
        centered = emb_full - mean_vec
        # SVD: centered = U @ diag(S) @ Vt
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # Project into PCA space and scale by 1/sqrt(eigenvalue)
        # eigenvalue_i = S_i^2 / n, so 1/sqrt(eig_i) = sqrt(n) / S_i
        n = centered.shape[0]
        # Clip small singular values to avoid division by near-zero
        S_safe = np.maximum(S, 1e-6)
        # Transform: X_white = (X - mean) @ V @ diag(sqrt(n)/S)
        scale = np.sqrt(n) / S_safe
        whiten_n_dims = getattr(args, "whiten_n_dims", 0)
        if whiten_n_dims > 0:
            # Only whiten top-k dims, zero out rest
            scale[whiten_n_dims:] = 0.0
            print(f"[WHITEN] Keeping top {whiten_n_dims} dimensions", flush=True)
        emb_full = (centered @ Vt.T * scale).astype(np.float64)
        norms = np.linalg.norm(emb_full, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        emb_full = (emb_full / norms).astype(np.float32)
        print(f"[WHITEN] Output shape: {emb_full.shape}", flush=True)
    elif getattr(args, "demean", False):
        print("[DEMEAN] Subtracting mean vector and re-normalizing embeddings", flush=True)
        mean_vec = emb_full.mean(axis=0, keepdims=True)
        emb_full = emb_full - mean_vec
        norms = np.linalg.norm(emb_full, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        emb_full = (emb_full / norms).astype(np.float32)

    # Reweight edges
    print("\n[EDGE WEIGHTING] semantic+edge hybrid weighting", flush=True)
    stats_rew = reweight_edges_semantic(
        g=g_full,
        desc_emb=emb_full,
        semantic_weight=args.semantic_weight,
        edge_weight=args.edge_weight,
        tau_edge=args.tau_edge,
        hub_penalty=args.hub_penalty,
        prune_pct=getattr(args, "prune_pct", 0.0),
    )
    print("[EDGE WEIGHTING] Stats:", stats_rew, flush=True)

    # Core/peripheral split
    core_vids_full, peripheral_vids_full, stats_core = split_core_peripheral(
        g_full,
        min_component_size=args.min_component_size,
        remove_isolates=True,
    )
    print("\n[PREPROCESS] Core/peripheral split", flush=True)
    for k, v in stats_core.items():
        print(f"  {k}: {v}", flush=True)

    g_core = g_full.subgraph(core_vids_full)
    emb_core = emb_full[core_vids_full]
    print(f"\n[GRAPH] core graph nodes={g_core.vcount():,} edges={g_core.ecount():,}", flush=True)
    print(f"[GRAPH] density={g_core.density():.6f} avg_degree={np.mean(g_core.degree()):.2f}", flush=True)
    print(f"[GRAPH] 2E/N ={2*g_core.ecount()/max(g_core.vcount(),1):.2f}", flush=True)

    # Level configs
    n_levels = int(max(1, args.n_levels))
    total_levels = 1 + n_levels

    kmins = pad_list(parse_csv_ints(args.kmins) or [6, 5, 4], total_levels)
    kmaxs = pad_list(parse_csv_ints(args.kmaxs) or [10, 10, 12], total_levels)
    min_sizes_list = pad_list(parse_csv_ints(args.min_sizes) or [15, 10, 6], total_levels)
    split_sizes_list = pad_list(parse_csv_ints(args.split_sizes) or [300, 100, 25], total_levels)

    print("\n[PARAMS] Adaptive targets (root + L-levels)", flush=True)
    print(f"{'Level':<6} {'Kmin':>6} {'Kmax':>6} {'min_size':>10} {'split_if_gt':>12}", flush=True)
    print("-" * 44, flush=True)
    print(f"{'root':<6} {kmins[0]:>6} {kmaxs[0]:>6} {min_sizes_list[0]:>10} {split_sizes_list[0]:>12}", flush=True)
    for i in range(1, total_levels):
        print(f"{('L'+str(i)):<6} {kmins[i]:>6} {kmaxs[i]:>6} {min_sizes_list[i]:>10} {split_sizes_list[i]:>12}", flush=True)

    cfgs = [LevelConfig(kmin=kmins[i], kmax=kmaxs[i], min_size=min_sizes_list[i], split_if_gt=split_sizes_list[i])
            for i in range(total_levels)]

    level_cols = ["root"] + [f"L{i}" for i in range(1, n_levels + 1)]
    deepest_col = level_cols[-1]

    # Run diagnostics metadata
    run_diag_meta = {
        "run_id": make_run_id(args),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "params": {
            "input": str(input_path),
            "strategy": strategy,
            "semantic_weight": float(args.semantic_weight),
            "edge_weight": float(args.edge_weight),
            "tau_edge": float(args.tau_edge),
            "hub_penalty": bool(args.hub_penalty),
            "embed_model": str(args.embed_model),
            "device": str(args.device),
            "min_component_size": int(args.min_component_size),
            "seed": int(args.seed),
            "iterations": int(args.iterations),
            "n_levels": int(args.n_levels),
            "kmins": str(args.kmins),
            "kmaxs": str(args.kmaxs),
            "min_sizes": str(args.min_sizes),
            "split_sizes": str(args.split_sizes),
        },
        "preprocess": {
            "n_rows_input": int(n_input_rows),
            "n_rows_after_filter_dedup": int(len(df)),
            "n_noise_flagged": int(df["flag_noise_structural"].sum()) if "flag_noise_structural" in df.columns else None,
            "core_stats": stats_core,
        },
        "graph": {
            "full_nodes": int(g_full.vcount()),
            "core_nodes": int(g_core.vcount()),
            "core_edges": int(g_core.ecount()),
            "edge_reweight_stats": stats_rew,
        },
    }

    return {
        "input_path": input_path,
        "out_dir": out_dir,
        "vis_dir": vis_dir,
        "suffix": suffix,
        "df": df,
        "g_full": g_full,
        "g_core": g_core,
        "emb_full": emb_full,
        "emb_core": emb_core,
        "core_vids_full": core_vids_full,
        "peripheral_vids_full": peripheral_vids_full,
        "titles_list": titles_list,
        "stats_core": stats_core,
        "stats_rew": stats_rew,
        "level_cols": level_cols,
        "cfgs": cfgs,
        "n_levels": n_levels,
        "deepest_col": deepest_col,
        "run_diag_meta": run_diag_meta,
    }


def finalize_clustering_outputs(
    ctx: dict,
    df_core_assign: pd.DataFrame,
    run_diag: dict,
    args: argparse.Namespace,
) -> None:
    """Peripheral attachment, CSV writes, plots, labeling, diagnostics JSON."""
    g_full = ctx["g_full"]
    g_core = ctx["g_core"]
    core_vids_full = ctx["core_vids_full"]
    peripheral_vids_full = ctx["peripheral_vids_full"]
    titles_list = ctx["titles_list"]
    level_cols = ctx["level_cols"]
    deepest_col = ctx["deepest_col"]
    n_levels = ctx["n_levels"]
    out_dir = ctx["out_dir"]
    vis_dir = ctx["vis_dir"]
    suffix = ctx["suffix"]
    df = ctx["df"]

    # Attach peripheral nodes
    attached_df = attach_peripheral_to_clusters(
        g_full=g_full,
        core_vids_full=core_vids_full,
        core_assign_df=df_core_assign,
        leaf_col=deepest_col,
        peripheral_vids_full=peripheral_vids_full,
        min_score_to_attach=0.0,
    )

    df_periph = pd.DataFrame({
        "vid_full": peripheral_vids_full,
        "page": [titles_list[v] for v in peripheral_vids_full],
    })
    df_periph = df_periph.merge(attached_df, on="vid_full", how="left")

    parent_cols = [c for c in level_cols if c != deepest_col]
    leaf_to_path_df = (
        df_core_assign
        .groupby(deepest_col, as_index=False)[parent_cols]
        .first()
    )
    leaf_to_path = leaf_to_path_df.set_index(deepest_col)[parent_cols].to_dict(orient="index")

    for lvl in level_cols:
        df_periph[lvl] = None

    for i, r in df_periph.iterrows():
        c = r.get("attached_cluster", None)
        if isinstance(c, str) and c in leaf_to_path:
            for lvl in parent_cols:
                df_periph.at[i, lvl] = leaf_to_path[c][lvl]
            df_periph.at[i, deepest_col] = c
        else:
            for lvl in level_cols:
                df_periph.at[i, lvl] = "Peripheral"

    df_final = pd.concat(
        [df_core_assign[level_cols + ["vid_full", "page"]], df_periph[level_cols + ["vid_full", "page"]]],
        ignore_index=True,
    )

    # Attach given-root for heatmap
    if args.given_root_col in df.columns:
        given_map = df.set_index(args.title_col)[args.given_root_col].to_dict()
        df_final["root_given"] = df_final["page"].map(given_map)
    else:
        df_final["root_given"] = None

    # Write outputs (raw IDs)
    out_assign = out_dir / f"page_assignments_L{n_levels}{suffix}.csv"
    df_final.to_csv(out_assign, index=False)
    print(f"\n[WRITE] {out_assign}", flush=True)

    out_long = out_dir / f"taxonomy_long_L{n_levels}{suffix}.csv"
    df_final[level_cols + ["page"] + (["root_given"] if "root_given" in df_final.columns else [])].to_csv(out_long, index=False)
    print(f"[WRITE] {out_long}", flush=True)

    grouped_out = out_dir / f"taxonomy_grouped_pages_L{n_levels}{suffix}.csv"
    df_grouped = (
        df_final[level_cols + ["page"] + (["root_given"] if "root_given" in df_final.columns else [])]
        .groupby(level_cols, dropna=False)
        .agg(
            pages=("page", lambda x: list(pd.unique(x.astype(str)))),
            root_given=("root_given", "first") if "root_given" in df_final.columns else ("page", lambda x: None),
        )
        .reset_index()
    )
    df_grouped[level_cols + ["pages", "root_given"]].to_csv(grouped_out, index=False)
    print(f"[WRITE] {grouped_out}", flush=True)

    # Plots: size distributions + heatmap
    if args.plot:
        sizes_html = viz_cluster_size_distributions(
            df_final, level_cols, vis_dir / f"cluster_size_distributions__L{n_levels}{suffix}.html"
        )
        print(f"[PLOT] {sizes_html}", flush=True)

        hm = viz_heatmap_given_vs_discovered(
            df_final=df_final,
            given_col="root_given",
            root_col="root",
            out_html=vis_dir / f"given_vs_discovered_root_heatmap__L{n_levels}{suffix}.html",
        )
        if hm is None:
            print("[PLOT] Heatmap skipped (no given root column or empty).", flush=True)
        else:
            print(f"[PLOT] {hm}", flush=True)

    # Labeling (OpenRouter) + labeled outputs
    labels_by_level: Dict[str, Dict[str, str]] = {}
    if args.label:
        api_key = read_openrouter_key()
        labels_dir = out_dir / f"labels__L{n_levels}{suffix}"
        ensure_dir(labels_dir)

        labels_by_level = label_clusters_from_df(
            df_assign=df_core_assign,
            level_cols=level_cols,
            out_dir=labels_dir,
            api_key=api_key,
            model=args.openrouter_model,
            user_agent=args.user_agent,
            sleep_s=args.sleep_s,
            max_pages_in_prompt=args.max_pages_in_prompt,
        )

        df_labeled = df_final.copy()
        print(f"\n[LABELING] df_labeled.head():\n{df_labeled.head()}")
        for lvl in level_cols:
            mp = labels_by_level.get(lvl, {})
            df_labeled[lvl] = df_labeled[lvl].astype(str).map(mp).fillna(df_labeled[lvl].astype(str))

        out_labeled = out_dir / f"taxonomy_long_labeled__L{n_levels}{suffix}.csv"
        cols_out = level_cols + ["page"]
        if "root_given" in df_labeled.columns:
            cols_out += ["root_given"]
        df_labeled[cols_out].to_csv(out_labeled, index=False)
        print(f"[WRITE] {out_labeled}", flush=True)

        grouped_out = out_dir / f"taxonomy_grouped_pages__L{n_levels}{suffix}.csv"
        df_grouped = (
            df_labeled[cols_out]
            .groupby(level_cols, dropna=False)
            .agg(
                pages=("page", lambda x: list(pd.unique(x.astype(str)))),
                root_given=("root_given", "first") if "root_given" in df_labeled.columns else ("page", lambda x: None),
            )
            .reset_index()
        )
        df_grouped[level_cols + ["pages", "root_given"]].to_csv(grouped_out, index=False)
        print(f"[WRITE] {grouped_out}", flush=True)

        labeled_assign = out_dir / f"page_assignments_labeled__L{n_levels}{suffix}.csv"
        df_labeled.to_csv(labeled_assign, index=False)
        print(f"[WRITE] {labeled_assign}", flush=True)

        df_final = df_labeled

    # Network plots per level
    if args.plot:
        core_set = set(int(v) for v in core_vids_full)
        df_plot_core = df_final[df_final["vid_full"].astype(int).isin(core_set)].copy()

        for lvl in level_cols:
            out_html = vis_dir / f"network_{lvl}__CORE_ONLY__L{n_levels}{suffix}.html"
            labels_map = labels_by_level.get(lvl) if (args.label and labels_by_level) else None

            p = viz_network_by_level(
                g_full=g_full,
                df_assign=df_plot_core,
                level_col=lvl,
                out_html=out_html,
                labels_map=labels_map,
                max_nodes=int(2500),
                seed=args.seed,
            )
            if p is None:
                print(f"[PLOT] network {lvl} skipped (missing vid_full or empty).", flush=True)
            else:
                print(f"[PLOT] {p}", flush=True)

        if args.label and labels_by_level:
            df_core_assign = df_core_assign.copy()
            for lvl in level_cols:
                mp = labels_by_level.get(lvl, {})
                if mp:
                    df_core_assign[lvl] = df_core_assign[lvl].astype(str).map(mp).fillna(df_core_assign[lvl].astype(str))

        df_core_assign = df_core_assign.reset_index(drop=True)
        df_core_assign["vid_core"] = np.arange(len(df_core_assign), dtype=int)
        out_html = vis_dir / f"network_L2_nodes_colored_by_root__L{n_levels}{suffix}.html"
        p = viz_L2_nodes_colored_by_root(
            g_core=g_core,
            df_core_assign=df_core_assign,
            out_html=out_html,
            max_nodes=300,
            min_edge_weight=0.0,
            seed=args.seed,
        )
        if p:
            print(f"[PLOT] {p}", flush=True)

        sun_path = vis_dir / f"sunburst_root_to_L2__L{n_levels}{suffix}.html"
        p = viz_taxonomy_sunburst_cluster(df_plot_core, sun_path)
        print(f"[PLOT] {p}", flush=True)

    # Write run diagnostics
    diag_json_path = write_run_diagnostics(out_dir, run_diag, args, suffix)
    print(f"[WRITE] {diag_json_path}", flush=True)

    summary_df = flatten_run_summary(run_diag)
    summary_path = out_dir / f"run_summary_by_level{suffix}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {summary_path}", flush=True)

    print("\n[DONE]", flush=True)
    print(f"Outputs in: {out_dir}", flush=True)
    print(f"Plots in:   {vis_dir}", flush=True)


# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parse_args()

    ctx = setup_clustering_infrastructure(args)

    strategy = getattr(args, "strategy", "adaptive")

    # Strategy dispatch
    common_kwargs = dict(
        g_core=ctx["g_core"],
        emb_core=ctx["emb_core"],
        core_vids_full=ctx["core_vids_full"],
        titles_full=ctx["titles_list"],
        level_cols=ctx["level_cols"],
        cfgs=ctx["cfgs"],
        iterations=args.iterations,
        seed=args.seed,
        margin_weight=args.margin_weight,
        small_frac_weight=args.small_frac_weight,
    )

    if strategy == "adaptive":
        df_core_assign, run_diag = recursive_split(
            **common_kwargs,
            mega_threshold=getattr(args, "mega_threshold", 0),
            local_tau=getattr(args, "local_tau", 0.0),
            prune_pct=getattr(args, "prune_pct", 0.0),
        )
    elif strategy == "simple":
        df_core_assign, run_diag = strategy_simple_sweep(**common_kwargs)
    elif strategy == "margin_adaptive":
        df_core_assign, run_diag = strategy_margin_adaptive(
            **common_kwargs,
            margin_threshold=args.margin_threshold,
        )
    elif strategy == "mega_adaptive":
        df_core_assign, run_diag = strategy_mega_adaptive(
            **common_kwargs,
            mega_pct=args.mega_pct,
        )
    else:
        raise SystemExit(f"Unknown strategy: {strategy}")

    # Post-hoc: collapse + resplit (collapse is inside posthoc_resplit)
    if getattr(args, "resplit", False):
        print("\n" + "=" * 98, flush=True)
        print("POST-HOC COLLAPSE + RESPLIT", flush=True)
        print("=" * 98, flush=True)

        df_core_assign, resplit_report = posthoc_resplit(
            assign=df_core_assign,
            g_core=ctx["g_core"],
            emb_core=ctx["emb_core"],
            core_vids_full=ctx["core_vids_full"],
            level_cols=ctx["level_cols"],
            cfgs=ctx["cfgs"],
            iterations=args.iterations,
            seed=args.seed,
            margin_weight=args.margin_weight,
            small_frac_weight=getattr(args, "small_frac_weight", 1.0),
            local_tau=getattr(args, "local_tau", 0.0),
            run_diag=run_diag,
            mega_frac=getattr(args, "mega_frac", 0.9),
            mega_min_size=getattr(args, "mega_min_size", 10),
            weak_margin=getattr(args, "weak_margin", 0.0),
            weak_min_size=getattr(args, "weak_min_size", 10),
        )

    # Merge metadata
    run_diag.update(ctx["run_diag_meta"])
    run_diag.setdefault("params", {})["strategy"] = strategy

    finalize_clustering_outputs(ctx, df_core_assign, run_diag, args)


# ------------------------------------------------------------------
# Public entry point (called by main.py CLI)
# ------------------------------------------------------------------

def run_adaptive_clustering(
    input_csv: Path,
    output_dir=None,
    *,
    title_col: str = "page_title",
    links_col: str = "linked_wiki_titles_json",
    desc_col: str = "description",
    given_root_col: str = "root_technology",
    min_component_size: int = 5,
    semantic_weight: float = 1.0,
    edge_weight: float = 0.0,
    tau_edge: float = 0.20,
    hub_penalty: bool = False,
    embed_model: str = "AI-Growth-Lab/PatentSBERTa",
    device=None,
    n_levels: int = 2,
    seed: int = 42,
    iterations: int = 50,
    kmins: str = "8,5,2",
    kmaxs: str = "12,8,6",
    min_sizes: str = "500,25,8",
    split_sizes: str = "3000,100,30",
    margin_weight: float = 10.0,
    small_frac_weight: float = 1.0,
    do_plot: bool = False,
    do_label: bool = False,
    openrouter_model: str = "openai/gpt-4o-mini",
    sleep_s: float = 1.5,
    max_pages_in_prompt: int = 40,
    user_agent: str = "wiki-link-extractor/1.0 (research use)",
    strategy: str = "adaptive",
    margin_threshold: float = 0.0,
    mega_pct: float = 0.9,
    mega_abs: int = 200,
    mega_threshold: int = 0,
    local_tau: float = 0.0,
    resplit: bool = False,
    demean: bool = False,
    whiten: bool = False,
    whiten_n_dims: int = 0,
    prune_pct: float = 0.0,
    mega_frac: float = 0.9,
    mega_min_size: int = 10,
    weak_margin: float = 0.0,
    weak_min_size: int = 10,
):
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if output_dir is None:
        output_dir = default_out_dir_from_input(input_csv)
    else:
        output_dir = Path(output_dir)

    ensure_dir(output_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    args = argparse.Namespace(
        input=str(input_csv),
        out_dir=str(output_dir),
        title_col=title_col,
        links_col=links_col,
        desc_col=desc_col,
        given_root_col=given_root_col,
        min_component_size=min_component_size,
        semantic_weight=semantic_weight,
        edge_weight=edge_weight,
        tau_edge=tau_edge,
        hub_penalty=hub_penalty,
        embed_model=embed_model,
        device=device,
        n_levels=n_levels,
        seed=seed,
        iterations=iterations,
        kmins=kmins,
        kmaxs=kmaxs,
        min_sizes=min_sizes,
        split_sizes=split_sizes,
        margin_weight=margin_weight,
        small_frac_weight=small_frac_weight,
        plot=do_plot,
        label=do_label,
        openrouter_model=openrouter_model,
        sleep_s=sleep_s,
        max_pages_in_prompt=max_pages_in_prompt,
        user_agent=user_agent,
        strategy=strategy,
        margin_threshold=margin_threshold,
        mega_pct=mega_pct,
        mega_abs=mega_abs,
        mega_threshold=mega_threshold,
        local_tau=local_tau,
        resplit=resplit,
        demean=demean,
        whiten=whiten,
        whiten_n_dims=whiten_n_dims,
        prune_pct=prune_pct,
        mega_frac=mega_frac,
        mega_min_size=mega_min_size,
        weak_margin=weak_margin,
        weak_min_size=weak_min_size,
    )

    global parse_args
    _orig = parse_args
    parse_args = lambda: args
    try:
        main()
    finally:
        parse_args = _orig

    return {"output_dir": str(output_dir), "status": "ok"}
