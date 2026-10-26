"""
p2_taxonomy/visualisations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
P2-specific visualisation helpers for patent topic and taxonomy outputs.

Provides:
- ``viz_prevalence``            -- horizontal bar chart of topic prevalence
- ``viz_topic_map``             -- BERTopic interactive topic map
- ``viz_topic_cpc``             -- CPC cross-tabulation heatmap
- ``viz_sunburst``              -- Plotly sunburst of taxonomy hierarchy
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from shared.utils import ensure_dir


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _short_label(s, max_len: int = 40) -> str:
    s = str(s)
    return s if len(s) <= max_len else s[:max_len] + "..."


def _list_part_files(parts_dir: str) -> List[Path]:
    files = sorted(Path(parts_dir).glob("part_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet parts found in: {parts_dir}")
    return files


# ------------------------------------------------------------------
# A) Prevalence bar chart
# ------------------------------------------------------------------

def viz_prevalence(
    primary_df: pd.DataFrame,
    out_path: str,
    title: str,
    level_col: str,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of topic prevalence (top *top_n* topics)."""
    import matplotlib.pyplot as plt

    vc = primary_df[level_col].value_counts().head(top_n)
    plot_df = vc[::-1]

    plt.figure(figsize=(10, max(5, 0.38 * len(plot_df))))
    labels = [_short_label(x, 30) for x in plot_df.index.astype(str)]
    plt.barh(labels, plot_df.values)
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()

    ensure_dir(str(Path(out_path).parent))
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[viz A] Saved: {out_path}")


# ------------------------------------------------------------------
# D) BERTopic topic map
# ------------------------------------------------------------------

def viz_topic_map(
    model_leaf: "BERTopic",
    out_path: str,
    title: str,
) -> None:
    """Interactive BERTopic topic-distance map (HTML)."""
    try:
        fig = model_leaf.visualize_topics()
        fig.update_layout(title=title)
        ensure_dir(str(Path(out_path).parent))
        fig.write_html(out_path)
        print(f"[viz D] Saved: {out_path}")
    except Exception as exc:
        print(f"[viz D] Skipped: {exc}")


# ------------------------------------------------------------------
# E) CPC cross-tabulation heatmap
# ------------------------------------------------------------------

def viz_topic_cpc(
    parts_dir: str,
    primary_df: pd.DataFrame,
    leaf_layer: int,
    out_path: str,
    title: str,
    top_topics: int = 20,
    top_cpcs: int = 20,
) -> None:
    """Heatmap showing CPC code distribution across leaf topics."""
    import matplotlib.pyplot as plt

    leaf_col = f"L{leaf_layer}_label"
    if leaf_col not in primary_df.columns:
        print("[viz E] Missing leaf label column; skipping.")
        return

    top_leaf = primary_df[leaf_col].value_counts().head(top_topics).index.tolist()
    df = primary_df[primary_df[leaf_col].isin(top_leaf)][["appln_id", leaf_col]]

    # Only these patents survive the inner-merge below, so drop the rest before the
    # (expensive) split+explode instead of exploding the whole corpus and discarding it.
    keep_ids = set(df["appln_id"])

    parts = _list_part_files(parts_dir)
    cpc_rows = []
    for p in tqdm(parts, desc="[viz E] load CPC parts"):
        tmp = pd.read_parquet(p, columns=["appln_id", "cpc_coarse_list"])
        tmp = tmp[tmp["appln_id"].isin(keep_ids)]
        tmp["cpc"] = tmp["cpc_coarse_list"].fillna("").apply(
            lambda s: s.split("|") if s else [],
        )
        tmp = tmp.explode("cpc")
        tmp["cpc"] = tmp["cpc"].astype(str).str.strip()
        tmp = tmp[tmp["cpc"] != ""]
        cpc_rows.append(tmp[["appln_id", "cpc"]])

    cpc_df = (
        pd.concat(cpc_rows, ignore_index=True)
        if cpc_rows
        else pd.DataFrame(columns=["appln_id", "cpc"])
    )
    merged = df.merge(cpc_df, on="appln_id", how="inner")
    if merged.empty:
        print("[viz E] No merged CPC data; skipping.")
        return

    top_cpc = merged["cpc"].value_counts().head(top_cpcs).index.tolist()
    merged = merged[merged["cpc"].isin(top_cpc)]

    ctab = pd.crosstab(merged[leaf_col], merged["cpc"])
    share = ctab.div(ctab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    plt.figure(figsize=(10, max(6, 0.38 * len(share))))
    plt.imshow(share.values, aspect="auto")
    y_labels = [_short_label(x, 40) for x in share.index.astype(str)]
    plt.yticks(range(len(y_labels)), y_labels, fontsize=7)
    plt.xticks(range(len(share.columns)), share.columns, rotation=45, ha="right")
    plt.colorbar(label="Share of CPC within topic")
    plt.title(title)
    plt.tight_layout()

    ensure_dir(str(Path(out_path).parent))
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[viz E] Saved: {out_path}")


# ------------------------------------------------------------------
# Sunburst
# ------------------------------------------------------------------

def viz_sunburst(
    taxonomy_df: pd.DataFrame,
    out_path: str,
    title: str,
) -> None:
    """Plotly sunburst visualisation of the taxonomy hierarchy."""
    try:
        import plotly.express as px

        fig = px.sunburst(taxonomy_df, path=list(taxonomy_df.columns), title=title)
        ensure_dir(str(Path(out_path).parent))
        fig.write_html(out_path)
        print(f"[viz Sunburst] Saved: {out_path}")
    except Exception as exc:
        print(f"[viz Sunburst] Skipped: {exc}")


# ------------------------------------------------------------------
# Leiden cluster visualisations
# ------------------------------------------------------------------

def viz_heatmap_given_vs_discovered(
    df_final: pd.DataFrame,
    given_col: str,
    root_col: str,
    out_html: Union[Path, str],
) -> Optional[Path]:
    """Heatmap of given root_technology vs discovered root cluster (page counts)."""
    import plotly.graph_objects as go

    out_html = Path(out_html)
    if given_col not in df_final.columns:
        return None
    if df_final[given_col].isna().all():
        return None

    ct = pd.crosstab(df_final[given_col].astype(str), df_final[root_col].astype(str))
    fig = go.Figure(data=go.Heatmap(z=ct.values, x=ct.columns.astype(str), y=ct.index.astype(str)))
    fig.update_layout(
        title="Given root_technology vs discovered root (page counts)",
        xaxis_title="discovered root",
        yaxis_title="given root_technology",
        height=900,
        margin=dict(l=140, r=20, t=50, b=140),
    )
    ensure_dir(str(out_html.parent))
    fig.write_html(str(out_html))
    print(f"[viz cluster heatmap] Saved: {out_html}")
    return out_html


def viz_cluster_size_distributions(
    df_assign: pd.DataFrame,
    level_cols: List[str],
    out_html: Union[Path, str],
) -> Path:
    """Histogram of cluster size distributions per level."""
    import plotly.graph_objects as go

    out_html = Path(out_html)
    fig = go.Figure()
    for lvl in level_cols:
        sizes = df_assign.groupby(lvl)["page"].size().values
        if len(sizes) == 0:
            continue
        fig.add_trace(go.Histogram(x=sizes, name=lvl, opacity=0.65))
    fig.update_layout(
        barmode="overlay",
        title="Cluster size distributions by level",
        xaxis_title="cluster size (#pages)",
        yaxis_title="count",
        height=650,
    )
    ensure_dir(str(out_html.parent))
    fig.write_html(str(out_html))
    print(f"[viz cluster sizes] Saved: {out_html}")
    return out_html


def viz_network_by_level(
    g_full: Any,
    df_assign: pd.DataFrame,
    level_col: str,
    out_html: Union[Path, str],
    labels_map: Optional[Dict[str, str]] = None,
    max_nodes: int = 2500,
    seed: int = 42,
) -> Optional[Path]:
    """Interactive network plot coloured by cluster level (core pages; hover shows edge weights)."""
    import plotly.graph_objects as go

    out_html = Path(out_html)
    ensure_dir(str(out_html.parent))

    pages = df_assign["page"].astype(str).tolist()
    n = len(pages)
    if n == 0:
        return None

    if "vid_full" not in df_assign.columns:
        return None

    rng = np.random.RandomState(seed)
    if n > max_nodes:
        keep_idx = rng.choice(np.arange(n), size=max_nodes, replace=False)
        keep_pages = set(df_assign.iloc[keep_idx]["page"].astype(str).tolist())
    else:
        keep_pages = set(pages)

    keep_df = df_assign[df_assign["page"].astype(str).isin(keep_pages)].copy()
    vids = keep_df["vid_full"].astype(int).tolist()
    if len(vids) < 2:
        return None

    subg = g_full.subgraph(vids)
    layout = subg.layout("fr")
    xs = [p[0] for p in layout]
    ys = [p[1] for p in layout]

    node_pages = keep_df["page"].astype(str).tolist()
    node_clusters_raw = keep_df[level_col].astype(str).tolist()
    node_clusters = [labels_map.get(c, c) for c in node_clusters_raw] if labels_map else node_clusters_raw

    uniq_clusters = list(dict.fromkeys(node_clusters))
    cluster_to_code = {c: i for i, c in enumerate(uniq_clusters)}
    node_color = [cluster_to_code[c] for c in node_clusters]

    edges_x, edges_y, edge_text = [], [], []
    wts = subg.es["weight"] if "weight" in subg.es.attributes() else [1.0] * subg.ecount()

    for e_i, e in enumerate(subg.es):
        u, v = e.tuple
        edges_x += [xs[u], xs[v], None]
        edges_y += [ys[u], ys[v], None]
        edge_text.append(f"{node_pages[u]} → {node_pages[v]}<br>weight={float(wts[e_i]):.4f}")

    edge_trace = go.Scatter(
        x=edges_x, y=edges_y, mode="lines", line=dict(width=1),
        hoverinfo="text", text=edge_text or None, name="edges", opacity=0.45,
    )
    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers", hoverinfo="text",
        text=[f"{p}<br>{level_col}: {c}" for p, c in zip(node_pages, node_clusters)],
        marker=dict(
            size=8, color=node_color, colorscale="cividis", showscale=True,
            colorbar=dict(title=f"{level_col} cluster"), line=dict(width=0.5),
        ),
        name="nodes",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Network — colored by {level_col} (edges show page→page weights)",
        height=850, margin=dict(l=20, r=20, t=60, b=20), showlegend=False,
    )
    fig.write_html(str(out_html))
    print(f"[viz network {level_col}] Saved: {out_html}")
    return out_html


def viz_L2_nodes_colored_by_root(
    g_core: Any,
    df_core_assign: pd.DataFrame,
    out_html: Union[Path, str],
    max_nodes: int = 300,
    seed: int = 42,
    min_edge_weight: float = 0.0,
    node_size_base: float = 8.0,
    node_size_scale: float = 2.0,
) -> Optional[Path]:
    """Cluster-level network: nodes = L2 clusters, size = #pages, color = root cluster."""
    import plotly.graph_objects as go

    out_html = Path(out_html)

    required = {"vid_core", "page", "root", "L1", "L2"}
    if not required.issubset(df_core_assign.columns):
        print(f"[WARN] Missing columns for L2 cluster plot: {required - set(df_core_assign.columns)}", flush=True)
        return None

    df = df_core_assign.copy()
    for c in ["root", "L1", "L2", "page"]:
        df[c] = df[c].astype(str)

    grp = df.groupby("L2", dropna=False)
    l2_size = grp["page"].nunique().astype(int).to_dict()
    l2_root = grp["root"].agg(lambda x: x.value_counts().index[0]).to_dict()
    l2_l1 = grp["L1"].agg(lambda x: x.value_counts().index[0]).to_dict()

    l2_top_pages: Dict[str, List[str]] = {}
    for l2, sub in grp:
        pages = sub["page"].astype(str).tolist()
        seen: set = set()
        uniq_pages: List[str] = []
        for p in pages:
            if p not in seen:
                uniq_pages.append(p)
                seen.add(p)
            if len(uniq_pages) >= 5:
                break
        l2_top_pages[str(l2)] = uniq_pages

    l2_sorted = sorted(l2_size.items(), key=lambda kv: kv[1], reverse=True)
    l2_keep = [cid for cid, _ in l2_sorted[:max_nodes]]
    l2_keep_set = set(l2_keep)

    vid_to_l2 = dict(zip(df["vid_core"].astype(int).tolist(), df["L2"].astype(str).tolist()))

    edge_sum: Dict[Any, float] = {}
    weights = g_core.es["weight"] if "weight" in g_core.es.attributes() else [1.0] * g_core.ecount()
    for e_id, e in enumerate(g_core.es):
        u, v = e.tuple
        cu = vid_to_l2.get(int(u))
        cv = vid_to_l2.get(int(v))
        if cu is None or cv is None or cu == cv:
            continue
        if cu not in l2_keep_set or cv not in l2_keep_set:
            continue
        a, b = (cu, cv) if cu < cv else (cv, cu)
        edge_sum[(a, b)] = edge_sum.get((a, b), 0.0) + float(weights[e_id])

    edges = [(a, b, w) for (a, b), w in edge_sum.items() if w >= float(min_edge_weight)]
    if not edges:
        print("[WARN] No inter-L2 edges after filtering.", flush=True)
        return None

    l2_index = {cid: i for i, cid in enumerate(l2_keep)}
    n = len(l2_keep)
    rng_np = np.random.default_rng(seed)
    pos = rng_np.normal(size=(n, 2))

    for _ in range(250):
        disp = np.zeros_like(pos)
        for a, b, w in edges:
            i = l2_index[a]
            j = l2_index[b]
            delta = pos[j] - pos[i]
            dist = float(np.linalg.norm(delta)) + 1e-9
            force = (w / dist) * 0.002
            disp[i] += force * delta
            disp[j] -= force * delta
        pos += disp

    xs, ys = pos[:, 0].tolist(), pos[:, 1].tolist()

    edge_x, edge_y, edge_text_list = [], [], []
    for a, b, w in edges:
        i = l2_index[a]
        j = l2_index[b]
        edge_x += [xs[i], xs[j], None]
        edge_y += [ys[i], ys[j], None]
        edge_text_list.append(f"{a} ↔ {b}<br>weight={w:.3f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", line=dict(width=1),
        hoverinfo="text", text=edge_text_list * 3, name="edges", opacity=0.35, showlegend=False,
    )

    roots = [l2_root.get(cid, "NA") for cid in l2_keep]
    uniq_roots = list(dict.fromkeys(roots))
    root_to_code = {r: i for i, r in enumerate(uniq_roots)}
    node_color = [root_to_code[r] for r in roots]

    sizes = [int(l2_size.get(cid, 1)) for cid in l2_keep]
    node_sizes = [node_size_base + node_size_scale * (s ** 0.5) for s in sizes]

    node_text = []
    for cid, s in zip(l2_keep, sizes):
        top5 = l2_top_pages.get(cid, [])
        top5_str = "<br>- " + "<br>- ".join(top5) if top5 else "<br>(no pages?)"
        node_text.append(
            f"root: {l2_root.get(cid)}<br>L1: {l2_l1.get(cid)}<br>L2: {cid}"
            f"<br>n_pages: {s}<br>top 5 pages:{top5_str}"
        )

    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers", hoverinfo="text", text=node_text,
        marker=dict(
            size=node_sizes, color=node_color, colorscale="cividis", showscale=True,
            colorbar=dict(title="root cluster"), line=dict(width=0.5),
        ),
        name="L2 clusters", showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"L2 cluster network (node=L2, size=#pages, color=root)  n={n}",
        height=900, margin=dict(l=20, r=20, t=60, b=20),
    )
    ensure_dir(str(out_html.parent))
    fig.write_html(str(out_html))
    print(f"[viz L2 cluster network] Saved: {out_html}")
    return out_html


def viz_taxonomy_sunburst_cluster(
    df_final: pd.DataFrame,
    out_html: Union[Path, str],
    title: str = "Taxonomy Sunburst (root → L1 → L2)",
) -> Path:
    """Plotly sunburst for Leiden cluster taxonomy (root → L1 → L2)."""
    import plotly.express as px

    out_html = Path(out_html)
    df_sb = df_final[["root", "L1", "L2"]].dropna(subset=["root", "L1", "L2"]).drop_duplicates().copy()
    df_sb["_weight"] = 1

    fig = px.sunburst(df_sb, path=["root", "L1", "L2"], values="_weight")
    fig.update_layout(title=title, height=900, margin=dict(l=20, r=20, t=60, b=20))
    ensure_dir(str(out_html.parent))
    fig.write_html(str(out_html))
    print(f"[viz cluster sunburst] Saved: {out_html}")
    return out_html


# ------------------------------------------------------------------
# H) Cross-strategy comparison visualisations
# ------------------------------------------------------------------

# Fixed color palette for strategies (consistent across all charts)
_STRATEGY_COLORS = {
    "adaptive": "#636EFA",
    "simple": "#EF553B",
    "margin_adaptive": "#00CC96",
    "mega_adaptive": "#AB63FA",
}


def _strategy_color(name: str) -> str:
    return _STRATEGY_COLORS.get(name, "#FFA15A")


def viz_comparison_cluster_counts(
    results: Dict[str, dict],
    level_cols: List[str],
    out_png: Union[Path, str],
) -> Path:
    """Grouped bar chart: cluster count per level, grouped by strategy."""
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    strategies = list(results.keys())
    n_strats = len(strategies)
    n_levels = len(level_cols)
    x = np.arange(n_levels)
    width = 0.8 / max(n_strats, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, strategy in enumerate(strategies):
        metrics = results[strategy]
        counts = []
        for lvl in level_cols:
            lvl_m = metrics.get("levels", {}).get(lvl, {})
            counts.append(lvl_m.get("n_clusters", 0))
        ax.bar(x + i * width, counts, width, label=strategy,
               color=_strategy_color(strategy))

    ax.set_xlabel("Level")
    ax.set_ylabel("Number of Clusters")
    ax.set_title("Cluster Counts by Strategy and Level")
    ax.set_xticks(x + width * (n_strats - 1) / 2)
    ax.set_xticklabels(level_cols)
    ax.legend()
    fig.tight_layout()
    ensure_dir(str(out_png.parent))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return out_png


def viz_comparison_margin_distributions(
    margin_data: Dict[str, pd.DataFrame],
    level: str,
    out_png: Union[Path, str],
) -> Path:
    """Box plots of per-cluster margin distributions, one box per strategy.

    Parameters
    ----------
    margin_data : dict
        strategy_name -> DataFrame with at least a ``margin`` column.
    """
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    box_data, labels, colors = [], [], []
    for strategy, df in margin_data.items():
        if df is None or df.empty or "margin" not in df.columns:
            continue
        box_data.append(df["margin"].dropna().values)
        labels.append(strategy)
        colors.append(_strategy_color(strategy))

    fig, ax = plt.subplots(figsize=(8, 5))
    if box_data:
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
    ax.axhline(y=0, linestyle="--", color="red", linewidth=1, label="margin=0")
    ax.set_ylabel("Margin (within_sim − nearest_other_sim)")
    ax.set_title(f"Per-Cluster Margin Distribution — {level}")
    ax.legend()
    fig.tight_layout()
    ensure_dir(str(out_png.parent))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return out_png


def viz_comparison_cluster_sizes(
    size_data: Dict[str, np.ndarray],
    level: str,
    out_png: Union[Path, str],
) -> Path:
    """Overlaid histograms of cluster sizes per strategy for a given level."""
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy, sizes in size_data.items():
        if sizes is None or len(sizes) == 0:
            continue
        ax.hist(sizes, bins=30, alpha=0.55, label=strategy,
                color=_strategy_color(strategy))

    ax.set_xlabel("Cluster Size (pages)")
    ax.set_ylabel("Count")
    ax.set_title(f"Cluster Size Distribution — {level}")
    ax.legend()
    fig.tight_layout()
    ensure_dir(str(out_png.parent))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return out_png


def viz_comparison_max_cluster_fraction(
    results: Dict[str, dict],
    level_cols: List[str],
    out_png: Union[Path, str],
) -> Path:
    """Grouped bar chart: max cluster size / total pages per strategy×level."""
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    strategies = list(results.keys())
    n_strats = len(strategies)
    n_levels = len(level_cols)
    x = np.arange(n_levels)
    width = 0.8 / max(n_strats, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, strategy in enumerate(strategies):
        metrics = results[strategy]
        fracs = []
        for lvl in level_cols:
            lvl_m = metrics.get("levels", {}).get(lvl, {})
            max_c = lvl_m.get("max_cluster", 0)
            total = lvl_m.get("total_pages", 1)
            fracs.append(max_c / max(total, 1))
        ax.bar(x + i * width, fracs, width, label=strategy,
               color=_strategy_color(strategy))

    ax.axhline(y=0.9, linestyle="--", color="red", linewidth=1,
               label="mega threshold (0.9)")
    ax.set_xlabel("Level")
    ax.set_ylabel("Max Cluster Size / Total Pages")
    ax.set_title("Largest Cluster Fraction by Strategy and Level")
    ax.set_xticks(x + width * (n_strats - 1) / 2)
    ax.set_xticklabels(level_cols)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    ensure_dir(str(out_png.parent))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return out_png


def viz_comparison_negative_margins(
    results: Dict[str, dict],
    level_cols: List[str],
    out_png: Union[Path, str],
) -> Path:
    """Grouped bar chart: count of negative-margin clusters per strategy×level."""
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    strategies = list(results.keys())
    n_strats = len(strategies)
    n_levels = len(level_cols)
    x = np.arange(n_levels)
    width = 0.8 / max(n_strats, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, strategy in enumerate(strategies):
        metrics = results[strategy]
        neg_counts = []
        for lvl in level_cols:
            lvl_m = metrics.get("levels", {}).get(lvl, {})
            neg_counts.append(lvl_m.get("n_negative_margins", 0))
        ax.bar(x + i * width, neg_counts, width, label=strategy,
               color=_strategy_color(strategy))

    ax.set_xlabel("Level")
    ax.set_ylabel("Clusters with margin < 0")
    ax.set_title("Negative-Margin Clusters by Strategy and Level")
    ax.set_xticks(x + width * (n_strats - 1) / 2)
    ax.set_xticklabels(level_cols)
    ax.legend()
    fig.tight_layout()
    ensure_dir(str(out_png.parent))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return out_png


def viz_comparison_coherence_ari(
    results: Dict[str, dict],
    level_cols: List[str],
    out_png: Union[Path, str],
) -> Path:
    """Two-panel chart: mean coherence and mean ARI stability per strategy×level."""
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    strategies = list(results.keys())
    n_strats = len(strategies)
    n_levels = len(level_cols)
    x = np.arange(n_levels)
    width = 0.8 / max(n_strats, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, strategy in enumerate(strategies):
        metrics = results[strategy]
        cohs, aris = [], []
        for lvl in level_cols:
            lvl_m = metrics.get("levels", {}).get(lvl, {})
            cohs.append(lvl_m.get("coherence_mean", 0))
            aris.append(lvl_m.get("ari_mean", 0))

        ax1.bar(x + i * width, cohs, width, label=strategy,
                color=_strategy_color(strategy))
        ax2.bar(x + i * width, aris, width, label=strategy,
                color=_strategy_color(strategy))

    for ax, title in [(ax1, "Mean Coherence"), (ax2, "Mean ARI Stability")]:
        ax.set_xlabel("Level")
        ax.set_title(title)
        ax.set_xticks(x + width * (n_strats - 1) / 2)
        ax.set_xticklabels(level_cols)
        ax.legend()

    fig.suptitle("Coherence & ARI Stability by Strategy", fontsize=13)
    fig.tight_layout()
    ensure_dir(str(out_png.parent))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return out_png


def viz_comparison_page_agreement(
    assignments: Dict[str, pd.DataFrame],
    level_cols: List[str],
    out_png: Union[Path, str],
) -> Path:
    """Heatmap of pairwise Adjusted Rand Index between strategies at each level.

    Parameters
    ----------
    assignments : dict
        strategy_name -> page_assignments DataFrame (must have ``page`` + level columns).
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score

    out_png = Path(out_png)
    strats = list(assignments.keys())
    n = len(strats)
    n_levels = len(level_cols)

    fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels + 1, 4.5))
    if n_levels == 1:
        axes = [axes]

    for li, lvl in enumerate(level_cols):
        ax = axes[li]
        # Build common page set
        page_sets = []
        for s in strats:
            df = assignments[s]
            if lvl in df.columns and "page" in df.columns:
                page_sets.append(set(df["page"].astype(str).tolist()))
            else:
                page_sets.append(set())
        common = set.intersection(*page_sets) if page_sets else set()
        common = sorted(common)

        ari_mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    ari_mat[i, j] = 1.0
                    continue
                df_i = assignments[strats[i]]
                df_j = assignments[strats[j]]
                if lvl not in df_i.columns or lvl not in df_j.columns:
                    continue
                map_i = df_i.set_index("page")[lvl].to_dict()
                map_j = df_j.set_index("page")[lvl].to_dict()
                labels_i = [str(map_i.get(p, "")) for p in common]
                labels_j = [str(map_j.get(p, "")) for p in common]
                ari_mat[i, j] = adjusted_rand_score(labels_i, labels_j)

        im = ax.imshow(ari_mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="equal")
        ax.set_xticks(range(n))
        ax.set_xticklabels(strats, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(strats, fontsize=8)
        ax.set_title(f"ARI — {lvl}")
        # Annotate cells
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{ari_mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if ari_mat[i, j] > 0.4 else "white")

    fig.colorbar(im, ax=axes, shrink=0.8, label="ARI")
    fig.suptitle("Pairwise Strategy Agreement (Adjusted Rand Index)", fontsize=13)
    fig.tight_layout()
    ensure_dir(str(out_png.parent))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return out_png
