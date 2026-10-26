"""
p2_taxonomy/cluster_strategy_compare.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run all clustering strategies on the same input, validate each,
and generate cross-strategy comparison outputs.

Usage (via CLI)::

    python -m p2_taxonomy.main compare-strategies \\
        --input filtered.csv --mode digital

Or programmatically::

    from p2_taxonomy.cluster_strategy_compare import run_strategy_comparison
    run_strategy_comparison(input_csv=Path("filtered.csv"), ...)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p2_taxonomy.analysis_cluster import (
    setup_clustering_infrastructure,
    finalize_clustering_outputs,
    recursive_split,
    strategy_simple_sweep,
    strategy_margin_adaptive,
    strategy_mega_adaptive,
    ensure_dir,
    LevelConfig,
)


# ------------------------------------------------------------------
# Strategy dispatch
# ------------------------------------------------------------------

def _dispatch_strategy(
    strategy: str,
    ctx: dict,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, dict]:
    """Run a single strategy using shared infrastructure from *ctx*."""
    common = dict(
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
        return recursive_split(**common)
    elif strategy == "simple":
        return strategy_simple_sweep(**common)
    elif strategy == "margin_adaptive":
        return strategy_margin_adaptive(
            **common,
            margin_threshold=getattr(args, "margin_threshold", 0.0),
        )
    elif strategy == "mega_adaptive":
        return strategy_mega_adaptive(
            **common,
            mega_pct=getattr(args, "mega_pct", 0.9),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ------------------------------------------------------------------
# Per-strategy validation (page distances + cluster margins)
# ------------------------------------------------------------------

def _validate_one_strategy(
    run_dir: Path,
    pages_csv: Path,
    mode: str,
    validation_dir: Path,
    embed_model: str,
    level_cols: List[str],
    skip_coverage: bool = True,
) -> Dict[str, Any]:
    """Run page distances + cluster margins for one strategy's output.

    Returns dict with margin stats per level.
    """
    import torch
    from p2_taxonomy.analysis_cluster import (
        CACHE_ROOT,
        clean_description_text,
        load_or_build_embeddings,
    )
    from p2_taxonomy.analysis_cluster_quality import compute_cluster_margins, run_page_distances

    ensure_dir(validation_dir)

    # Find output files in run_dir
    taxonomy_files = list(run_dir.glob("taxonomy_long_labeled__*.csv"))
    if not taxonomy_files:
        taxonomy_files = list(run_dir.glob("taxonomy_long_*.csv"))
    assign_files = list(run_dir.glob("page_assignments_L2*.csv"))
    if not assign_files:
        assign_files = list(run_dir.glob("page_assignments_*.csv"))

    if not taxonomy_files or not assign_files:
        print(f"[validate] Skipping {run_dir.name}: no taxonomy/assignment CSVs found")
        return {}

    taxonomy_csv = taxonomy_files[0]
    assign_csv = assign_files[0]

    # Part 1: Page distances
    page_dist_out = validation_dir / f"page_distances_{mode}.csv"
    try:
        run_page_distances(
            taxonomy_csv=taxonomy_csv,
            pages_csv=pages_csv,
            taxonomy_page_col="page",
            taxonomy_cols=",".join(level_cols),
            pages_title_col="page_title",
            pages_desc_col="description",
            embed_model=embed_model,
            exclude_peripheral=True,
            out=str(page_dist_out),
        )
        print(f"[validate] Page distances → {page_dist_out}")
    except Exception as e:
        print(f"[validate] Page distances failed: {e}")

    # Part 2: Cluster margins
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

    margin_results: Dict[str, Any] = {}
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

        margins_df = compute_cluster_margins(emb_f, labels_f)
        margins_df = margins_df.sort_values("margin")
        csv_path = validation_dir / f"cluster_margins_{mode}_{level}.csv"
        margins_df.to_csv(csv_path, index=False)

        n_neg = int((margins_df["margin"] < 0).sum())
        margin_results[level] = {
            "margins_df": margins_df,
            "n_clusters": len(unique),
            "n_negative_margins": n_neg,
            "mean_margin": float(margins_df["margin"].mean()),
            "median_margin": float(margins_df["margin"].median()),
            "min_margin": float(margins_df["margin"].min()),
        }

    # Optional: coverage
    if not skip_coverage:
        try:
            from p2_taxonomy.analysis_cluster_quality import run_coverage_analysis

            # Filter to core pages
            l2_col = level_cols[-1] if level_cols else None
            if l2_col and l2_col in df_assign.columns:
                l2_labels = df_assign[l2_col].astype(str)
                is_peripheral = (
                    l2_labels.str.contains("Peripheral", case=False, na=False)
                    | (l2_labels == "nan") | (l2_labels == "None")
                )
                core_pages = set(df_assign.loc[~is_peripheral, "page"].tolist())
                df_core = df_input[df_input["page_title"].isin(core_pages)].copy()
                core_csv = validation_dir / f"core_pages_filtered_{mode}.csv"
                df_core.to_csv(core_csv, index=False)

                run_coverage_analysis(
                    wiki_csv=core_csv,
                    mode=mode,
                    output_dir=validation_dir,
                    embed_model=embed_model,
                    threshold=0.5,
                )
                print(f"[validate] Coverage → {validation_dir}")
        except Exception as e:
            print(f"[validate] Coverage failed: {e}")

    return margin_results


# ------------------------------------------------------------------
# Metric collection
# ------------------------------------------------------------------

def _collect_strategy_metrics(
    run_dir: Path,
    validation_margins: Dict[str, Any],
    run_diag: dict,
    level_cols: List[str],
) -> Dict[str, Any]:
    """Extract key metrics from a single strategy run for comparison."""
    metrics: Dict[str, Any] = {"levels": {}}

    total_pages = 0
    for lvl in level_cols:
        lvl_diag = run_diag.get("levels", {}).get(lvl, {})
        val_m = validation_margins.get(lvl, {})

        n_clusters = lvl_diag.get("n_clusters", 0)
        min_cluster = lvl_diag.get("min_cluster", 0)
        median_cluster = lvl_diag.get("median_cluster", 0)
        max_cluster = lvl_diag.get("max_cluster", 0)
        p90_cluster = lvl_diag.get("p90_cluster", 0)

        # Total pages: sum of all cluster sizes at root level
        if lvl == level_cols[0]:
            total_pages = sum(
                g.get("n_nodes", 0) for g in run_diag.get("groups", [])
                if g.get("level") == lvl
            )

        metrics["levels"][lvl] = {
            "n_clusters": n_clusters,
            "min_cluster": min_cluster,
            "median_cluster": median_cluster,
            "p90_cluster": p90_cluster,
            "max_cluster": max_cluster,
            "total_pages": total_pages if total_pages > 0 else max_cluster,
            "coherence_mean": lvl_diag.get("coherence_mean"),
            "ari_mean": lvl_diag.get("ari_mean"),
            # From validation margins
            "n_negative_margins": val_m.get("n_negative_margins", 0),
            "mean_margin": val_m.get("mean_margin"),
            "median_margin": val_m.get("median_margin"),
            "min_margin": val_m.get("min_margin"),
        }

    metrics["run_diag"] = run_diag
    return metrics


# ------------------------------------------------------------------
# Comparison output generation
# ------------------------------------------------------------------

def _generate_comparison_outputs(
    results: Dict[str, Dict[str, Any]],
    margin_dfs: Dict[str, Dict[str, pd.DataFrame]],
    assignments: Dict[str, pd.DataFrame],
    comparison_dir: Path,
    level_cols: List[str],
) -> None:
    """Generate comparison CSV, JSON, and all comparison charts."""
    from p2_taxonomy.visualisations import (
        viz_comparison_cluster_counts,
        viz_comparison_margin_distributions,
        viz_comparison_cluster_sizes,
        viz_comparison_max_cluster_fraction,
        viz_comparison_negative_margins,
        viz_comparison_coherence_ari,
        viz_comparison_page_agreement,
    )

    ensure_dir(comparison_dir)

    # --- Summary CSV ---
    rows = []
    for strategy, metrics in results.items():
        for lvl, lvl_m in metrics.get("levels", {}).items():
            row = {"strategy": strategy, "level": lvl}
            row.update(lvl_m)
            rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_csv = comparison_dir / "strategy_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[compare] Summary CSV → {summary_csv}")

    # --- Summary JSON ---
    summary_json = comparison_dir / "strategy_summary.json"
    # Strip non-serialisable objects
    clean_results = {}
    for s, m in results.items():
        clean = {"levels": m.get("levels", {})}
        clean_results[s] = clean
    summary_json.write_text(
        json.dumps(clean_results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[compare] Summary JSON → {summary_json}")

    # --- Charts ---
    print("[compare] Generating comparison charts...")

    viz_comparison_cluster_counts(
        results, level_cols,
        comparison_dir / "cluster_counts.png",
    )

    viz_comparison_max_cluster_fraction(
        results, level_cols,
        comparison_dir / "max_cluster_fraction.png",
    )

    viz_comparison_negative_margins(
        results, level_cols,
        comparison_dir / "negative_margins.png",
    )

    viz_comparison_coherence_ari(
        results, level_cols,
        comparison_dir / "coherence_ari.png",
    )

    # Per-level charts
    for lvl in level_cols:
        # Margin distributions
        lvl_margins = {}
        for s, s_margins in margin_dfs.items():
            if lvl in s_margins:
                lvl_margins[s] = s_margins[lvl]
        if lvl_margins:
            viz_comparison_margin_distributions(
                lvl_margins, lvl,
                comparison_dir / f"margin_distributions_{lvl}.png",
            )

        # Cluster size distributions
        lvl_sizes = {}
        for s, df_a in assignments.items():
            if lvl in df_a.columns:
                sizes = df_a.groupby(lvl)["page"].size().values
                # Exclude peripheral
                labels = df_a[lvl].astype(str)
                mask = ~labels.str.contains("Peripheral", case=False, na=False)
                sizes = df_a.loc[mask].groupby(lvl)["page"].size().values
                lvl_sizes[s] = sizes
        if lvl_sizes:
            viz_comparison_cluster_sizes(
                lvl_sizes, lvl,
                comparison_dir / f"cluster_size_distributions_{lvl}.png",
            )

    # Page agreement heatmap
    if len(assignments) >= 2:
        viz_comparison_page_agreement(
            assignments, level_cols,
            comparison_dir / "page_agreement.png",
        )

    print(f"[compare] All comparison outputs → {comparison_dir}")


# ------------------------------------------------------------------
# Main orchestrator
# ------------------------------------------------------------------

def run_strategy_comparison(
    input_csv: Path,
    output_dir: Path,
    mode: str = "digital",
    strategies: Optional[List[str]] = None,
    skip_coverage: bool = True,
    *,
    # All clustering params
    semantic_weight: float = 1.0,
    edge_weight: float = 0.0,
    tau_edge: float = 0.20,
    hub_penalty: bool = False,
    embed_model: str = "AI-Growth-Lab/PatentSBERTa",
    n_levels: int = 2,
    seed: int = 42,
    iterations: int = 50,
    kmins: str = "8,5,2",
    kmaxs: str = "12,8,6",
    min_sizes: str = "500,25,8",
    split_sizes: str = "3000,100,30",
    margin_weight: float = 10.0,
    small_frac_weight: float = 1.0,
    margin_threshold: float = 0.0,
    mega_pct: float = 0.9,
    mega_abs: int = 200,
    do_label: bool = False,
    openrouter_model: str = "google/gemini-3-flash-preview",
    sleep_s: float = 1.5,
    max_pages_in_prompt: int = 40,
    user_agent: str = "wiki-link-extractor/1.0 (research use)",
) -> dict:
    """Run all strategies, validate each, generate comparison.

    Returns dict with per-strategy metrics and comparison paths.
    """
    import torch

    if strategies is None:
        strategies = ["adaptive", "simple", "margin_adaptive", "mega_adaptive"]

    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 80}")
    print(f"  STRATEGY COMPARISON")
    print(f"  input     = {input_csv}")
    print(f"  output    = {output_dir}")
    print(f"  strategies= {strategies}")
    print(f"  mode      = {mode}")
    print(f"{'=' * 80}\n")

    # ------------------------------------------------------------------
    # 1. Build shared infrastructure ONCE
    # ------------------------------------------------------------------
    base_args = argparse.Namespace(
        input=str(input_csv),
        out_dir=str(output_dir),
        title_col="page_title",
        links_col="linked_wiki_titles_json",
        desc_col="description",
        given_root_col="root_technology",
        min_component_size=5,
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
        strategy=strategies[0],
        margin_threshold=margin_threshold,
        mega_pct=mega_pct,
        mega_abs=mega_abs,
        plot=False,
        label=do_label,
        openrouter_model=openrouter_model,
        sleep_s=sleep_s,
        max_pages_in_prompt=max_pages_in_prompt,
        user_agent=user_agent,
    )

    ctx = setup_clustering_infrastructure(base_args)
    level_cols = ctx["level_cols"]

    # ------------------------------------------------------------------
    # 2. Run each strategy
    # ------------------------------------------------------------------
    all_results: Dict[str, dict] = {}
    all_margin_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}
    all_assignments: Dict[str, pd.DataFrame] = {}

    for si, strategy in enumerate(strategies):
        print(f"\n{'#' * 80}")
        print(f"  STRATEGY {si + 1}/{len(strategies)}: {strategy}")
        print(f"{'#' * 80}\n")

        # Per-strategy output directory
        strat_dir = output_dir / strategy
        ensure_dir(strat_dir)

        # Make a per-strategy copy of ctx with updated paths
        strat_ctx = dict(ctx)
        strat_ctx["out_dir"] = strat_dir
        strat_ctx["suffix"] = f"_description_{strategy}"

        # Make a per-strategy copy of args
        strat_args = copy.copy(base_args)
        strat_args.strategy = strategy
        strat_args.out_dir = str(strat_dir)
        strat_args.margin_threshold = margin_threshold
        strat_args.mega_pct = mega_pct
        strat_args.mega_abs = mega_abs

        # Dispatch strategy
        df_core_assign, run_diag = _dispatch_strategy(strategy, strat_ctx, strat_args)

        # Merge metadata
        run_diag.update(ctx["run_diag_meta"])
        run_diag.setdefault("params", {})["strategy"] = strategy

        # Finalize outputs (writes CSVs, plots if enabled, labels if enabled)
        finalize_clustering_outputs(strat_ctx, df_core_assign, run_diag, strat_args)

        # Validation (page distances + margins)
        validation_dir = strat_dir / "validation"
        margin_results = _validate_one_strategy(
            run_dir=strat_dir,
            pages_csv=input_csv,
            mode=mode,
            validation_dir=validation_dir,
            embed_model=embed_model,
            level_cols=level_cols,
            skip_coverage=skip_coverage,
        )

        # Collect metrics
        metrics = _collect_strategy_metrics(
            run_dir=strat_dir,
            validation_margins=margin_results,
            run_diag=run_diag,
            level_cols=level_cols,
        )
        all_results[strategy] = metrics

        # Store margin DataFrames for comparison charts
        all_margin_dfs[strategy] = {
            lvl: margin_results[lvl]["margins_df"]
            for lvl in margin_results
            if "margins_df" in margin_results.get(lvl, {})
        }

        # Load page assignments for ARI comparison
        assign_files = list(strat_dir.glob("page_assignments_L2*.csv"))
        if not assign_files:
            assign_files = list(strat_dir.glob("page_assignments_*.csv"))
        if assign_files:
            all_assignments[strategy] = pd.read_csv(assign_files[0])
            all_assignments[strategy]["page"] = all_assignments[strategy]["page"].astype(str)

    # ------------------------------------------------------------------
    # 3. Generate comparison outputs
    # ------------------------------------------------------------------
    comparison_dir = output_dir / "comparison"
    _generate_comparison_outputs(
        results=all_results,
        margin_dfs=all_margin_dfs,
        assignments=all_assignments,
        comparison_dir=comparison_dir,
        level_cols=level_cols,
    )

    print(f"\n{'=' * 80}")
    print(f"  STRATEGY COMPARISON COMPLETE")
    print(f"  Output: {output_dir}")
    for s in strategies:
        print(f"    {s}/")
    print(f"    comparison/")
    print(f"{'=' * 80}\n")

    return {
        "output_dir": str(output_dir),
        "strategies": strategies,
        "results": {s: m.get("levels", {}) for s, m in all_results.items()},
        "comparison_dir": str(comparison_dir),
    }
