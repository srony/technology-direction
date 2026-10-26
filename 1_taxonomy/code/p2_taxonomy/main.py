"""
p2_taxonomy/main.py
~~~~~~~~~~~~~~~~~~~
CLI entry point for the P2 taxonomy module.

Pipeline steps::

    01  crawl             BFS Wikipedia crawl from seeds
    02  clean             Filter/clean raw taxonomy
    03  match-patents     FAISS wiki-patent matching              → 03_match/
    04  filter-matches    Threshold + dedup + noise + LLM         → 04_filter/
    04b page-relevance    Per-page patent coverage + uniqueness   → 04_filter/
    05  diagnose          Pre-cluster coverage diagnostics        → 05_coverage_pre/
    06  cluster           Adaptive Leiden clustering              → 06_cluster/
    06b validate-clusters Post-cluster: page-distances + margins  → 06_cluster/<run>/validation/
    07  tag-patents       Tag patents with taxonomy labels        → 07_tag_patents/

Usage::

    python -m p2_taxonomy.main crawl     --seeds-file seeds.txt --output out.csv
    python -m p2_taxonomy.main clean     --input raw.csv --output clean.csv --mode green
    python -m p2_taxonomy.main match-patents --wiki-csv clean.csv --patent-dir parts/ --mode digital
    python -m p2_taxonomy.main filter-matches --wiki-csv clean.csv --match-dir 03_match/digital/ --validate-llm
    python -m p2_taxonomy.main diagnose  --wiki-csv filtered.csv --mode digital
    python -m p2_taxonomy.main cluster   --input filtered.csv --output-dir out/
    python -m p2_taxonomy.main validate-clusters --run-dir 06_cluster/run/ --pages-csv filtered.csv --mode digital
    python -m p2_taxonomy.main tag-patents --taxonomy-csv tax.csv --pages-csv pages.csv --patents-dir parts/ --output-dir out/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Ensure the project root is on the path so ``shared`` can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ------------------------------------------------------------------
# Sub-command: crawl
# ------------------------------------------------------------------

def _cmd_crawl(args: argparse.Namespace) -> None:
    from p2_taxonomy.data_extraction_wikipedia import (
        WikiSummaryFetcher,
        build_taxonomy,
        crawl_from_seeds,
        extract_topic_from_filename,
        load_seed_categories,
    )

    seeds_file = Path(args.seeds_file)
    topic = extract_topic_from_filename(seeds_file)
    output_csv = args.output or f"{topic}_taxonomy.csv"

    seed_categories = load_seed_categories(seeds_file)

    print(f"\n=== P2 TAXONOMY CRAWLER ===")
    print(f"Topic: {topic}")
    print(f"Seeds: {len(seed_categories)}")
    print(f"Max depth: {args.max_depth}")
    print(f"Max categories: {args.max_categories}")
    print(f"Output: {output_csv}\n")

    cat_children, cat_pages = crawl_from_seeds(
        seed_categories, args.max_depth, args.max_categories,
    )

    rows = build_taxonomy(seed_categories, cat_children, cat_pages)
    print(f"\nTotal taxonomy rows: {len(rows)}")

    if args.descriptions:
        fetcher = WikiSummaryFetcher()
        from tqdm import tqdm
        for row in tqdm(rows, desc="Fetching descriptions"):
            title_clean = row["page_title"].replace("_", " ")
            row["description"] = fetcher.get_summary(title_clean)
        fetcher.save_cache()
    else:
        for row in rows:
            row["description"] = ""

    # Add level columns
    max_cols = max((len(r["hierarchy_path"].split(" > ")) for r in rows), default=0)
    for row in rows:
        parts = row["hierarchy_path"].split(" > ")
        for i in range(max_cols):
            row[f"level_{i}"] = parts[i] if i < len(parts) else ""

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {output_csv}")
    else:
        print("No rows produced.")


# ------------------------------------------------------------------
# Sub-command: clean
# ------------------------------------------------------------------

def _cmd_clean(args: argparse.Namespace) -> None:
    from p2_taxonomy.cleaning_wikipedia import run_cleaning_pipeline

    rc = run_cleaning_pipeline(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        mode=args.mode,
        include_lists=args.include_lists,
        sleep=args.sleep,
        timeout=args.timeout,
        use_embeddings=args.use_embeddings,
        embed_model=args.model,
        description_cache=args.description_cache,
    )
    raise SystemExit(rc)


# ------------------------------------------------------------------
# Sub-command: match-patents
# ------------------------------------------------------------------

def _cmd_match_patents(args: argparse.Namespace) -> None:
    from p2_taxonomy.cleaning_patent_wiki_match import run_matching

    wiki_csv = Path(args.wiki_csv)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = wiki_csv.parent.parent / "03_match" / args.mode

    out = run_matching(
        wiki_csv=wiki_csv,
        patent_parts_dir=Path(args.patent_dir),
        output_dir=output_dir,
        mode=args.mode,
        embed_model=args.embed_model,
    )
    print(f"[done] Output CSV: {out}")


# ------------------------------------------------------------------
# Sub-command: filter-matches
# ------------------------------------------------------------------

def _cmd_filter_matches(args: argparse.Namespace) -> None:
    from p2_taxonomy.cleaning_patent_wiki_match import filter_matched_pages

    if args.match_dir:
        match_dir = Path(args.match_dir)
    else:
        match_dir = Path(args.wiki_csv).parent.parent / "03_match" / args.mode

    filter_matched_pages(
        wiki_csv=Path(args.wiki_csv),
        match_dir=match_dir,
        output_csv=Path(args.output) if args.output else None,
        threshold=args.threshold,
        validate_llm=args.validate_llm,
        openrouter_model=args.openrouter_model,
        sleep_s=args.sleep_s,
        mode=args.mode,
    )


# ------------------------------------------------------------------
# Sub-command: page-relevance
# ------------------------------------------------------------------

def _cmd_page_relevance(args: argparse.Namespace) -> None:
    from p2_taxonomy.cleaning_patent_wiki_match import compute_page_relevance

    pages_csv = Path(args.pages_csv)
    pipeline_dir = pages_csv.parent.parent

    if args.faiss_index:
        faiss_index = Path(args.faiss_index)
    else:
        faiss_index = pipeline_dir / "03_match" / args.mode / "faiss_patent.index"

    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = pages_csv.parent / "page_relevance.csv"

    compute_page_relevance(
        pages_csv=pages_csv,
        faiss_index_path=faiss_index,
        output_csv=output_csv,
        embed_model=args.embed_model,
        sim_threshold=args.sim_threshold,
        k_search=args.k_search,
        mode=args.mode,
    )


# ------------------------------------------------------------------
# Sub-command: ingest
# ------------------------------------------------------------------

def _cmd_ingest(args: argparse.Namespace) -> None:
    from shared.patent_cache import (
        ingest_corpus,
        compute_and_cache_embeddings,
        INPUT_TSV,
        CORPUS_ROOT,
        CACHE_ROOT,
    )

    input_tsv = args.input_tsv or INPUT_TSV
    corpus_root = args.corpus_root or CORPUS_ROOT
    cache_root = args.cache_root or CACHE_ROOT

    modes = ["digital", "green"] if args.mode == "all" else [args.mode]

    for mode in modes:
        print(f"\n=== INGEST mode={mode} ===")
        ingest_corpus(mode=mode, input_tsv=input_tsv, cache_root=corpus_root)
        compute_and_cache_embeddings(
            mode=mode,
            cache_root=cache_root,
            corpus_root=corpus_root,
            embed_model=args.embed_model,
            views=args.views,
        )

    print("\n[ingest] ALL DONE")


# ------------------------------------------------------------------
# Sub-command: topics
# ------------------------------------------------------------------

def _cmd_topics(args: argparse.Namespace) -> None:
    from p2_taxonomy.data_extraction_patent_topics import RunConfig, main_run

    cfg = RunConfig(
        mode=args.mode,
        views=args.views,
        layers=args.layers,
        embed_model=args.embed_model,
        exposure=args.exposure,
        exposure_topk=args.exposure_topk,
        sample_n=args.sample_n,
        anchor_guided=args.anchor_guided,
        anchor_alpha=args.anchor_alpha,
        label=args.label,
        plot=args.plot,
        openrouter_model=args.openrouter_model,
        sleep_s=args.sleep_s,
        max_docs_in_prompt=args.max_docs_in_prompt,
        user_agent=args.user_agent,
    )
    main_run(cfg)


# ------------------------------------------------------------------
# Sub-command: cluster
# ------------------------------------------------------------------

def _build_run_name(args: argparse.Namespace) -> str:
    """Build a descriptive run folder name from clustering params."""
    hub = 1 if args.hub_penalty else 0
    name = f"sw{args.semantic_weight}_tau{args.tau_edge}_hub{hub}_{args.strategy}_mw{int(args.margin_weight)}_iter{args.iterations}"
    return name


def _cmd_cluster(args: argparse.Namespace) -> None:
    from p2_taxonomy.analysis_cluster import run_adaptive_clustering

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        input_csv = Path(args.input)
        cluster_root = input_csv.parent.parent / "06_cluster" / args.mode
        run_name = _build_run_name(args)
        out_dir = cluster_root / run_name

    result = run_adaptive_clustering(
        input_csv=Path(args.input),
        output_dir=out_dir,
        semantic_weight=args.semantic_weight,
        edge_weight=args.edge_weight,
        tau_edge=args.tau_edge,
        hub_penalty=args.hub_penalty,
        embed_model=args.embed_model,
        n_levels=args.n_levels,
        seed=args.seed,
        iterations=args.iterations,
        kmins=args.kmins,
        kmaxs=args.kmaxs,
        min_sizes=args.min_sizes,
        split_sizes=args.split_sizes,
        margin_weight=args.margin_weight,
        do_plot=args.plot,
        do_label=args.label,
        openrouter_model=args.openrouter_model,
        sleep_s=args.sleep_s,
        max_pages_in_prompt=args.max_pages_in_prompt,
        user_agent=args.user_agent,
        strategy=args.strategy,
        margin_threshold=args.margin_threshold,
        mega_pct=args.mega_pct,
        mega_abs=args.mega_abs,
        mega_threshold=args.mega_threshold,
        local_tau=args.local_tau,
        resplit=args.resplit,
        prune_pct=getattr(args, "prune_pct", 0.0),
        mega_frac=args.mega_frac,
        mega_min_size=args.mega_min_size,
        weak_margin=args.weak_margin,
        weak_min_size=args.weak_min_size,
    )
    print(f"[cluster] status={result['status']} output_dir={result['output_dir']}")

    if args.validate:
        print(f"\n{'='*60}")
        print(f"  POST-CLUSTER VALIDATION (--validate)")
        print(f"{'='*60}\n")
        validate_args = argparse.Namespace(
            run_dir=result["output_dir"],
            pages_csv=args.input,
            mode=args.mode,
            output_dir=None,
            embed_model=args.embed_model,
            threshold=0.5,
            citation_csv=r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\patent\state_patent_citations.csv",
            min_component_size=10,
            skip_coverage=False,
            skip_external=False,
            cpc_digits="3,4",
            skip_wiki_category=False,
            skip_modularity=False,
            llm_verify=False,
            llm_model="google/gemini-2.5-flash",
            expert_csv=r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\taxonomy\wiki\pipeline\06_cluster\digital\expert_validation_set.csv",
        )
        _cmd_validate_clusters(validate_args)


# ------------------------------------------------------------------
# Sub-command: validate-clusters  (step 06b)
# ------------------------------------------------------------------

def _cmd_validate_clusters(args: argparse.Namespace) -> None:
    """Thin adapter: delegates to run_unified_validation() in analysis_cluster_quality."""
    from p2_taxonomy.analysis_cluster_quality import run_unified_validation

    pages_csv = args.pages_csv
    if not pages_csv:
        pages_csv = (
            r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\taxonomy"
            r"\wiki\pipeline\04_filter\digital_technology_taxonomy_filtered.csv"
        )

    run_unified_validation(
        run_dir=Path(args.run_dir),
        pages_csv=Path(pages_csv),
        mode=args.mode,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        embed_model=args.embed_model,
        internal=not getattr(args, "skip_internal", False),
        external=not getattr(args, "skip_external", False),
        llm_verify=getattr(args, "llm_verify", False),
        threshold=args.threshold,
        citation_csv=Path(args.citation_csv) if args.citation_csv else None,
        cpc_digits=args.cpc_digits,
        skip_wiki_category=args.skip_wiki_category,
        skip_modularity=args.skip_modularity,
        skip_coverage=args.skip_coverage,
        min_component_size=args.min_component_size,
        expert_csv=Path(args.expert_csv) if args.expert_csv else None,
        llm_model=getattr(args, "llm_model", "google/gemini-2.5-flash"),
        logical=getattr(args, "logical", False),
        backend=getattr(args, "backend", "openrouter"),
        claude_model=getattr(args, "claude_model", "opus"),
    )


# ------------------------------------------------------------------
# Sub-command: diagnose
# ------------------------------------------------------------------

def _cmd_diagnose(args: argparse.Namespace) -> None:
    from p2_taxonomy.analysis_cluster_quality import run_coverage_analysis

    result = run_coverage_analysis(
        wiki_csv=Path(args.wiki_csv),
        mode=args.mode,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        embed_model=args.embed_model,
        threshold=args.threshold,
        citation_csv=Path(args.citation_csv),
        min_component_size=args.min_component_size,
    )
    print(f"[diagnose] Done. {len(result['analyses'])} views analysed → {result.get('mode')}")


# ------------------------------------------------------------------
# Sub-command: openalex-coverage
# ------------------------------------------------------------------

def _cmd_openalex_coverage(args: argparse.Namespace) -> None:
    from p2_taxonomy.analysis_coverage_openalex import run_openalex_coverage

    result = run_openalex_coverage(
        mode=args.mode,
        xlsx_path=Path(args.xlsx) if args.xlsx else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        embed_model=args.embed_model,
    )
    for src, info in result.get("sources", {}).items():
        ov = info.get("overall", {})
        print(f"  [{src}] mean={ov.get('mean', 0):.4f}  "
              f"median={ov.get('median', 0):.4f}  "
              f">=0.5: {ov.get('pct_ge_0.5', 0):.1f}%")


# ------------------------------------------------------------------
# Sub-command: extract-cpc
# ------------------------------------------------------------------

def _cmd_extract_cpc(args: argparse.Namespace) -> None:
    from p2_taxonomy.data_extraction_patent_cpc import extract_cpc_tables

    paths = extract_cpc_tables(
        output_dir=Path(args.output_dir),
        modes=args.modes.split(),
    )
    for p in paths:
        print(f"[extract-cpc] {p}")


# ------------------------------------------------------------------
# Sub-command: tag-patents
# ------------------------------------------------------------------

def _cmd_tag_patents(args: argparse.Namespace) -> None:
    from p2_taxonomy.analysis_patent_taxonomy import run, expand_xlsx_to_taxonomy_dataframe

    if args.taxonomy_xlsx:
        if not args.filtered_csv:
            raise SystemExit("--filtered-csv is required when using --taxonomy-xlsx")
        tax_df = expand_xlsx_to_taxonomy_dataframe(
            Path(args.taxonomy_xlsx),
            Path(args.filtered_csv),
        )
        run(
            taxonomy_df=tax_df,
            patents_dir=Path(args.patents_dir),
            output_dir=Path(args.output_dir),
            threshold=args.threshold,
            aggregate=args.aggregate,
        )
    else:
        if not args.taxonomy_csv or not args.pages_csv:
            raise SystemExit(
                "Provide either --taxonomy-xlsx (with --filtered-csv) "
                "OR both --taxonomy-csv and --pages-csv"
            )
        run(
            taxonomy_csv=Path(args.taxonomy_csv),
            pages_csv=Path(args.pages_csv),
            patents_dir=Path(args.patents_dir),
            output_dir=Path(args.output_dir),
            threshold=args.threshold,
            aggregate=args.aggregate,
        )


# ------------------------------------------------------------------
# Sub-command: compare-strategies
# ------------------------------------------------------------------

def _cmd_compare_strategies(args: argparse.Namespace) -> None:
    from p2_taxonomy.cluster_strategy_compare import run_strategy_comparison

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        input_csv = Path(args.input)
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = input_csv.parent.parent / "06_cluster" / args.mode / "experiments" / f"compare_{ts}"

    result = run_strategy_comparison(
        input_csv=Path(args.input),
        output_dir=out_dir,
        mode=args.mode,
        strategies=args.strategies,
        skip_coverage=not args.include_coverage,
        semantic_weight=args.semantic_weight,
        edge_weight=args.edge_weight,
        tau_edge=args.tau_edge,
        hub_penalty=args.hub_penalty,
        embed_model=args.embed_model,
        n_levels=args.n_levels,
        seed=args.seed,
        iterations=args.iterations,
        kmins=args.kmins,
        kmaxs=args.kmaxs,
        min_sizes=args.min_sizes,
        split_sizes=args.split_sizes,
        margin_weight=args.margin_weight,
        small_frac_weight=getattr(args, "small_frac_weight", 1.0),
        margin_threshold=args.margin_threshold,
        mega_pct=args.mega_pct,
        mega_abs=args.mega_abs,
        do_label=args.label,
        openrouter_model=args.openrouter_model,
        sleep_s=args.sleep_s,
        max_pages_in_prompt=args.max_pages_in_prompt,
        user_agent=args.user_agent,
    )
    print(f"[compare-strategies] Done → {result['output_dir']}")
    print(f"[compare-strategies] Comparison charts → {result['comparison_dir']}")


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m p2_taxonomy.main",
        description="ST4TE P2 taxonomy -- unified CLI",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- crawl ----
    cr = sub.add_parser("crawl", help="BFS crawl Wikipedia categories from seeds")
    cr.add_argument("--seeds-file", required=True, help="Path to <topic>_seeds.txt")
    cr.add_argument("--descriptions", action="store_true", help="Fetch page descriptions")
    cr.add_argument("--max-depth", type=int, default=4)
    cr.add_argument("--max-categories", type=int, default=100_000)
    cr.add_argument("--output", default=None, help="Output CSV (auto-generated if omitted)")

    # ---- clean ----
    cl = sub.add_parser("clean", help="Run the taxonomy cleaning pipeline")
    cl.add_argument("--input", required=True, help="Raw taxonomy CSV")
    cl.add_argument("--output", required=True, help="Cleaned output CSV")
    cl.add_argument("--mode", choices=["green", "digital", "both"], default="green")
    cl.add_argument("--include-lists", action="store_true")
    cl.add_argument("--sleep", type=float, default=0.1)
    cl.add_argument("--timeout", type=float, default=30.0)
    cl.add_argument("--use-embeddings", action="store_true")
    cl.add_argument("--model", default="AI-Growth-Lab/PatentSBERTa")
    cl.add_argument("--description-cache", default="wiki_description_cache.csv")
    
    # ---- ingest ----
    ig = sub.add_parser("ingest", help="Ingest patent TSV + pre-compute embeddings")
    ig.add_argument("--mode", choices=["digital", "green", "all"], default="all",
                    help="Which patent subset to ingest (default: all)")
    ig.add_argument("--views", nargs="+", choices=["title", "abstract", "combined"],
                    default=["title", "abstract", "combined"],
                    help="Embedding views to pre-compute (default: all three)")
    ig.add_argument("--embed-model", default="AI-Growth-Lab/PatentSBERTa")
    ig.add_argument("--input-tsv", default=None, help="Override default TSV path")
    ig.add_argument("--corpus-root", default=None, help="Override corpus cache root")
    ig.add_argument("--cache-root", default=None, help="Override embedding cache root")

    # ---- match-patents ----
    mp = sub.add_parser("match-patents", help="FAISS patent matching")
    mp.add_argument("--wiki-csv", required=True, help="Cleaned wiki taxonomy CSV (step 02 output)")
    mp.add_argument("--patent-dir", required=True, help="Patent parquet parts directory")
    mp.add_argument("--output-dir", default=None,
                    help="Output directory (default: <wiki-csv-parent>/wiki_to_patent_output)")
    mp.add_argument("--mode", choices=["digital", "green"], default="green",
                    help="Patent subset (for shared embedding cache)")
    mp.add_argument("--embed-model", default="AI-Growth-Lab/PatentSBERTa")

    # ---- filter-matches ----
    fm = sub.add_parser("filter-matches",
                        help="Threshold + dedup + noise + LLM validation (run after match-patents)")
    fm.add_argument("--wiki-csv", required=True, help="Cleaned wiki taxonomy CSV (step 02 output)")
    fm.add_argument("--match-dir", default=None,
                    help="match-patents output dir (default: 03_match/{mode}/)")
    fm.add_argument("--output", default=None,
                    help="Filtered output CSV (auto-derived from wiki-csv if omitted)")
    fm.add_argument("--threshold", type=float, default=0.6,
                    help="Min top1_similarity to keep (default 0.6)")
    fm.add_argument("--validate-llm", action="store_true",
                    help="Run per-row LLM validation (resumable from checkpoint)")
    fm.add_argument("--openrouter-model", default="google/gemini-3-flash-preview")
    fm.add_argument("--sleep-s", type=float, default=0.05)
    fm.add_argument("--mode", choices=["digital", "green", "both"], default="digital",
                    help="Technology domain for LLM validation (default: digital)")

    # ---- page-relevance ----
    pr = sub.add_parser("page-relevance",
                        help="Compute per-page patent relevance (coverage + uniqueness)")
    pr.add_argument("--pages-csv", required=True,
                    help="Filtered wiki pages CSV (step 04 filter-matches output)")
    pr.add_argument("--faiss-index", default=None,
                    help="Path to FAISS patent index (default: 03_match/{mode}/faiss_patent.index)")
    pr.add_argument("--output", default=None,
                    help="Output CSV (default: next to --pages-csv)")
    pr.add_argument("--mode", choices=["digital", "green"], default="digital")
    pr.add_argument("--sim-threshold", type=float, default=0.5,
                    help="Min cosine sim to count a patent as covered (default 0.5)")
    pr.add_argument("--k-search", type=int, default=2000,
                    help="FAISS k neighbours per page (default 2000)")
    pr.add_argument("--embed-model", default="AI-Growth-Lab/PatentSBERTa")

    # ---- diagnose ----
    dg = sub.add_parser("diagnose",
                        help="Coverage diagnostics: how well do filtered wiki pages cover the patent corpus?")
    dg.add_argument("--wiki-csv", required=True,
                    help="Filtered wiki taxonomy CSV (with page_title, description)")
    dg.add_argument("--mode", required=True, choices=["digital", "green"],
                    help="Patent subset (must match pre-computed embeddings)")
    dg.add_argument("--output-dir", default=None,
                    help="Output directory (default: coverage_report_<mode>/ next to wiki CSV)")
    dg.add_argument("--embed-model", default="AI-Growth-Lab/PatentSBERTa")
    dg.add_argument("--threshold", type=float, default=0.5,
                    help="Similarity threshold for coverage analyses (default 0.5)")
    dg.add_argument("--citation-csv",
                    default=r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\patent\state_patent_citations.csv",
                    help="Citation CSV path (default: standard Dropbox location)")
    dg.add_argument("--min-component-size", type=int, default=10,
                    help="Min patents in an unmatched cluster to be reported (default 10)")

    # ---- openalex-coverage ----
    oa = sub.add_parser("openalex-coverage",
                        help="OpenAlex topic coverage: how well do 4,516 topics cover the patent corpus?")
    oa.add_argument("--mode", required=True, choices=["digital", "green"],
                    help="Patent subset (must match pre-computed embeddings)")
    oa.add_argument("--xlsx", default=None,
                    help="Path to OpenAlex topic mapping Excel file (default: standard Dropbox location)")
    oa.add_argument("--output-dir", default=None,
                    help="Output directory (default: same directory as the xlsx file)")
    oa.add_argument("--embed-model", default="AI-Growth-Lab/PatentSBERTa")

    # ---- topics ----
    tp = sub.add_parser("topics", help="BERTopic patent topic modelling")
    tp.add_argument("--mode", required=True, choices=["digital", "green"])
    tp.add_argument("--views", nargs="+", choices=["title", "abstract", "combined"], default=["title"])
    tp.add_argument("--layers", nargs="+", type=int, default=[12, 50, 150])
    tp.add_argument("--embed-model", default="AI-Growth-Lab/PatentSBERTa")
    tp.add_argument("--sample-n", type=int, default=300_000)
    tp.add_argument("--exposure", choices=["on", "off"], default="off")
    tp.add_argument("--exposure-topk", type=int, default=3)
    tp.add_argument("--anchor-guided", choices=["on", "off"], default="on")
    tp.add_argument("--anchor-alpha", type=float, default=3.0)
    tp.add_argument("--label", action="store_true")
    tp.add_argument("--plot", action="store_true")
    tp.add_argument("--openrouter-model", default="google/gemini-3-flash-preview")
    tp.add_argument("--sleep-s", type=float, default=.1)
    tp.add_argument("--max-docs-in-prompt", type=int, default=40)
    tp.add_argument("--user-agent", default="patent-topics/1.0 (research use)")

    # ---- cluster ----
    ac = sub.add_parser("cluster", help="Adaptive Leiden clustering")
    ac.add_argument("--input", required=True, help="Cleaned/filtered wiki taxonomy CSV")
    ac.add_argument("--mode", choices=["digital", "green"], default="digital",
                    help="Technology domain (used for output folder, default: digital)")
    ac.add_argument("--output-dir", default=None, help="Output directory (auto if omitted)")
    ac.add_argument("--semantic-weight", type=float, default=1.0)
    ac.add_argument("--edge-weight", type=float, default=0.0)
    ac.add_argument("--tau-edge", type=float, default=0.20)
    ac.add_argument("--hub-penalty", action="store_true")
    ac.add_argument("--embed-model", default="AI-Growth-Lab/PatentSBERTa")
    ac.add_argument("--n-levels", type=int, default=2)
    ac.add_argument("--seed", type=int, default=42)
    ac.add_argument("--iterations", type=int, default=50)
    ac.add_argument("--kmins", default="8,5,2")
    ac.add_argument("--kmaxs", default="12,8,6")
    ac.add_argument("--min-sizes", default="500,25,8")
    ac.add_argument("--split-sizes", default="3000,100,30")
    ac.add_argument("--margin-weight", type=float, default=10.0,
                    help="Weight for semantic margin in gamma search (default: 10.0)")
    ac.add_argument("--small-frac-weight", type=float, default=1,
                    help="Weight for small fraction in gamma search (default: 1.0)")
    ac.add_argument("--strategy", default="adaptive",
                    choices=["adaptive", "simple", "margin_adaptive", "mega_adaptive"],
                    help="Clustering strategy (default: adaptive)")
    ac.add_argument("--margin-threshold", type=float, default=0.0,
                    help="Min per-cluster margin for margin_adaptive strategy (default: 0.0)")
    ac.add_argument("--mega-pct", type=float, default=0.9,
                    help="Mega-cluster percentage threshold for mega_adaptive (default: 0.9)")
    ac.add_argument("--mega-abs", type=int, default=200,
                    help="Mega-cluster absolute size threshold for mega_adaptive (default: 200)")
    ac.add_argument("--mega-threshold", type=int, default=0,
                    help="Subgraph size above which local tau pruning is applied (0=disabled)")
    ac.add_argument("--local-tau", type=float, default=0.0,
                    help="Extra edge-weight threshold applied inside flagged clusters (0=disabled)")
    ac.add_argument("--resplit", action="store_true",
                    help="Post-hoc collapse + resplit MEGA/WEAK flagged clusters after initial clustering")
    ac.add_argument("--mega-frac", type=float, default=0.9,
                    help="Fraction threshold for MEGA flag during resplit (default: 0.9)")
    ac.add_argument("--mega-min-size", type=int, default=10,
                    help="Min cluster size for MEGA flag during resplit (default: 10)")
    ac.add_argument("--weak-margin", type=float, default=0.0,
                    help="Margin threshold for WEAK flag during resplit (default: 0.0)")
    ac.add_argument("--weak-min-size", type=int, default=10,
                    help="Min cluster size for WEAK flag during resplit (default: 10)")
    ac.add_argument("--prune-pct", type=float, default=0.0,
                    help="Percentile-based edge pruning (0=disabled). Prune bottom X%% globally + per-subgraph.")
    ac.add_argument("--plot", action="store_true")
    ac.add_argument("--label", action="store_true")
    ac.add_argument("--openrouter-model", default="google/gemini-3-flash-preview")
    ac.add_argument("--sleep-s", type=float, default=.1)
    ac.add_argument("--max-pages-in-prompt", type=int, default=40)
    ac.add_argument("--user-agent", default="wiki-link-extractor/1.0 (research use)")
    ac.add_argument("--validate", action="store_true",
                    help="Run post-cluster validation (page-distances + margins + plots) after clustering")

    # ---- validate-clusters (step 06b) ----
    vc = sub.add_parser("validate-clusters",
                        help="Post-cluster validation: page-distances + cluster margins + coverage → <run-dir>/validation/")
    vc.add_argument("--run-dir", required=True,
                    help="Cluster output dir (contains page_assignments + taxonomy CSVs)")
    vc.add_argument("--pages-csv", default=None,
                    help="Input CSV with page_title + description (default: standard filtered CSV)")
    vc.add_argument("--mode", choices=["digital", "green"], default="digital",
                    help="Patent subset (default: digital)")
    vc.add_argument("--output-dir", default=None,
                    help="Output directory (default: <run-dir>/validation/)")
    vc.add_argument("--embed-model", default="AI-Growth-Lab/PatentSBERTa")
    vc.add_argument("--threshold", type=float, default=0.5,
                    help="Similarity threshold for coverage analyses (default 0.5)")
    vc.add_argument("--citation-csv",
                    default=r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\patent\state_patent_citations.csv",
                    help="Citation CSV path")
    vc.add_argument("--min-component-size", type=int, default=10,
                    help="Min patents in an unmatched cluster to be reported (default 10)")
    vc.add_argument("--skip-coverage", action="store_true",
                    help="Skip post-cluster patent-wiki coverage analysis")
    vc.add_argument("--skip-internal", action="store_true",
                    help="Skip internal validation (page distances, coherence, margins)")
    vc.add_argument("--skip-external", action="store_true",
                    help="Skip CPC homogeneity + citation coherence cluster validation")
    vc.add_argument("--expert-csv",
                    default=r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\taxonomy\wiki\pipeline\06_cluster\digital\expert_validation_set.csv",
                    help="Expert gold-standard validation CSV (default: digital expert set)")
    vc.add_argument("--cpc-digits", default="3,4",
                    help="Comma-separated CPC digit levels to analyze (default: '3,4')")
    vc.add_argument("--skip-wiki-category", action="store_true",
                    help="Skip wiki category share analysis")
    vc.add_argument("--skip-modularity", action="store_true",
                    help="Skip Newman modularity Q computation")
    vc.add_argument("--llm-verify", action="store_true",
                    help="Run optional LLM cluster verification (costs API credits)")
    vc.add_argument("--llm-model", default="google/gemini-2.5-flash",
                    help="LLM model for cluster verification (default: gemini-2.5-flash)")
    vc.add_argument("--logical", action="store_true",
                    help="Run LLM-based logical consistency validation (subsumption, sibling exclusivity, cross-cutting)")
    vc.add_argument("--backend", choices=["openrouter", "claude-code"], default="openrouter",
                    help="LLM backend for logical consistency (default: openrouter)")
    vc.add_argument("--claude-model", default="opus",
                    help="Claude model alias when backend=claude-code (default: opus)")

    # ---- extract-cpc ----
    ec = sub.add_parser("extract-cpc",
                        help="Extract 4-char CPC subclass codes with titles and descriptions")
    ec.add_argument("--modes", default="digital green",
                    help="Space-separated patent modes to process (default: 'digital green')")
    ec.add_argument("--output-dir",
                    default=r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\taxonomy\patent_cpc",
                    help="Output directory for CSV files")

    # ---- tag-patents ----
    tp2 = sub.add_parser("tag-patents",
                         help="Tag patents with taxonomy labels via FAISS matching + per-patent profile")
    tp2.add_argument("--taxonomy-csv",
                     help="Long-format taxonomy CSV with [root, L1, L2, page, root_given] (mutually exclusive with --taxonomy-xlsx)")
    tp2.add_argument("--pages-csv",
                     help="Pages CSV with [page_title, description] (required with --taxonomy-csv)")
    tp2.add_argument("--taxonomy-xlsx",
                     help="v3-style taxonomy xlsx with sheet 'taxonomy' and columns [root, L1, L2, pages]")
    tp2.add_argument("--filtered-csv",
                     default=r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\taxonomy\wiki\pipeline\04_filter\digital_technology_taxonomy_filtered.csv",
                     help="Filtered descriptions CSV (provides page_title + description; required with --taxonomy-xlsx)")
    tp2.add_argument("--patents-dir",
                     default=r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\patent\_corpus_cache_v2\digital",
                     help="Patent parquet parts directory (default: digital corpus cache)")
    tp2.add_argument("--output-dir", required=True,
                     help="Output directory for matches + per-patent parquet")
    tp2.add_argument("--threshold", type=float, default=0.5,
                     help="Min similarity to include patent-page match (default: 0.5)")
    tp2.add_argument("--aggregate", action="store_true",
                     help="Also write legacy page/L2/CPC aggregations (off by default)")

    # ---- compare-strategies ----
    cs = sub.add_parser("compare-strategies",
                        help="Run all clustering strategies, validate each, generate comparison")
    cs.add_argument("--input", required=True, help="Cleaned/filtered wiki taxonomy CSV")
    cs.add_argument("--mode", choices=["digital", "green"], default="digital",
                    help="Technology domain (default: digital)")
    cs.add_argument("--output-dir", default=None,
                    help="Output directory (auto-timestamped if omitted)")
    cs.add_argument("--strategies", nargs="+",
                    default=["adaptive", "simple", "margin_adaptive", "mega_adaptive"],
                    choices=["adaptive", "simple", "margin_adaptive", "mega_adaptive"],
                    help="Strategies to compare (default: all four)")
    cs.add_argument("--include-coverage", action="store_true",
                    help="Run patent coverage analysis for each strategy (slow, off by default)")
    cs.add_argument("--semantic-weight", type=float, default=1.0)
    cs.add_argument("--edge-weight", type=float, default=0.0)
    cs.add_argument("--tau-edge", type=float, default=0.20)
    cs.add_argument("--hub-penalty", action="store_true")
    cs.add_argument("--embed-model", default="AI-Growth-Lab/PatentSBERTa")
    cs.add_argument("--n-levels", type=int, default=2)
    cs.add_argument("--seed", type=int, default=42)
    cs.add_argument("--iterations", type=int, default=50)
    cs.add_argument("--kmins", default="8,5,2")
    cs.add_argument("--kmaxs", default="12,8,6")
    cs.add_argument("--min-sizes", default="500,25,8")
    cs.add_argument("--split-sizes", default="3000,100,30")
    cs.add_argument("--margin-weight", type=float, default=10.0)
    cs.add_argument("--small-frac-weight", type=float, default=1.0)
    cs.add_argument("--margin-threshold", type=float, default=0.0,
                    help="Min margin for margin_adaptive strategy (default: 0.0)")
    cs.add_argument("--mega-pct", type=float, default=0.9,
                    help="Mega-cluster pct threshold for mega_adaptive (default: 0.9)")
    cs.add_argument("--mega-abs", type=int, default=200)
    cs.add_argument("--label", action="store_true",
                    help="Run LLM labeling for each strategy")
    cs.add_argument("--openrouter-model", default="google/gemini-3-flash-preview")
    cs.add_argument("--sleep-s", type=float, default=0.1)
    cs.add_argument("--max-pages-in-prompt", type=int, default=40)
    cs.add_argument("--user-agent", default="wiki-link-extractor/1.0 (research use)")

    return p


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "crawl": _cmd_crawl,
        "clean": _cmd_clean,
        "ingest": _cmd_ingest,
        "match-patents": _cmd_match_patents,
        "filter-matches": _cmd_filter_matches,
        "page-relevance": _cmd_page_relevance,
        "diagnose": _cmd_diagnose,
        "openalex-coverage": _cmd_openalex_coverage,
        "topics": _cmd_topics,
        "cluster": _cmd_cluster,
        "validate-clusters": _cmd_validate_clusters,
        "extract-cpc": _cmd_extract_cpc,
        "tag-patents": _cmd_tag_patents,
        "compare-strategies": _cmd_compare_strategies,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        raise SystemExit(1)

    handler(args)


if __name__ == "__main__":
    main()
