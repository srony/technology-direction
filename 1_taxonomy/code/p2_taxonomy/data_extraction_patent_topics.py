"""
p2_taxonomy/data_extraction_patent_topics.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BERTopic patent topic modelling -- cleaner module interface wrapping
the logic in ``03_patent_topics.py``.

Provides:
- ``PATENT_STOPWORDS`` and ``make_vectorizer``
- ``compute_embeddings``          -- cached embeddings with optional anchor guidance
- ``ensure_base_model``           -- base BERTopic model caching
- ``build_taxonomy_tree_multi_layers`` -- multi-layer strict tree taxonomy
- ``assign_all``                  -- transform full corpus
- ``RunConfig`` dataclass
- ``main_run``                    -- orchestrator

Text / CPC helpers and corpus ingest are in ``shared.patent_cache``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import hdbscan
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from shared.patent_cache import device_auto
from shared.utils import ensure_dir
from shared.patent_cache import (
    build_view_text,
    extract_cpc_codes_before_space,
    coarse_prefix,
    _list_part_files,
    _iter_parts,
    _fast_row_count,
    ingest_corpus,
    load_embeddings as _load_base_embeddings,
    compute_and_cache_embeddings,
    CACHE_ROOT,
    CORPUS_ROOT,
)


# ------------------------------------------------------------------
# Default paths (user setup -- override via RunConfig / CLI)
# ------------------------------------------------------------------

OUTPUT_ROOT = r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\patent\topic_outputs"
VIZ_ROOT = r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\visualisations\patent\topic_outputs"


# ------------------------------------------------------------------
# Defaults for large corpora
# ------------------------------------------------------------------

SEED = 42
CHUNKSIZE = 300_000
SAMPLE_N = 300_000

EMBED_BATCH = 256
TRANSFORM_BATCH = 100_000

DEFAULT_MIN_TOPIC_SIZE = 50
DEFAULT_MIN_SAMPLES = 10

TOP_N_WORDS = 12

DEFAULT_EMBED_MODEL = "AI-Growth-Lab/PatentSBERTa"

UMAP_N_NEIGHBORS = 25
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"

HDB_METRIC = "euclidean"
HDB_SELECTION_METHOD = "eom"


# ------------------------------------------------------------------
# Anchor-guided seeds (L1 anchors)
# ------------------------------------------------------------------

DEFAULT_L1_SEEDS: Dict[str, List[str]] = {
    "Artificial Intelligence": [
        "machine learning", "deep learning", "neural network", "inference",
        "training", "classifier", "reinforcement learning", "generative AI",
        "Natural Language Processing", "computer vision",
    ],
    "Big Data": [
        "big data", "data pipeline", "data lake", "distributed processing",
        "spark", "hadoop", "data warehouse",
    ],
    "Blockchain": [
        "blockchain", "distributed ledger", "smart contract", "cryptographic",
    ],
    "Cloud Computing": [
        "cloud", "virtual machine", "container", "orchestration",
        "serverless", "multi-tenant",
    ],
    "Internet of Things": [
        "iot", "sensor network", "edge device", "connected device",
        "telemetry", "actuator",
    ],
    "Quantum Computing": [
        "quantum", "qubit", "quantum circuit", "quantum gate",
        "quantum error correction",
    ],
    "Robotics": [
        "robot", "robotic arm", "autonomous", "navigation",
        "manipulation", "actuation",
    ],
    "Smart Cities": [
        "smart city", "traffic", "city management", "public transport",
    ],
    "Virtual Reality": [
        "virtual reality", "augmented reality", "head-mounted display",
        "immersive",
    ],
}

GREEN_L1_SEEDS: Dict[str, List[str]] = {
    "Solar Energy": [
        "solar panel", "photovoltaic", "concentrated solar power",
        "solar thermal", "perovskite solar cell", "solar irradiance",
    ],
    "Wind Energy": [
        "wind turbine", "offshore wind", "wind farm", "wind power",
        "blade aerodynamics", "wind energy conversion",
    ],
    "Hydropower & Marine": [
        "hydroelectric", "tidal energy", "wave energy", "run-of-river",
        "pumped storage", "ocean current",
    ],
    "Energy Storage & Hydrogen": [
        "battery storage", "lithium-ion battery", "solid-state battery",
        "fuel cell", "hydrogen storage", "electrolysis", "green hydrogen",
        "redox flow battery",
    ],
    "Electric Vehicles": [
        "electric vehicle", "EV battery", "battery electric vehicle",
        "charging station", "regenerative braking", "powertrain electrification",
        "vehicle-to-grid",
    ],
    "Energy Efficiency & Buildings": [
        "energy efficiency", "insulation", "heat pump", "LED lighting",
        "building retrofit", "passive house", "thermal management",
        "demand response",
    ],
    "Carbon Capture & Storage": [
        "carbon capture", "CO2 sequestration", "direct air capture",
        "carbon storage", "post-combustion capture", "geological storage",
        "carbon utilisation",
    ],
    "Smart Grid & Power Electronics": [
        "smart grid", "grid management", "power electronics",
        "inverter", "voltage regulation", "distributed generation",
        "microgrid", "energy management system",
    ],
    "Sustainable Agriculture & Land Use": [
        "precision agriculture", "crop monitoring", "soil health",
        "nitrogen fixation", "irrigation efficiency", "agroforestry",
        "biofuel feedstock",
    ],
    "Waste & Circular Economy": [
        "recycling", "waste treatment", "composting", "circular economy",
        "waste-to-energy", "plastic recycling", "biomass conversion",
    ],
    "Water Treatment & Management": [
        "water purification", "wastewater treatment", "desalination",
        "water recycling", "membrane filtration", "reverse osmosis",
    ],
}


def _seeds_for_mode(mode: str) -> Dict[str, List[str]]:
    """Return the appropriate L1 anchor seeds for the given mode."""
    if mode == "green":
        return GREEN_L1_SEEDS
    return DEFAULT_L1_SEEDS


# ------------------------------------------------------------------
# Patent boilerplate stopwords
# ------------------------------------------------------------------

PATENT_STOPWORDS = {
    "method", "methods", "system", "systems", "apparatus", "device", "devices",
    "means", "process", "processes", "using", "use", "used", "based",
    "including", "include", "includes", "according", "wherein", "thereof",
    "providing", "provide", "provides", "provided",
    "plurality", "portion", "unit", "units", "module", "modules",
    "configured", "configuration",
    "embodiment", "embodiments", "example", "examples",
    "present", "disclosure", "invention", "inventions",
    "implementation", "implementations",
    "readable", "medium", "media", "nontransitory", "transitory",
    "instruction", "instructions",
    "equipment",
    "terminal", "dialogue", "interaction",
    "transmitting", "transmission",
    "message", "messages", "service", "services",
    "management",
    "generation", "construction",
    "determination", "calculation", "estimation",
}


def make_vectorizer() -> CountVectorizer:
    """Build the patent-aware CountVectorizer."""
    custom = set(ENGLISH_STOP_WORDS).union(PATENT_STOPWORDS)
    return CountVectorizer(
        stop_words=sorted(custom),
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
        min_df=5,
        max_df=0.85,
    )


def _pick_sample_indices(n: int, k: int, seed: int = 42) -> np.ndarray:
    k = min(k, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    idx.sort()
    return idx


# ------------------------------------------------------------------
# Run directories
# ------------------------------------------------------------------

def _run_dirs(mode: str, run_tag: str) -> Dict[str, str]:
    out_base = os.path.join(OUTPUT_ROOT, f"outputs_{mode}", run_tag)
    viz_base = os.path.join(VIZ_ROOT, mode, run_tag)

    dirs = {
        "out_base": out_base,
        "parts": os.path.join(CORPUS_ROOT, mode),
        "models": os.path.join(out_base, "models"),
        "assign": os.path.join(out_base, "assignments"),
        "taxonomy": os.path.join(out_base, "taxonomy"),
        "summaries": os.path.join(out_base, "summaries"),
        "viz": viz_base,
        "base_models": os.path.join(OUTPUT_ROOT, f"outputs_{mode}", "_base_models"),
    }
    for d in dirs.values():
        ensure_dir(d)
    ensure_dir(CACHE_ROOT)
    ensure_dir(CORPUS_ROOT)
    return dirs


# ------------------------------------------------------------------
# Embeddings cache + optional anchor guidance
# ------------------------------------------------------------------

def _anchor_cache_fingerprint(seeds: Dict[str, List[str]], alpha: float) -> str:
    payload = {"alpha": float(alpha), "seeds": seeds}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _anchor_embedding_cache_paths(
    mode: str, view: str, embed_model: str, anchor_alpha: float,
    anchor_seeds: Dict[str, List[str]],
) -> Dict[str, str]:
    """Cache paths for anchor-guided embeddings (separate from base)."""
    afp = _anchor_cache_fingerprint(anchor_seeds, anchor_alpha)
    raw = f"{mode}|{view}|{embed_model}|anch_{afp}"
    cid = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    base = os.path.join(CACHE_ROOT, f"emb_{cid}")
    return {"meta": base + ".meta.json", "ids": base + ".appln_id.npy", "emb": base + ".embeddings.npy"}


def compute_embeddings(
    parts_dir: str,
    mode: str,
    view: str,
    embed_model: str,
    anchor_guided: bool = False,
    anchor_alpha: float = 3.0,
    anchor_seeds: Optional[Dict[str, List[str]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (or load cached) patent embeddings with optional anchor guidance.

    When ``anchor_guided`` is False, delegates entirely to
    ``shared.patent_cache.load_embeddings``.  When True, loads the base
    embeddings and appends anchor-similarity columns, caching the result
    separately.
    """
    if not anchor_guided:
        return _load_base_embeddings(mode, view, embed_model, CACHE_ROOT, CORPUS_ROOT)

    seeds = anchor_seeds if anchor_seeds is not None else _seeds_for_mode(mode)
    paths = _anchor_embedding_cache_paths(mode, view, embed_model, anchor_alpha, seeds)

    parts = _list_part_files(parts_dir)
    n_rows = _fast_row_count(parts)

    # Check anchor-guided cache
    if os.path.exists(paths["meta"]) and os.path.exists(paths["ids"]) and os.path.exists(paths["emb"]):
        try:
            meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
            if (
                meta.get("rows") == n_rows
                and meta.get("mode") == mode
                and meta.get("view") == view
                and meta.get("embed_model") == embed_model
                and bool(meta.get("anchor_guided", False))
            ):
                ids = np.load(paths["ids"], allow_pickle=False)
                emb = np.load(paths["emb"], mmap_mode="r")
                if len(ids) == n_rows and emb.shape[0] == n_rows:
                    print(f"[embed-cache] HIT anchor-guided view={view}: {paths['emb']}")
                    return emb, ids
        except Exception:
            pass

    # Load base embeddings from shared cache
    base_emb, base_ids = _load_base_embeddings(mode, view, embed_model, CACHE_ROOT, CORPUS_ROOT)

    print(f"[embed-cache] Building anchor-guided embeddings for view={view}.")
    dev = device_auto()
    embedder = SentenceTransformer(embed_model, device=dev)

    anchor_names = list(seeds.keys())
    embs_list = []
    for name in anchor_names:
        phrases = seeds[name]
        e = embedder.encode(phrases, batch_size=64, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        v = e.mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        embs_list.append(v)
    anchor_embs = np.vstack(embs_list).astype(np.float32)
    del embedder

    dim0 = base_emb.shape[1]
    dim = dim0 + len(seeds)

    ids_all = np.array(base_ids, dtype=np.int64)
    emb_mm = np.lib.format.open_memmap(paths["emb"], mode="w+", dtype=np.float32, shape=(n_rows, dim))

    chunk_size = 50_000
    for start in tqdm(range(0, n_rows, chunk_size), desc=f"[embed] anchor ({view})"):
        end = min(start + chunk_size, n_rows)
        base_chunk = np.asarray(base_emb[start:end], dtype=np.float32)
        sims = base_chunk @ anchor_embs.T
        sims = (float(anchor_alpha) * sims).astype(np.float32)
        emb_mm[start:end, :] = np.concatenate([base_chunk, sims], axis=1)
        del base_chunk, sims
        gc.collect()

    emb_mm.flush()
    np.save(paths["ids"], ids_all, allow_pickle=False)

    meta = {
        "mode": mode, "view": view, "embed_model": embed_model,
        "rows": int(n_rows), "dim": int(dim), "normalized": True,
        "anchor_guided": True,
        "anchor_alpha": float(anchor_alpha),
        "anchor_names": anchor_names,
    }
    Path(paths["meta"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[embed-cache] Saved anchor-guided: {paths['emb']}")

    emb = np.load(paths["emb"], mmap_mode="r")
    ids = np.load(paths["ids"], allow_pickle=False)
    return emb, ids


# ------------------------------------------------------------------
# 3) Base model caching
# ------------------------------------------------------------------

def _base_model_cache_dir(
    mode: str, view: str, embed_model: str, sample_n: int,
    anchor_guided: bool, anchor_alpha: float,
    anchor_seeds: Optional[Dict[str, List[str]]] = None,
) -> str:
    seeds = anchor_seeds if anchor_seeds is not None else _seeds_for_mode(mode)
    anchor_tag = "plain"
    if anchor_guided:
        anchor_tag = f"anch_{_anchor_cache_fingerprint(seeds, anchor_alpha)}"
    raw = (
        f"{mode}|{view}|{embed_model}|{anchor_tag}|"
        f"seed={SEED}|sample={sample_n}|"
        f"umap_n={UMAP_N_NEIGHBORS}|umap_c={UMAP_N_COMPONENTS}|"
        f"umap_min_dist={UMAP_MIN_DIST}|umap_metric={UMAP_METRIC}|"
        f"hdb_min_cluster={DEFAULT_MIN_TOPIC_SIZE}|hdb_min_samples={DEFAULT_MIN_SAMPLES}|"
        f"hdb_metric={HDB_METRIC}|hdb_sel={HDB_SELECTION_METHOD}|"
        f"vec=english+patent|ngram=1-2|min_df=5|max_df=0.85"
    )
    bid = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    safe = embed_model.replace("/", "_")
    return os.path.join(OUTPUT_ROOT, f"outputs_{mode}", "_base_models", f"base_{view}_{safe}_{bid}")


def _make_base_model(calculate_probabilities: bool) -> BERTopic:
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC, random_state=SEED,
    )
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=DEFAULT_MIN_TOPIC_SIZE, min_samples=DEFAULT_MIN_SAMPLES,
        metric=HDB_METRIC, cluster_selection_method=HDB_SELECTION_METHOD,
        prediction_data=True,
    )
    return BERTopic(
        embedding_model=None, umap_model=umap_model, hdbscan_model=hdb,
        vectorizer_model=make_vectorizer(), top_n_words=TOP_N_WORDS,
        nr_topics=None, calculate_probabilities=calculate_probabilities, verbose=True,
    )


def _collect_sample_docs(parts_dir: str, view: str, idx: np.ndarray) -> List[str]:
    parts = _list_part_files(parts_dir)
    docs: List[str] = []
    cursor = 0
    for df in tqdm(
        _iter_parts(parts, ["appln_title", "appln_abstract"]),
        total=len(parts), desc=f"[fit] collect sample docs ({view})",
    ):
        n = len(df)
        mask = np.isin(np.arange(cursor, cursor + n), idx)
        if mask.any():
            sub = df.loc[mask, ["appln_title", "appln_abstract"]]
            docs.extend(build_view_text(sub, view).tolist())
        cursor += n
    return docs


def ensure_base_model(
    parts_dir: str,
    mode: str,
    view: str,
    embed_model: str,
    emb_all: np.ndarray,
    sample_n: int = SAMPLE_N,
    anchor_guided: bool = False,
    anchor_alpha: float = 3.0,
    anchor_seeds: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, List[str], np.ndarray]:
    """Fit (or load cached) the BERTopic base model on a sample."""
    bdir = _base_model_cache_dir(mode, view, embed_model, sample_n, anchor_guided, anchor_alpha, anchor_seeds)
    meta_path = os.path.join(bdir, "meta.json")

    n_all = emb_all.shape[0]
    idx = _pick_sample_indices(n_all, sample_n, seed=SEED)

    docs_sample = _collect_sample_docs(parts_dir, view, idx)
    emb_sample = np.asarray(emb_all[idx, :], dtype=np.float32)

    if os.path.exists(meta_path):
        print(f"[base] HIT base model: {bdir}")
        return bdir, docs_sample, emb_sample

    print(f"[base] MISS. Fitting base model for view={view}")
    ensure_dir(bdir)
    model = _make_base_model(calculate_probabilities=False)
    model.fit(docs_sample, embeddings=emb_sample)
    model.save(bdir, serialization="safetensors")

    meta = {
        "mode": mode, "view": view, "embed_model": embed_model,
        "anchor_guided": bool(anchor_guided),
        "seed": SEED, "sample_n": int(len(docs_sample)),
    }
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[base] Saved base model: {bdir}")
    return bdir, docs_sample, emb_sample


# ------------------------------------------------------------------
# 4) Multi-layer strict tree taxonomy
# ------------------------------------------------------------------

def _topic_keywords_label(model: BERTopic, topic_id: int, n_words: int = 6) -> str:
    kw = model.get_topic(topic_id) or []
    words = [w for w, _ in kw[:n_words]]
    return " / ".join(words) if words else f"Topic {topic_id}"


def _reduce_from_base(base_dir: str, docs_sample: List[str], target_k: int, calc_probs: bool) -> BERTopic:
    m = BERTopic.load(base_dir)
    m.calculate_probabilities = bool(calc_probs)
    m = m.reduce_topics(docs_sample, nr_topics=int(target_k))
    try:
        m.update_topics(docs_sample, vectorizer_model=make_vectorizer())
    except Exception:
        pass
    return m


def build_taxonomy_tree_multi_layers(
    base_dir: str,
    docs_sample: List[str],
    layers: List[int],
    emb_sample: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, str]], Dict[int, Dict[int, int]], Dict[int, Dict[int, str]], Dict[int, BERTopic]]:
    """Build a strict hierarchical tree across the requested layer sizes.

    Returns ``(taxonomy_df, leaf_lookup, mapping, labels_by_layer, models_by_layer)``.
    """
    layers = sorted(int(x) for x in layers)
    if len(layers) < 2:
        raise ValueError("--layers must contain at least 2 values.")

    models_by_layer: Dict[int, BERTopic] = {}
    for k in layers:
        models_by_layer[k] = _reduce_from_base(base_dir, docs_sample, k, calc_probs=False)
        n = len([t for t in models_by_layer[k].get_topics() if t != -1])
        print(f"[taxonomy] Reduced to L{k}: topics(excl -1)={n}")

    # Transform sample docs through each layer
    topics_by_layer: Dict[int, np.ndarray] = {}
    for k in layers:
        m = models_by_layer[k]
        t, _ = m.transform(docs_sample, embeddings=emb_sample) if emb_sample is not None else m.transform(docs_sample)
        topics_by_layer[k] = np.asarray(t, dtype=np.int32)

    # Build child->parent mapping via hard majority vote
    mapping: Dict[int, Dict[int, int]] = {}
    for i in range(len(layers) - 1):
        parent_k, child_k = layers[i], layers[i + 1]
        parent_topics = topics_by_layer[parent_k]
        child_topics = topics_by_layer[child_k]

        parent_non_out = parent_topics[parent_topics != -1]
        fallback = int(pd.Series(parent_non_out).value_counts().idxmax())

        m: Dict[int, int] = {}
        child_ids = sorted(t for t in models_by_layer[child_k].get_topics() if t != -1)
        for tid in child_ids:
            mask = (child_topics == int(tid)) & (parent_topics != -1)
            if not np.any(mask):
                m[int(tid)] = fallback
            else:
                m[int(tid)] = int(pd.Series(parent_topics[mask]).value_counts().idxmax())
        mapping[child_k] = m

    # Keyword labels
    labels_by_layer: Dict[int, Dict[int, str]] = {}
    for k in layers:
        model = models_by_layer[k]
        ids = sorted(t for t in model.get_topics() if t != -1)
        labels_by_layer[k] = {tid: _topic_keywords_label(model, tid) for tid in ids}

    # Build taxonomy table
    leaf_k = layers[-1]
    leaf_ids = sorted(t for t in models_by_layer[leaf_k].get_topics() if t != -1)
    leaf_lookup: Dict[int, Dict[str, str]] = {}
    rows: list[dict[str, str]] = []

    for leaf_id in leaf_ids:
        cur = int(leaf_id)
        row: Dict[str, str] = {f"L{leaf_k}": labels_by_layer[leaf_k].get(cur, f"L{leaf_k} {cur}")}
        for j in range(len(layers) - 2, -1, -1):
            parent_k = layers[j]
            child_k = layers[j + 1]
            parent_id = mapping[child_k].get(cur)
            if parent_id is None:
                row = {}
                break
            row[f"L{parent_k}"] = labels_by_layer[parent_k].get(parent_id, f"L{parent_k} {parent_id}")
            cur = parent_id
        if row and all(f"L{k}" in row for k in layers):
            rows.append(row)
            leaf_lookup[int(leaf_id)] = row

    taxonomy_df = pd.DataFrame(rows).drop_duplicates()
    taxonomy_df = taxonomy_df[[f"L{k}" for k in layers]]
    return taxonomy_df, leaf_lookup, mapping, labels_by_layer, models_by_layer


# ------------------------------------------------------------------
# 5) Transform full corpus
# ------------------------------------------------------------------

def assign_all(
    parts_dir: str,
    view: str,
    emb_all: np.ndarray,
    ids_all: np.ndarray,
    leaf_layer: int,
    model_leaf: BERTopic,
    leaf_lookup: Dict[int, Dict[str, str]],
    layers: List[int],
    out_assign_dir: str,
    do_exposure: bool = False,
    exposure_topk: int = 3,
) -> Tuple[str, Optional[str], Dict[str, float], pd.DataFrame]:
    """Transform full corpus through the leaf model and write primary assignments.

    Returns ``(primary_path, exposure_path, stats_dict, primary_df)``.
    """
    ensure_dir(out_assign_dir)
    primary_parts_dir = os.path.join(out_assign_dir, f"_primary_parts_{view}")
    ensure_dir(primary_parts_dir)

    parts = _list_part_files(parts_dir)
    n_all = emb_all.shape[0]

    cursor = 0
    part_idx = 0
    outliers_raw = 0
    non_outliers_raw = 0

    # Per-layer topic->label lookup, built once (was rebuilt per cell via prim.at).
    # {} / missing topic / missing layer-key all map to None, matching the old
    # `None`-init + `if row:` guard exactly (verified on outlier/unknown/empty cases).
    _lut = {k: {tid: row.get(f"L{k}") for tid, row in leaf_lookup.items()} for k in layers}

    for df in tqdm(
        _iter_parts(parts, ["appln_id", "appln_title", "appln_abstract", "cpc_coarse_list"]),
        total=len(parts), desc=f"[assign] parts ({view})",
    ):
        n = len(df)
        texts = build_view_text(df, view).tolist()
        emb_batch = np.asarray(emb_all[cursor : cursor + n, :], dtype=np.float32)

        topics, probs = model_leaf.transform(texts, embeddings=emb_batch)
        assigned = np.asarray(topics, dtype=np.int32)
        raw_outlier = assigned == -1

        prim = pd.DataFrame({
            "appln_id": df["appln_id"].astype(np.int64).values,
            f"topic_L{leaf_layer}": assigned,
        })
        prim["is_outlier_raw"] = raw_outlier.astype(bool)

        for k in layers:
            prim[f"L{k}_label"] = [_lut[k].get(int(t)) for t in assigned]

        outliers_raw += int(prim["is_outlier_raw"].sum())
        non_outliers_raw += int((~prim["is_outlier_raw"]).sum())

        prim.to_parquet(os.path.join(primary_parts_dir, f"part_{part_idx:05d}.parquet"), index=False)
        cursor += n
        part_idx += 1
        del df, texts, emb_batch, prim
        gc.collect()

    # Combine parts
    primary_out = os.path.join(out_assign_dir, f"patent_primary_{view}.parquet")
    primary_files = sorted(Path(primary_parts_dir).glob("part_*.parquet"))
    df_primary = pd.concat([pd.read_parquet(f) for f in primary_files], ignore_index=True)
    df_primary.to_parquet(primary_out, index=False)

    # Cleanup temp parts
    for f in primary_files:
        try:
            f.unlink()
        except Exception:
            pass
    try:
        Path(primary_parts_dir).rmdir()
    except Exception:
        pass

    total = outliers_raw + non_outliers_raw
    stats = {
        "n_docs": int(total),
        "n_outliers_raw": int(outliers_raw),
        "outlier_rate_raw": float(outliers_raw / total) if total else 0.0,
    }
    return primary_out, None, stats, df_primary


# ------------------------------------------------------------------
# RunConfig + orchestrator
# ------------------------------------------------------------------

@dataclass
class RunConfig:
    """Configuration for a BERTopic patent topics run."""

    mode: str
    views: List[str]
    layers: List[int]
    embed_model: str = DEFAULT_EMBED_MODEL
    exposure: str = "off"
    exposure_topk: int = 3
    sample_n: int = SAMPLE_N
    anchor_guided: str = "on"
    anchor_alpha: float = 3.0
    label: bool = False
    plot: bool = True
    openrouter_model: str = "openai/gpt-4o-mini"
    sleep_s: float = 1.5
    max_docs_in_prompt: int = 40
    user_agent: str = "patent-topics/1.0 (research use)"


def main_run(cfg: RunConfig) -> None:
    """Orchestrate a full BERTopic patent topics pipeline run."""
    from p2_taxonomy.llm import label_groups_openrouter, rebuild_taxonomy_from_labels
    from p2_taxonomy.visualisations import (
        viz_prevalence,
        viz_sunburst,
        viz_topic_cpc,
        viz_topic_map,
    )

    layers = sorted(int(x) for x in cfg.layers)
    leaf_layer = layers[-1]
    run_tag = f"single_layers_{'-'.join(map(str, layers))}"
    if cfg.anchor_guided == "on":
        run_tag += f"_anch_a{str(cfg.anchor_alpha).replace('.', 'p')}"
    dirs = _run_dirs(cfg.mode, run_tag)

    seeds = _seeds_for_mode(cfg.mode)

    ingest_corpus(cfg.mode, cache_root=CORPUS_ROOT)

    # Pre-compute base embeddings for all requested views
    compute_and_cache_embeddings(
        mode=cfg.mode, cache_root=CACHE_ROOT, corpus_root=CORPUS_ROOT,
        embed_model=cfg.embed_model, views=cfg.views,
    )

    for view in cfg.views:
        print(f"\n[topics] mode={cfg.mode} view={view} layers={layers}")

        emb_all, ids_all = compute_embeddings(
            dirs["parts"], cfg.mode, view, cfg.embed_model,
            anchor_guided=(cfg.anchor_guided == "on"),
            anchor_alpha=float(cfg.anchor_alpha),
            anchor_seeds=seeds,
        )

        base_dir, docs_sample, emb_sample = ensure_base_model(
            dirs["parts"], cfg.mode, view, cfg.embed_model, emb_all,
            sample_n=cfg.sample_n,
            anchor_guided=(cfg.anchor_guided == "on"),
            anchor_alpha=float(cfg.anchor_alpha),
            anchor_seeds=seeds,
        )

        taxonomy_df, leaf_lookup, mapping, labels_by_layer, models_by_layer = (
            build_taxonomy_tree_multi_layers(base_dir, docs_sample, layers, emb_sample=emb_sample)
        )

        ensure_dir(dirs["taxonomy"])
        tax_path = os.path.join(dirs["taxonomy"], f"taxonomy_table_{view}.csv")
        taxonomy_df.to_csv(tax_path, index=False)

        if cfg.label:
            from shared.llm import read_openrouter_key
            api_key = read_openrouter_key()
            labels_dir = os.path.join(dirs["taxonomy"], f"labels_llm_{view}")
            labels_by_layer = label_groups_openrouter(
                models_by_layer=models_by_layer,
                layers=layers,
                mapping=mapping,
                labels_by_layer_kw=labels_by_layer,
                out_dir=labels_dir,
                api_key=api_key,
                model=cfg.openrouter_model,
                user_agent=cfg.user_agent,
                sleep_s=cfg.sleep_s,
                max_docs_in_prompt=cfg.max_docs_in_prompt,
                domain="technology",
            )
            taxonomy_df, leaf_lookup = rebuild_taxonomy_from_labels(
                labels_by_layer, layers, mapping, models_by_layer,
            )
            tax_llm_path = os.path.join(dirs["taxonomy"], f"taxonomy_table_{view}_llm.csv")
            taxonomy_df.to_csv(tax_llm_path, index=False)

        model_leaf = _reduce_from_base(
            base_dir, docs_sample, leaf_layer,
            calc_probs=(cfg.exposure == "on"),
        )

        primary_out, _, stats, primary_df = assign_all(
            parts_dir=dirs["parts"], view=view,
            emb_all=emb_all, ids_all=ids_all,
            leaf_layer=leaf_layer, model_leaf=model_leaf,
            leaf_lookup=leaf_lookup, layers=layers,
            out_assign_dir=dirs["assign"],
            do_exposure=(cfg.exposure == "on"),
            exposure_topk=cfg.exposure_topk,
        )
        print(f"[out] primary: {primary_out}")

        # Visualisations
        if cfg.plot:
            ensure_dir(dirs["viz"])
            leaf_label_col = f"L{leaf_layer}_label"

            viz_sunburst(
                taxonomy_df,
                out_path=os.path.join(dirs["viz"], f"Sunburst_taxonomy_{view}.html"),
                title=f"[{cfg.mode}] Taxonomy Sunburst (layers={layers}) -- {view}",
            )
            viz_prevalence(
                primary_df,
                out_path=os.path.join(dirs["viz"], f"A_prevalence_leaf_{view}.png"),
                title=f"[{cfg.mode}] Leaf prevalence (L{leaf_layer}) -- {view}",
                level_col=leaf_label_col,
            )
            viz_topic_map(
                model_leaf,
                out_path=os.path.join(dirs["viz"], f"D_topic_map_{view}.html"),
                title=f"[{cfg.mode}] Topic map (leaf=L{leaf_layer}) -- {view}",
            )
            viz_topic_cpc(
                dirs["parts"], primary_df, leaf_layer,
                out_path=os.path.join(dirs["viz"], f"E_topic_cpc_{view}.png"),
                title=f"[{cfg.mode}] CPC correspondence (leaf=L{leaf_layer}) -- {view}",
            )

        gc.collect()

    print("[topics] DONE")
