"""
p2_taxonomy/llm.py
~~~~~~~~~~~~~~~~~~
LLM-based labeling for P2 taxonomy clusters.

Two labeling functions, sharing the same OpenRouter transport:

- ``label_groups_openrouter``   -- BERTopic patent topics (topics pipeline)
- ``label_clusters_from_df``    -- Leiden wiki clusters (in-memory df_assign,
                                   used by the cluster pipeline).

Also provides patching helpers to write labels back into taxonomy outputs:
- ``rebuild_taxonomy_from_labels`` -- for BERTopic output DataFrames
- ``patch_cluster_labels``         -- for Leiden grouped-pages CSVs
"""

from __future__ import annotations

import ast
import json as _json
import os
import sys
import re as _re
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from shared.llm import openrouter_chat, read_openrouter_key
from shared.utils import ensure_dir


# ------------------------------------------------------------------
# Unified LLM labeling
# ------------------------------------------------------------------

def label_groups_openrouter(
    models_by_layer: Dict[int, "BERTopic"],
    layers: List[int],
    mapping: Dict[int, Dict[int, int]],
    labels_by_layer_kw: Dict[int, Dict[int, str]],
    out_dir: str,
    api_key: str,
    model: str,
    user_agent: str,
    sleep_s: float,
    max_docs_in_prompt: int,
    domain: str = "technology",
) -> Dict[int, Dict[int, str]]:
    """LLM-label each topic at every layer, coarsest first, with parent context.

    Parameters
    ----------
    models_by_layer : dict
        ``{layer_k: BERTopic_model}`` for each layer.
    layers : list[int]
        Sorted list of layer sizes (e.g. ``[12, 50, 150]``).
    mapping : dict
        ``{child_k: {child_tid: parent_tid}}`` parent links.
    labels_by_layer_kw : dict
        Keyword-based labels to include as context in the prompt.
    out_dir : str
        Directory to write per-layer label CSVs.
    api_key : str
        OpenRouter API key.
    model : str
        OpenRouter model id (e.g. ``openai/gpt-4o-mini``).
    user_agent : str
        HTTP User-Agent string.
    sleep_s : float
        Sleep between API calls.
    max_docs_in_prompt : int
        Max representative docs per topic in the prompt.
    domain : str
        Controls the system prompt phrasing.  Supported values:
        ``"technology"`` (default), ``"patent"``.

    Returns
    -------
    dict
        ``{layer_k: {topic_id: label_str}}``.
    """
    ensure_dir(out_dir)
    labels: Dict[int, Dict[int, str]] = {}

    # Build a domain-sensitive preamble
    if domain == "patent":
        preamble = "We are building a technology taxonomy from patent titles and abstracts."
    else:
        preamble = "We are building a technology taxonomy."

    for layer_i, k in enumerate(layers):
        btopic = models_by_layer[k]
        topic_ids = sorted(t for t in btopic.get_topics().keys() if t != -1)
        rep_docs = btopic.get_representative_docs() or {}

        print(f"\n[LABEL] Layer L{k}  topics={len(topic_ids)}", flush=True)
        layer_map: Dict[int, str] = {}

        for tid in tqdm(topic_ids, desc=f"Labeling L{k}", unit="topic"):
            docs = rep_docs.get(tid, [])
            sample = docs[:max_docs_in_prompt]

            # Build parent-context lines from already-labelled coarser layers
            ctx_lines: List[str] = []
            if layer_i > 0:
                child_k = k
                cur = tid
                for j in range(layer_i - 1, -1, -1):
                    parent_k = layers[j]
                    parent_id = mapping.get(child_k, {}).get(cur)
                    if parent_id is not None:
                        plab = labels.get(parent_k, {}).get(
                            parent_id, f"L{parent_k} {parent_id}",
                        )
                        ctx_lines.append(f"- Parent L{parent_k}: {plab}")
                    child_k = parent_k
                    cur = parent_id if parent_id is not None else cur

            kw_label = labels_by_layer_kw.get(k, {}).get(tid, f"Topic {tid}")
            prompt = (
                f"{preamble}\n"
                "Give a short, specific technology-subfield label.\n"
                + ("\n".join(ctx_lines) + "\n\n" if ctx_lines else "")
                + f"Cluster layer: L{k}\n"
                f"Cluster size (representative docs): {len(docs)}\n"
                f"Top keywords: {kw_label}\n"
                "Representative documents:\n- "
                + "\n- ".join(sample if sample else [kw_label])
                + "\n\nReturn only the label."
            )

            try:
                lab = openrouter_chat(api_key, model, user_agent, prompt)
            except Exception as exc:
                print(f"[WARN] labeling failed for L{k} tid={tid}: {exc}", flush=True)
                lab = kw_label

            layer_map[tid] = lab
            if sleep_s and sleep_s > 0:
                time.sleep(float(sleep_s))

        labels[k] = layer_map

        out_path = os.path.join(out_dir, f"labels__L{k}__llm.csv")
        pd.DataFrame(
            [{"layer": k, "topic_id": tid, "label": lab} for tid, lab in layer_map.items()]
        ).to_csv(out_path, index=False)
        print(f"[WRITE] {out_path}", flush=True)

    return labels


# ------------------------------------------------------------------
# Rebuild taxonomy from LLM labels
# ------------------------------------------------------------------

def rebuild_taxonomy_from_labels(
    labels_by_layer: Dict[int, Dict[int, str]],
    layers: List[int],
    mapping: Dict[int, Dict[int, int]],
    models_by_layer: Dict[int, "BERTopic"],
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, str]]]:
    """Rebuild ``taxonomy_df`` and ``leaf_lookup`` with (possibly LLM-updated) labels.

    Returns ``(taxonomy_df, leaf_lookup)``.
    """
    leaf_k = layers[-1]
    leaf_ids = sorted(t for t in models_by_layer[leaf_k].get_topics().keys() if t != -1)

    leaf_lookup: Dict[int, Dict[str, str]] = {}
    rows: list[dict[str, str]] = []

    for leaf_id in leaf_ids:
        cur = int(leaf_id)
        row: Dict[str, str] = {
            f"L{leaf_k}": labels_by_layer[leaf_k].get(cur, f"L{leaf_k} {cur}"),
        }

        for j in range(len(layers) - 2, -1, -1):
            parent_k = layers[j]
            child_k = layers[j + 1]
            parent_id = mapping[child_k].get(cur)
            if parent_id is None:
                row = {}
                break
            row[f"L{parent_k}"] = labels_by_layer[parent_k].get(
                parent_id, f"L{parent_k} {parent_id}",
            )
            cur = parent_id

        if row and all(f"L{k}" in row for k in layers):
            rows.append(row)
            leaf_lookup[int(leaf_id)] = row

    taxonomy_df = pd.DataFrame(rows).drop_duplicates()
    taxonomy_df = taxonomy_df[[f"L{k}" for k in layers]]
    return taxonomy_df, leaf_lookup
def label_clusters_from_df(
    df_assign: pd.DataFrame,
    level_cols: List[str],
    out_dir: Union[Path, str],
    api_key: str,
    model: str,
    user_agent: str,
    sleep_s: float = 1.5,
    max_pages_in_prompt: int = 40,
    domain: str = "technology",
) -> Dict[str, Dict[str, str]]:
    """LLM-label each Leiden cluster using an in-memory assignment DataFrame.

    Parameters
    ----------
    df_assign : pd.DataFrame
        Assignment DataFrame with one row per page and columns
        ``[level_cols..., page]`` (as produced by the clustering pipeline).
    level_cols : list[str]
        Hierarchy levels from coarsest to finest, e.g. ``["root", "L1", "L2"]``.
    out_dir : Path or str
        Directory to write per-level label CSVs
        (``labels__{lvl}__adaptive.csv``).
    api_key : str
        OpenRouter API key.
    model : str
        OpenRouter model id (e.g. ``openai/gpt-4o-mini``).
    user_agent : str
        HTTP User-Agent string.
    sleep_s : float
        Sleep between API calls.
    max_pages_in_prompt : int
        Max page titles shown to the LLM per cluster.
    domain : str
        Controls the system prompt phrasing.

    Returns
    -------
    dict
        ``{level: {cluster_id: label_str}}``.
    """
    ensure_dir(out_dir)
    out_dir = Path(out_dir)

    if domain == "patent":
        preamble = "We are building a technology taxonomy from patent titles and abstracts."
    else:
        preamble = "We are building a technology taxonomy from Wikipedia pages."

    labels: Dict[str, Dict[str, str]] = {}

    for lvl_i, lvl in enumerate(level_cols):
        parents = level_cols[:lvl_i]
        grp_cols = parents + [lvl]

        grouped = df_assign.groupby(grp_cols)["page"].apply(list).reset_index()
        grouped["n"] = grouped["page"].apply(len)
        grouped = grouped.sort_values("n", ascending=False)

        print(f"\n[LABEL] {lvl}  clusters={len(grouped)}", flush=True)
        mapping: Dict[str, str] = {}

        for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc=f"Labeling {lvl}", unit="cluster"):
            cid = str(row[lvl])
            pages = row["page"]
            n = int(row["n"])

            ctx_lines: List[str] = []
            for p in parents:
                pid = str(row[p])
                plab = labels.get(p, {}).get(pid, pid)
                ctx_lines.append(f"- Parent {p}: {plab}")

            sample = pages[:max_pages_in_prompt]
            prompt = (
                f"{preamble}\n"
                "Give a short, specific technology-subfield label.\n"
                + ("\n".join(ctx_lines) + "\n\n" if ctx_lines else "")
                + f"Cluster level: {lvl}\n"
                f"Cluster size: {n}\n"
                "Pages:\n- " + "\n- ".join(sample)
                + "\n\nReturn only the label."
            )

            try:
                lab = openrouter_chat(api_key, model, user_agent, prompt)
            except Exception as exc:
                print(f"[WARN] labeling failed for {lvl} cid={cid}: {exc}", flush=True)
                lab = cid

            mapping[cid] = lab
            if sleep_s and sleep_s > 0:
                time.sleep(float(sleep_s))

        labels[lvl] = mapping

        out_path = out_dir / f"labels__{lvl}__adaptive.csv"
        pd.DataFrame(
            [{"level": lvl, "cluster_id": k, "label": v} for k, v in mapping.items()]
        ).to_csv(out_path, index=False)
        print(f"[WRITE] {out_path}", flush=True)

    return labels


def patch_cluster_labels(
    grouped_csv: Union[Path, str],
    labels_by_level: Dict[str, Dict[str, str]],
    level_cols: List[str],
    output_csv: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """Patch LLM labels into a grouped-pages CSV, replacing cluster IDs with label strings.

    Parameters
    ----------
    grouped_csv : Path or str
        ``taxonomy_grouped_pages__L*.csv`` (the raw, unlabeled version).
    labels_by_level : dict
        ``{level: {cluster_id: label_str}}`` as returned by
        ``label_clusters_from_df``.
    level_cols : list[str]
        Hierarchy levels, e.g. ``["root", "L1", "L2"]``.
    output_csv : Path or str or None
        If given, write the labeled DataFrame to this path.

    Returns
    -------
    pd.DataFrame
        Copy of the grouped CSV with cluster IDs replaced by LLM labels.
    """
    df = pd.read_csv(grouped_csv)
    df_out = df.copy()

    for lvl in level_cols:
        if lvl in labels_by_level and lvl in df_out.columns:
            mp = labels_by_level[lvl]
            df_out[lvl] = df_out[lvl].astype(str).map(mp).fillna(df_out[lvl].astype(str))

    if output_csv is not None:
        df_out.to_csv(output_csv, index=False)
        print(f"[WRITE] {output_csv}", flush=True)

    return df_out


# ------------------------------------------------------------------
# Context-aware taxonomy labeling (v2) — dual backend
# ------------------------------------------------------------------

ChatFn = Callable[[str, str, int, Optional[dict]], str]
"""Signature: (prompt, system_prompt, max_tokens, json_schema=None) -> response_text"""


# ------------------------------------------------------------------
# JSON schemas for structured LLM output
# ------------------------------------------------------------------



def _validate_json_response(
    text: str, required_keys: List[str],
) -> Optional[dict]:
    """Parse *text* as JSON and check it contains *required_keys*.

    Returns the parsed dict on success, ``None`` on failure.
    """
    cleaned = _re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`')
    try:
        obj = _json.loads(cleaned)
    except (ValueError, _json.JSONDecodeError):
        # Try extracting first JSON object from text
        m = _re.search(r'\{[\s\S]*\}', cleaned)
        if not m:
            return None
        try:
            obj = _json.loads(m.group())
        except (ValueError, _json.JSONDecodeError):
            return None
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in required_keys):
        return None
    return obj


def _make_chat_fn(
    backend: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    user_agent: Optional[str] = None,
    claude_model: str = "opus",
) -> ChatFn:
    """Return a unified chat callable for the chosen backend.

    The returned callable has signature::

        (prompt, system_prompt, max_tokens, json_schema=None) -> str

    When *json_schema* is provided:

    - **openrouter**: passes the schema to the API for strict enforcement.
    - **claude-code**: appends a JSON instruction to the prompt, parses
      the response, and retries (up to 2×) if the response doesn't
      contain the required keys.

    The *json_schema* dict follows the OpenAI-style format::

        {"name": "...", "strict": True,
         "schema": {"type": "object", "properties": {...}, "required": [...]}}

    Parameters
    ----------
    backend : ``"claude-code"`` or ``"openrouter"``
    claude_model : model alias for ``claude --model`` (default ``"opus"``).
    """
    MAX_RETRIES = 2

    if backend == "claude-code":
        def _chat(
            prompt: str, system_prompt: str, max_tokens: int,
            json_schema: Optional[dict] = None,
        ) -> str:
            effective_prompt = prompt
            if json_schema is not None:
                schema_body = json_schema.get("schema", {})
                required = schema_body.get("required", [])
                effective_prompt += (
                    "\n\nIMPORTANT: Respond with ONLY a valid JSON object "
                    f"containing exactly these keys: {required}. "
                    "No markdown, no explanation, no code fences."
                )

            for attempt in range(1, MAX_RETRIES + 2):
                cmd = [
                    "claude", "--print",
                    "--model", claude_model,
                    "--output-format", "text",
                    "--system-prompt", system_prompt,
                    "--no-session-persistence",
                    effective_prompt,
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300,
                    stdin=subprocess.DEVNULL,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"claude exited {result.returncode}: "
                        f"{result.stderr[:300]}"
                    )
                text = result.stdout.strip()

                if json_schema is None:
                    return text

                # Validate response has required keys
                schema_body = json_schema.get("schema", {})
                required = schema_body.get("required", [])
                parsed = _validate_json_response(text, required)
                if parsed is not None:
                    return _json.dumps(parsed, ensure_ascii=False)

                if attempt <= MAX_RETRIES:
                    effective_prompt = (
                        f"Your previous response was not valid JSON with "
                        f"required keys {required}. Try again.\n\n"
                        + effective_prompt
                    )

            # Return raw text after all retries exhausted
            return text
        return _chat

    elif backend == "openrouter":
        if not api_key:
            api_key = read_openrouter_key()
        if not model:
            model = "google/gemini-3-flash-preview"
        if not user_agent:
            user_agent = "st4te-taxonomy"

        def _chat(
            prompt: str, system_prompt: str, max_tokens: int,
            json_schema: Optional[dict] = None,
        ) -> str:
            return openrouter_chat(
                api_key, model, user_agent, prompt,
                max_tokens=max_tokens, system_prompt=system_prompt,
                json_schema=json_schema,
            )
        return _chat

    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'claude-code' or 'openrouter'.")


def _centroid_sample(
    df: pd.DataFrame,
    cluster_col: str,
    cluster_id: str,
    ranked_pages: Dict[tuple, List[str]],
    max_pages: int = 50,
) -> List[str]:
    """Return pages closest to the cluster centroid.

    Uses pre-computed page-distance rankings from ``ranked_pages``
    (keyed by ``(level, cluster_id)``).  Falls back to alphabetical
    order if no ranking is available.
    """
    ranked = ranked_pages.get((cluster_col, str(cluster_id)))
    if ranked:
        return ranked[:max_pages]

    # Fallback: alphabetical (deterministic but not distance-ranked)
    subset = df[df[cluster_col] == cluster_id]
    if subset.empty:
        return []
    return sorted(subset["page"].unique())[:max_pages]




def _parse_cluster_info(
    text: str,
    cluster_ids: List[str],
) -> Dict[str, dict]:
    """Parse LLM response into ``{cluster_id: {technologies, description, label}}``.

    Handles numbered JSON lines, a JSON array, or a single JSON object.
    """
    # Strip markdown code fences
    cleaned = _re.sub(r'```(?:json)?\s*', '', text).strip()

    results: List[dict] = []

    # Try 1: JSON array
    try:
        parsed = _json.loads(cleaned)
        if isinstance(parsed, list):
            results = [x for x in parsed if isinstance(x, dict)]
    except (ValueError, _json.JSONDecodeError):
        pass

    # Try 2: single JSON object
    if not results:
        try:
            parsed = _json.loads(cleaned)
            if isinstance(parsed, dict) and "label" in parsed:
                results = [parsed]
        except (ValueError, _json.JSONDecodeError):
            pass

    # Try 3: extract JSON objects from text (handles numbered lines)
    if not results:
        for m in _re.finditer(r'\{[^{}]*(?:\[[^\[\]]*\][^{}]*)?\}', cleaned):
            try:
                obj = _json.loads(m.group())
                if "label" in obj:
                    results.append(obj)
            except (ValueError, _json.JSONDecodeError):
                continue

    # Build mapping
    mapping: Dict[str, dict] = {}
    for i, cid in enumerate(cluster_ids):
        if i < len(results):
            info = results[i]
            techs = info.get("technologies", [])
            if isinstance(techs, str):
                techs = [t.strip() for t in techs.split(",")]
            mapping[cid] = {
                "technologies": techs,
                "description": str(info.get("description", "")),
                "label": str(info.get("label", cid)).strip().strip('"').strip("'"),
            }
        else:
            mapping[cid] = {"technologies": [], "description": "", "label": cid}

    return mapping


def _build_cluster_content(
    df_assign: pd.DataFrame,
    lvl: str,
    cid: str,
    child_col: Optional[str],
    cluster_info: Dict[str, Dict[str, dict]],
    max_pages: int,
    ranked_pages: Optional[Dict[tuple, List[str]]] = None,
    page_descs: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """Build content block for a cluster prompt.

    For leaf levels (L2), returns page titles + descriptions ranked by
    centroid proximity.
    For internal levels (L1, root), returns combined unique technologies
    plus child-level summaries (labels + technologies + descriptions).

    Returns ``(content_block, content_type)`` where *content_type* is
    ``"pages"`` or ``"children"``.
    """
    if child_col and child_col in cluster_info:
        # Internal node: combined technologies + child summaries
        child_ids = sorted(
            df_assign[df_assign[lvl].astype(str) == str(cid)][child_col]
            .astype(str).unique()
        )
        all_techs: List[str] = []
        lines: List[str] = []
        for child_id in child_ids:
            if child_id in cluster_info[child_col]:
                info = cluster_info[child_col][child_id]
                techs = ", ".join(info["technologies"]) if info["technologies"] else "N/A"
                all_techs.extend(info["technologies"])
                lines.append(
                    f'- "{info["label"]}" -- technologies: [{techs}]\n'
                    f'  {info["description"]}'
                )
        # Deduplicate while preserving order
        seen: set = set()
        unique_techs: List[str] = []
        for t in all_techs:
            if t not in seen:
                seen.add(t)
                unique_techs.append(t)
        combined = ", ".join(unique_techs) if unique_techs else "N/A"
        header = (
            f"Combined technologies: [{combined}]\n"
            f"Sub-clusters ({child_col}, {len(lines)} total)"
        )
        return f"{header}:\n" + "\n".join(lines), "children"
    else:
        # Leaf node: page titles + descriptions ranked by centroid distance
        pages = _centroid_sample(
            df_assign, lvl, cid, ranked_pages or {}, max_pages,
        )
        _descs = page_descs or {}
        page_lines: List[str] = []
        for p in pages:
            desc = _descs.get(p, "")
            if desc:
                page_lines.append(f"- {p}: {desc[:200]}")
            else:
                page_lines.append(f"- {p}")
        return (
            f"Wikipedia technology pages ({len(pages)} shown, "
            f"ranked by representativeness):\n" + "\n".join(page_lines)
        ), "pages"


def label_taxonomy_contextual(
    df_assign: pd.DataFrame,
    level_cols: List[str],
    out_dir: Union[Path, str],
    backend: str = "claude-code",
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    user_agent: Optional[str] = None,
    claude_model: str = "opus",
    sleep_s: float = 1.0,
    max_pages_in_prompt: int = 50,
    review_pass: bool = True,
    page_distances: Optional[pd.DataFrame] = None,
    page_descriptions: Optional[Dict[str, str]] = None,
    log_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Dict[str, str]]:
    """Bottom-up LLM labeling that produces a coherent taxonomy.

    Processes levels **bottom-up** (L2 → L1 → root):

    - **L2** (leaf): labeled from page titles ranked by centroid proximity.
    - **L1**: labeled from L2 children's technologies, descriptions,
      and labels.
    - **root**: labeled from L1 children's summaries (batch call).

    Per-cluster output: *technologies* (1-3), *description* (2 sentences),
    *label* (1-4 words).

    Supports two backends:

    - ``backend="claude-code"`` — calls the ``claude`` CLI.
    - ``backend="openrouter"`` — calls OpenRouter.

    Parameters
    ----------
    df_assign : pd.DataFrame
        Taxonomy DataFrame with columns ``[*level_cols, page]``.
    level_cols : list[str]
        Hierarchy levels coarsest→finest, e.g. ``["root", "L1", "L2"]``.
    out_dir : Path or str
        Output directory for label CSVs.
    backend : str
        ``"claude-code"`` or ``"openrouter"``.
    api_key, model, user_agent : str or None
        OpenRouter credentials (ignored for claude-code backend).
    claude_model : str
        Model alias for ``claude --model`` (default ``"opus"``).
    sleep_s : float
        Pause between API calls.
    max_pages_in_prompt : int
        Max page titles per cluster in prompt (L2 leaf level).
    review_pass : bool
        If True, run a second pass to resolve overlapping labels.
    page_distances : DataFrame, optional
        From ``page_distances_*.csv`` with columns like
        ``page, L2, L2_dist_to_own, ...``.  Used to rank pages by
        proximity to centroid for L2 leaf prompts.

    Returns
    -------
    dict
        ``{level: {cluster_path: label_str}}``.
    """
    ensure_dir(out_dir)
    out_dir = Path(out_dir)

    # --- Logging to file (defaults to out_dir/labeling.log) ---
    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self._streams:
                s.flush()

    _effective_log = Path(log_path) if log_path is not None else out_dir / "labeling.log"
    _log_fh = open(_effective_log, "a", encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _log_fh)
    print(f"[LOG] Logging to {_effective_log}", flush=True)

    # Pre-build page-distance lookup: {(level, cluster_id)} -> sorted page list
    _ranked_pages: Dict[tuple, List[str]] = {}
    if page_distances is not None:
        for lvl in level_cols:
            dist_col = f"{lvl}_dist_to_own"
            if dist_col not in page_distances.columns or lvl not in page_distances.columns:
                continue
            df_lvl = page_distances.dropna(subset=[dist_col]).copy()
            df_lvl[dist_col] = pd.to_numeric(df_lvl[dist_col], errors="coerce")
            for cid, grp in df_lvl.groupby(lvl):
                ranked = grp.sort_values(dist_col)["page"].tolist()
                _ranked_pages[(lvl, str(cid))] = ranked
        print(f"  [page-distances] built rankings for {len(_ranked_pages)} clusters",
              flush=True)

    _page_descs: Dict[str, str] = page_descriptions or {}
    if _page_descs:
        print(f"  [page-descs] {len(_page_descs)} page descriptions available",
              flush=True)

    chat = _make_chat_fn(backend, api_key, model, user_agent, claude_model)

    SYSTEM = (
        "You are an expert in digital technology taxonomies built from Wikipedia. "
        "You analyze clusters of Wikipedia technology pages and produce structured "
        "outputs: key technologies, concise descriptions, and specific labels. "
        "Labels should be 1-5 words, specific, and non-overlapping. "
        "Technologies should identify the core domain(s) — 1 preferred, up to 3. "
        "Descriptions should be exactly 2 sentences. "
        "Return valid JSON as instructed."
    )

    suffix = "claude" if backend == "claude-code" else "openrouter"
    cluster_info: Dict[str, Dict[str, dict]] = {}

    # Process bottom-up: L2 → L1 → root
    for lvl in reversed(level_cols):
        lvl_i = level_cols.index(lvl)
        parents = level_cols[:lvl_i]
        child_col = level_cols[lvl_i + 1] if lvl_i + 1 < len(level_cols) else None

        grp_cols = parents + [lvl]
        grouped = df_assign.groupby(grp_cols)["page"].apply(list).reset_index()
        grouped["n"] = grouped["page"].apply(len)
        grouped = grouped.sort_values("n", ascending=False)

        print(f"\n[LABEL] {lvl}  clusters={len(grouped)}  (bottom-up)", flush=True)

        ckpt_path = out_dir / f".checkpoint__{lvl}__{suffix}.csv"

        if lvl_i == 0:
            # Root — batch all roots, using L1 child summaries
            cluster_info[lvl] = _label_roots_batch(
                df_assign, lvl, child_col, grouped, cluster_info,
                chat, SYSTEM, max_pages_in_prompt, sleep_s,
            )
        else:
            # L1 or L2 — label with sibling context
            cluster_info[lvl] = _label_level_with_siblings(
                df_assign, lvl, parents, child_col, grouped,
                cluster_info, chat, SYSTEM, max_pages_in_prompt, sleep_s,
                checkpoint_path=ckpt_path,
                ranked_pages=_ranked_pages, page_descs=_page_descs,
            )

        review_results: Dict[str, dict] = {}
        if review_pass:
            label_map = {cid: info["label"] for cid, info in cluster_info[lvl].items()}
            tech_map = {cid: info["technologies"] for cid, info in cluster_info[lvl].items()}
            review_results = _review_labels(label_map, lvl, chat, SYSTEM, sleep_s, tech_map)
            for cid, rev in review_results.items():
                if cid in cluster_info[lvl]:
                    cluster_info[lvl][cid]["label"] = rev["reviewed_label"]

        # Write final CSV with all fields and remove checkpoint
        out_path = out_dir / f"labels__{lvl}__{suffix}.csv"
        rows = []
        for k, v in cluster_info[lvl].items():
            rev = review_results.get(k, {})
            rows.append({
                "level": lvl,
                "cluster_id": k,
                "label": v["label"],
                "technologies": "|".join(v["technologies"]),
                "description": v["description"],
                "reviewed_label": rev.get("reviewed_label", ""),
                "review_reason": rev.get("reason", ""),
            })
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[WRITE] {out_path}", flush=True)
        if ckpt_path.exists():
            ckpt_path.unlink()

    # Restore stdout
    if _orig_stdout is not None:
        sys.stdout = _orig_stdout
    if _log_fh is not None:
        _log_fh.close()

    # Return {level: {cluster_id: label}} for backward compat
    return {
        lvl: {cid: info["label"] for cid, info in cluster_info[lvl].items()}
        for lvl in level_cols
    }


# ------------------------------------------------------------------
# Internal labeling helpers
# ------------------------------------------------------------------

def _label_roots_batch(
    df_assign: pd.DataFrame,
    lvl: str,
    child_col: Optional[str],
    grouped: pd.DataFrame,
    cluster_info: Dict[str, Dict[str, dict]],
    chat: ChatFn,
    system: str,
    max_pages: int,
    sleep_s: float,
) -> Dict[str, dict]:
    """Label all root clusters in a single LLM call for coherence.

    Uses L1 child summaries (technologies/descriptions/labels) when
    available; falls back to page titles for leaf-level roots.
    """
    cluster_blocks: List[str] = []
    cluster_ids: List[str] = []

    for _, row in grouped.iterrows():
        cid = str(row[lvl])
        if "Peripheral" in cid:
            continue
        cluster_ids.append(cid)
        content, _ = _build_cluster_content(
            df_assign, lvl, cid, child_col, cluster_info, max_pages,
        )
        block = f"Cluster {len(cluster_ids)} ({row['n']} pages):\n{content}"
        cluster_blocks.append(block)

    prompt = (
        "We are building a hierarchical digital technology taxonomy from Wikipedia.\n\n"
        'A "technology" is the core technology concept that a cluster of '
        "Wikipedia pages represents.\n\n"
        "Below are ALL root-level clusters described by their sub-cluster summaries.\n"
        "For each root cluster, provide:\n"
        "- technologies: the overarching technology/technologies represented "
        "(1 preferred, up to 3)\n"
        "- description: two-sentence description of the cluster\n"
        "- label: a specific 1-5 word label representing the cluster of "
        "technologies, pages and clusters\n\n"
        "The full set of labels must form a coherent, non-overlapping partition "
        "of digital technology.\n\n"
        + "\n\n".join(cluster_blocks)
        + "\n\nReturn one JSON object per cluster, numbered 1-N:\n"
        '1. {"technologies": ["tech"], "description": "...", "label": "..."}\n'
        '2. {"technologies": ["tech"], "description": "...", "label": "..."}'
    )

    print(f"  Labeling {len(cluster_ids)} root clusters (single batch call)...",
          flush=True)
    try:
        resp = chat(prompt, system, 2048)
    except Exception as exc:
        print(f"[WARN] batch root labeling failed: {exc}", flush=True)
        return {cid: {"technologies": [], "description": "", "label": cid}
                for cid in cluster_ids}

    mapping = _parse_cluster_info(resp, cluster_ids)

    # Always label Peripheral
    for _, row in grouped.iterrows():
        cid = str(row[lvl])
        if "Peripheral" in cid and cid not in mapping:
            mapping[cid] = {
                "technologies": [], "description": "", "label": "Peripheral",
            }

    if sleep_s:
        time.sleep(float(sleep_s))

    labels_preview = [v["label"] for v in mapping.values()]
    print(f"  Root labels: {labels_preview}", flush=True)
    return mapping


def _label_level_with_siblings(
    df_assign: pd.DataFrame,
    lvl: str,
    parents: List[str],
    child_col: Optional[str],
    grouped: pd.DataFrame,
    cluster_info: Dict[str, Dict[str, dict]],
    chat: ChatFn,
    system: str,
    max_pages: int,
    sleep_s: float,
    checkpoint_path: Optional[Path] = None,
    ranked_pages: Optional[Dict[tuple, List[str]]] = None,
    page_descs: Optional[Dict[str, str]] = None,
) -> Dict[str, dict]:
    """Label L1/L2 clusters with sibling context (bottom-up).

    For L2 (leaf): content comes from page titles.
    For L1: content comes from L2 children's summaries in *cluster_info*.
    """
    # Load checkpoint if available
    mapping: Dict[str, dict] = {}
    if checkpoint_path and checkpoint_path.exists():
        ckpt = pd.read_csv(checkpoint_path, dtype=str).fillna("")
        for _, r in ckpt.iterrows():
            techs = [t for t in str(r.get("technologies", "")).split("|") if t]
            mapping[r["cluster_id"]] = {
                "technologies": techs,
                "description": str(r.get("description", "")),
                "label": str(r["label"]),
            }
        print(f"  [CHECKPOINT] resumed {len(mapping)} labels from {checkpoint_path.name}",
              flush=True)

    # Group by parent path to label siblings together
    parent_groups: Dict[tuple, list] = {}
    for _, row in grouped.iterrows():
        parent_key = tuple(str(row[p]) for p in parents)
        parent_groups.setdefault(parent_key, []).append(row)

    total_clusters = sum(
        len([r for r in rows if "Peripheral" not in str(r[lvl])])
        for rows in parent_groups.values()
    )
    pbar = tqdm(total=total_clusters, desc=f"Labeling {lvl}", unit="cluster")
    already_done = sum(1 for cid in mapping if "Peripheral" not in cid)
    pbar.update(already_done)

    for parent_key, rows in parent_groups.items():
        parent_ctx_parts = []
        for p, pid in zip(parents, parent_key):
            # Use label from cluster_info if already processed (bottom-up)
            plab = cluster_info.get(p, {}).get(pid, {}).get("label", pid)
            parent_ctx_parts.append(f"{p}: {plab}")
        parent_ctx = " > ".join(parent_ctx_parts)

        non_peripheral = [r for r in rows if "Peripheral" not in str(r[lvl])]

        # Skip parent group if all clusters already checkpointed
        unchecked = [r for r in non_peripheral if str(r[lvl]) not in mapping]
        if not unchecked and non_peripheral:
            for row in rows:
                cid = str(row[lvl])
                if "Peripheral" in cid and cid not in mapping:
                    mapping[cid] = {
                        "technologies": [], "description": "", "label": "Peripheral",
                    }
            continue

        batch_info = _batch_label_siblings(
            df_assign, lvl, child_col, non_peripheral,
            parent_ctx, cluster_info, chat, system, max_pages, sleep_s,
            ranked_pages=ranked_pages, page_descs=page_descs,
        )
        mapping.update(batch_info)
        pbar.update(len(non_peripheral))

        # Checkpoint after each parent group
        if checkpoint_path:
            _write_label_checkpoint(checkpoint_path, lvl, mapping)

        for row in rows:
            cid = str(row[lvl])
            if "Peripheral" in cid and cid not in mapping:
                mapping[cid] = {
                    "technologies": [], "description": "", "label": "Peripheral",
                }

    pbar.close()
    return mapping


def _write_label_checkpoint(path: Path, lvl: str, mapping: Dict[str, dict]) -> None:
    """Write current cluster info mapping to a checkpoint CSV."""
    rows = []
    for k, v in mapping.items():
        rows.append({
            "level": lvl,
            "cluster_id": k,
            "label": v["label"],
            "technologies": "|".join(v["technologies"]),
            "description": v["description"],
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  [CHECKPOINT] saved {len(mapping)} labels -> {path.name}", flush=True)


def _batch_label_siblings(
    df_assign: pd.DataFrame,
    lvl: str,
    child_col: Optional[str],
    rows: list,
    parent_ctx: str,
    cluster_info: Dict[str, Dict[str, dict]],
    chat: ChatFn,
    system: str,
    max_pages: int,
    sleep_s: float,
    ranked_pages: Optional[Dict[tuple, List[str]]] = None,
    page_descs: Optional[Dict[str, str]] = None,
) -> Dict[str, dict]:
    """Batch-label all sibling clusters in one LLM call.

    Returns ``{cluster_id: {technologies, description, label}}``.
    """
    cluster_blocks: List[str] = []
    cluster_ids: List[str] = []

    for row in rows:
        cid = str(row[lvl])
        cluster_ids.append(cid)
        content, _ = _build_cluster_content(
            df_assign, lvl, cid, child_col, cluster_info, max_pages,
            ranked_pages=ranked_pages, page_descs=page_descs,
        )
        block = f"Cluster {len(cluster_ids)} ({row['n']} pages):\n{content}"
        cluster_blocks.append(block)

    prompt = (
        "We are building a hierarchical digital technology taxonomy from Wikipedia.\n\n"
        'A "page" is a Wikipedia article about a specific technology.\n'
        'A "technology" is the core technology concept that a cluster of pages represents.\n\n'
        f"Parent path: {parent_ctx}\n\n"
        f"Below are all {lvl}-level clusters under this parent.\n"
        "For each cluster, provide:\n"
        "- technologies: the core technology/technologies represented "
        "(1 preferred, up to 3)\n"
        "- description: two-sentence description of the cluster\n"
        "- label: a specific 1-5 word label representing the cluster of "
        "technologies, pages and clusters\n\n"
        "Labels must form a coherent, non-overlapping partition.\n\n"
        + "\n\n".join(cluster_blocks)
        + "\n\nReturn one JSON object per cluster, numbered 1-N:\n"
        '1. {"technologies": ["tech"], "description": "...", "label": "..."}\n'
        '2. {"technologies": ["tech"], "description": "...", "label": "..."}'
    )

    try:
        resp = chat(prompt, system, 2048)
    except Exception as exc:
        print(f"[WARN] batch sibling labeling failed: {exc}", flush=True)
        return {cid: {"technologies": [], "description": "", "label": cid}
                for cid in cluster_ids}

    mapping = _parse_cluster_info(resp, cluster_ids)

    if sleep_s:
        time.sleep(float(sleep_s))

    return mapping


def _review_labels(
    mapping: Dict[str, str],
    lvl: str,
    chat: ChatFn,
    system: str,
    sleep_s: float,
    tech_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, dict]:
    """Review labels at a level, batched by parent group (siblings).

    For root level all clusters are siblings (one batch).  For L1/L2 the
    parent is extracted from the cluster-path string so that each review
    prompt contains only the siblings under one parent.

    Returns ``{cluster_id: {"reviewed_label": str, "reason": str}}``.
    Unchanged labels have ``reason = "no change"``.
    """
    _techs = tech_map or {}

    non_peripheral = {k: v for k, v in mapping.items() if v != "Peripheral"}
    if len(non_peripheral) < 2:
        return {cid: {"reviewed_label": lbl, "reason": "no change"}
                for cid, lbl in mapping.items()}

    # Group cluster IDs by parent path
    parent_groups: Dict[str, List[str]] = {}
    for cid in non_peripheral:
        parts = cid.rsplit("|", 1)
        parent = parts[0] if len(parts) > 1 else "__ROOT__"
        parent_groups.setdefault(parent, []).append(cid)

    result: Dict[str, dict] = {
        cid: {"reviewed_label": lbl, "reason": "no change"}
        for cid, lbl in mapping.items()
    }
    total_changes = 0

    for parent, cids in parent_groups.items():
        if len(cids) < 2:
            continue

        label_lines: List[str] = []
        for i, cid in enumerate(cids):
            techs = _techs.get(cid, [])
            tech_str = f" -- technologies: [{', '.join(techs)}]" if techs else ""
            label_lines.append(f"{i+1}. {non_peripheral[cid]}{tech_str}")
        label_list = "\n".join(label_lines)

        prompt = (
            f"Below are all {lvl}-level sibling labels under one parent "
            f"in a digital technology taxonomy built from Wikipedia.\n"
            "Review them for:\n"
            "- Overlapping or redundant labels (merge/rename)\n"
            "- Labels that are too vague or generic\n"
            "- Inconsistent granularity (some too broad, others too narrow)\n\n"
            f"{label_list}\n\n"
            "Return one JSON per label, numbered 1-N:\n"
            '1. {"reviewed_label": "...", "reason": "..."}\n'
            '2. {"reviewed_label": "...", "reason": "..."}\n\n'
            'Use "no change" as reason if the label is fine as-is.'
        )

        try:
            resp = chat(prompt, system, 1024)
        except Exception as exc:
            print(f"[WARN] review pass failed for {lvl} (parent={parent}): {exc}",
                  flush=True)
            continue

        # Parse numbered JSON objects
        reviews: List[dict] = []
        cleaned = _re.sub(r'```(?:json)?\s*', '', resp).strip().rstrip('`')
        for m in _re.finditer(r'\{[^{}]*\}', cleaned):
            try:
                obj = _json.loads(m.group())
                if "reviewed_label" in obj:
                    reviews.append(obj)
            except (ValueError, _json.JSONDecodeError):
                continue

        for i, cid in enumerate(cids):
            if i < len(reviews):
                rev = reviews[i]
                new_label = str(rev.get("reviewed_label", "")).strip()
                reason = str(rev.get("reason", "no change")).strip()
                old_label = non_peripheral[cid]
                if new_label and new_label != old_label:
                    print(f"  [REVIEW] {lvl}: '{old_label}' -> '{new_label}' ({reason})",
                          flush=True)
                    result[cid] = {"reviewed_label": new_label, "reason": reason}
                    total_changes += 1
                else:
                    result[cid] = {"reviewed_label": old_label, "reason": reason}

        if sleep_s:
            time.sleep(float(sleep_s))

    print(f"  [REVIEW] {lvl}: {total_changes} labels revised out of {len(non_peripheral)}", flush=True)
    return result
