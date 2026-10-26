"""Phrase-native taxonomy labelling and consistency validation.

Operates on a phrase-cluster DataFrame (no page bridge). Bottom-up: L2 → L1 → root.

Input DataFrame must have columns: ``phrase, n_pages, root, L1, L2``
(rename your phrase-clustering layers to root/L1/L2 before calling).

Two entry points:
- :func:`label_phrase_taxonomy` — assigns ``label / description / functionalities`` per cluster.
- :func:`validate_phrase_consistency` — runs 3 LLM checks:
    0. Functionality verification (do the LLM-extracted functionalities match the cluster's phrases?)
    1. Parent–child subsumption
    2. Sibling exclusivity
"""
from __future__ import annotations
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from p2_taxonomy.llm import _make_chat_fn, _validate_json_response, read_openrouter_key


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYS_LABEL_L2 = (
    "You are creating a FUNCTIONALITY taxonomy. A functionality is what a technology DOES — "
    "an action, not a category. Labels MUST be verb-led. "
    "GOOD: 'Routes network packets', 'Authenticates users', 'Converts analog to digital signals'. "
    "BAD: 'Network Communication', 'Security and Authentication', 'Signal Processing'. "
    "Never use abstract category nouns like 'Management', 'Governance', 'Architecture', "
    "'Orchestration', 'Engineering', 'Framework'. Describe ACTIONS performed."
)

PROMPT_LABEL_L2 = """Given these {n} functionality phrases (verb-led actions) from a single cluster, identify the action they collectively perform.

PHRASES:
{phrases}

Return strict JSON. CRITICAL: every field describes an ACTION (verb-led), not a category (noun-led).

- "label": 2-6 word VERB-LED phrase describing what these technologies DO. MUST start with a present-tense verb (Routes / Authenticates / Converts / Encodes / Schedules / etc.). NO noun phrases like "X Management" or "Y Architecture".
- "description": 1-2 sentences in active voice describing the common action. Start with a verb.
- "functionalities": list of 1-3 verb-led actions (each starting with a verb)."""

SYS_LABEL_PARENT = (
    "You are creating a FUNCTIONALITY taxonomy. A functionality is what a technology DOES, not "
    "a category it belongs to. Parent labels MUST be verb-led actions that subsume children's actions. "
    "GOOD: 'Manages network traffic and addressing'. BAD: 'Network Communication and Routing'. "
    "Avoid abstract nouns ('Management', 'Architecture', 'Engineering', 'Orchestration')."
)

PROMPT_LABEL_PARENT = """Below are the {n} verb-led child functionalities that share a common parent functionality.

CHILDREN:
{children}

Return strict JSON. CRITICAL: every field describes an ACTION (verb-led).

- "label": 2-6 word VERB-LED phrase describing the parent action. MUST start with a present-tense verb. NO noun-led category names ('X Management', 'Y Architecture', 'Z Engineering').
- "description": 1-2 sentences in active voice describing the parent action. Start with a verb.
- "functionalities": list of 1-3 verb-led actions (each starting with a verb)."""

SYS_TECH_VERIFY = (
    "You verify whether a list of extracted functionalities accurately describes a cluster of "
    "technical phrases. Verdicts: 'correct' (functionalities fully describe the phrases), "
    "'partial' (functionalities are accurate but incomplete), 'incorrect' (functionalities "
    "do not match the phrases)."
)

PROMPT_TECH_VERIFY = """Cluster's top phrases:
{phrases}

Extracted functionalities for this cluster:
{functionalities}

Do the extracted functionalities accurately describe what these phrases collectively express?

Return strict JSON:
- "verdict": one of "correct", "partial", "incorrect"
- "reasoning": one sentence explanation"""

SYS_SUBSUMPTION = (
    "You verify whether a child cluster is a logical sub-functionality of its parent. "
    "Verdicts: 'yes' (clear subset), 'partial' (some overlap but not strict subset), "
    "'no' (child does not belong under parent)."
)

PROMPT_SUBSUMPTION = """Parent cluster:
- label: {parent_label}
- description: {parent_desc}
- functionalities: {parent_funcs}

Child cluster:
- label: {child_label}
- description: {child_desc}
- functionalities: {child_funcs}

Is the child a logical sub-functionality of the parent?

Return strict JSON:
- "verdict": one of "yes", "partial", "no"
- "reasoning": one sentence"""

SYS_EXCLUSIVITY = (
    "You verify whether a group of sibling clusters describe mutually exclusive functionalities. "
    "Verdicts: 'exclusive' (siblings express distinct non-overlapping functionalities), "
    "'overlapping' (some pairs overlap but most are distinct), 'redundant' (siblings express "
    "largely the same functionality and should be merged)."
)

PROMPT_EXCLUSIVITY = """Sibling clusters under parent "{parent_label}":

{siblings}

Are these siblings describing genuinely distinct functionalities, or do some overlap?

Return strict JSON:
- "verdict": one of "exclusive", "overlapping", "redundant"
- "overlap_pairs": list of "label_a -- label_b" strings for pairs that overlap (empty list if exclusive)
- "reasoning": one sentence"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chat(model: str = "google/gemini-3-flash-preview"):
    return _make_chat_fn(
        backend="openrouter",
        api_key=read_openrouter_key(),
        model=model,
        user_agent="st4te-taxonomy-phrase",
    )


def _label_schema() -> dict:
    return {
        "name": "label", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "description": {"type": "string"},
                "functionalities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["label", "description", "functionalities"],
        },
    }


def _verdict_schema(extra_keys: List[str] = ()) -> dict:
    props = {"verdict": {"type": "string"}, "reasoning": {"type": "string"}}
    required = ["verdict", "reasoning"]
    for k in extra_keys:
        props[k] = {"type": "array", "items": {"type": "string"}}
        required.append(k)
    return {
        "name": "verdict", "strict": True,
        "schema": {"type": "object", "properties": props, "required": required},
    }


def _top_phrases(df: pd.DataFrame, layer: str, cluster_id, n: int = 30) -> List[str]:
    sub = df[df[layer] == cluster_id]
    return sub.nlargest(n, "n_pages")["phrase"].astype(str).tolist()


def _safe_call(chat_fn, prompt, system_prompt, schema, required, max_tokens=400, sleep_s=0.3):
    try:
        resp = chat_fn(prompt, system_prompt, max_tokens=max_tokens, json_schema=schema)
        parsed = _validate_json_response(resp, required)
    except Exception as e:
        print(f"  [warn] chat failed: {e}")
        parsed = None
    time.sleep(sleep_s)
    return parsed or {k: "" for k in required}


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def label_L2(df: pd.DataFrame, chat_fn, n_phrases: int = 30) -> Dict:
    """Label every L2 leaf cluster from its top phrases."""
    schema = _label_schema()
    out = {}
    for cid in tqdm(sorted(df["L2"].unique()), desc="Label L2"):
        phrases = _top_phrases(df, "L2", cid, n=n_phrases)
        prompt = PROMPT_LABEL_L2.format(
            n=len(phrases),
            phrases="\n".join(f"- {p}" for p in phrases),
        )
        out[cid] = _safe_call(
            chat_fn, prompt, SYS_LABEL_L2, schema,
            required=["label", "description", "functionalities"],
            max_tokens=500,
        )
    return out


def label_parent(
    df: pd.DataFrame, child_col: str, parent_col: str,
    child_labels: Dict, chat_fn,
) -> Dict:
    """Label each parent cluster from the labels of its children."""
    schema = _label_schema()
    parent_to_children = (
        df.groupby(parent_col)[child_col].apply(lambda s: sorted(s.unique())).to_dict()
    )
    out = {}
    for pid, children in tqdm(parent_to_children.items(), desc=f"Label {parent_col}"):
        rows = []
        for cid in children:
            info = child_labels.get(cid, {})
            rows.append(
                f"- {info.get('label','?')}: {info.get('description','')} "
                f"[functionalities: {', '.join(info.get('functionalities',[]))}]"
            )
        prompt = PROMPT_LABEL_PARENT.format(n=len(children), children="\n".join(rows))
        out[pid] = _safe_call(
            chat_fn, prompt, SYS_LABEL_PARENT, schema,
            required=["label", "description", "functionalities"],
            max_tokens=500,
        )
    return out


def label_phrase_taxonomy(
    df: pd.DataFrame, out_dir: Path,
    *, model: str = "google/gemini-3-flash-preview", n_phrases: int = 30,
) -> Dict[str, Dict]:
    """Bottom-up label all clusters: L2 → L1 → root.

    Input DataFrame must have columns: ``phrase, n_pages, root, L1, L2``.
    Writes ``labels_{root,L1,L2}.csv`` to ``out_dir`` and returns a dict
    ``{level: {cluster_id: label_dict}}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chat_fn = _make_chat(model)

    labels = {}

    print("\n=== Labelling L2 (leaves) ===")
    labels["L2"] = label_L2(df, chat_fn, n_phrases=n_phrases)
    _save_labels(labels["L2"], out_dir / "labels_L2.csv")

    print("\n=== Labelling L1 ===")
    labels["L1"] = label_parent(df, "L2", "L1", labels["L2"], chat_fn)
    _save_labels(labels["L1"], out_dir / "labels_L1.csv")

    print("\n=== Labelling root ===")
    labels["root"] = label_parent(df, "L1", "root", labels["L1"], chat_fn)
    _save_labels(labels["root"], out_dir / "labels_root.csv")

    # Also write a combined JSON for easy reuse
    (out_dir / "labels_all.json").write_text(
        json.dumps({lvl: {str(k): v for k, v in d.items()} for lvl, d in labels.items()}, indent=2),
        encoding="utf-8",
    )
    return labels


def _save_labels(d: Dict, path: Path):
    rows = []
    for cid, info in d.items():
        rows.append({
            "cluster_id": cid,
            "label": info.get("label", ""),
            "description": info.get("description", ""),
            "functionalities": "; ".join(info.get("functionalities", [])),
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    print(f"  -> {path.name} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

def verify_tech(
    df: pd.DataFrame, l2_labels: Dict, chat_fn, n_phrases: int = 30,
) -> List[dict]:
    schema = _verdict_schema()
    rows = []
    for cid, info in tqdm(l2_labels.items(), desc="Verify L2 functionalities"):
        phrases = _top_phrases(df, "L2", cid, n=n_phrases)
        prompt = PROMPT_TECH_VERIFY.format(
            phrases="\n".join(f"- {p}" for p in phrases),
            functionalities=", ".join(info.get("functionalities", [])),
        )
        parsed = _safe_call(
            chat_fn, prompt, SYS_TECH_VERIFY, schema,
            required=["verdict", "reasoning"], max_tokens=200,
        )
        rows.append({"cluster_id": cid, "label": info.get("label", ""), **parsed})
    return rows


def verify_subsumption(
    df: pd.DataFrame, labels: Dict[str, Dict], chat_fn,
    level_pairs: Tuple[Tuple[str, str], ...] = (("root", "L1"), ("L1", "L2")),
) -> List[dict]:
    schema = _verdict_schema()
    rows = []
    for parent_lvl, child_lvl in level_pairs:
        ptoc = df.groupby(parent_lvl)[child_lvl].apply(lambda s: sorted(s.unique())).to_dict()
        for pid, children in tqdm(ptoc.items(), desc=f"Subsumption {parent_lvl}->{child_lvl}"):
            pinfo = labels[parent_lvl].get(pid, {})
            for cid in children:
                cinfo = labels[child_lvl].get(cid, {})
                prompt = PROMPT_SUBSUMPTION.format(
                    parent_label=pinfo.get("label", ""),
                    parent_desc=pinfo.get("description", ""),
                    parent_funcs=", ".join(pinfo.get("functionalities", [])),
                    child_label=cinfo.get("label", ""),
                    child_desc=cinfo.get("description", ""),
                    child_funcs=", ".join(cinfo.get("functionalities", [])),
                )
                parsed = _safe_call(
                    chat_fn, prompt, SYS_SUBSUMPTION, schema,
                    required=["verdict", "reasoning"], max_tokens=200,
                )
                rows.append({
                    "parent_level": parent_lvl, "parent_id": pid,
                    "parent_label": pinfo.get("label", ""),
                    "child_level": child_lvl, "child_id": cid,
                    "child_label": cinfo.get("label", ""),
                    **parsed,
                })
    return rows


def verify_exclusivity(
    df: pd.DataFrame, labels: Dict[str, Dict], chat_fn,
    level_pairs: Tuple[Tuple[str, str], ...] = (("root", "L1"), ("L1", "L2")),
) -> List[dict]:
    schema = _verdict_schema(extra_keys=["overlap_pairs"])
    rows = []
    for parent_lvl, child_lvl in level_pairs:
        ptoc = df.groupby(parent_lvl)[child_lvl].apply(lambda s: sorted(s.unique())).to_dict()
        for pid, children in tqdm(
            ptoc.items(), desc=f"Exclusivity {parent_lvl} ({child_lvl} siblings)"
        ):
            pinfo = labels[parent_lvl].get(pid, {})
            sib_lines = []
            for cid in children:
                cinfo = labels[child_lvl].get(cid, {})
                sib_lines.append(f"- {cinfo.get('label','?')}: {cinfo.get('description','')}")
            prompt = PROMPT_EXCLUSIVITY.format(
                parent_label=pinfo.get("label", ""),
                siblings="\n".join(sib_lines),
            )
            parsed = _safe_call(
                chat_fn, prompt, SYS_EXCLUSIVITY, schema,
                required=["verdict", "reasoning", "overlap_pairs"], max_tokens=400,
            )
            rows.append({
                "parent_level": parent_lvl, "parent_id": pid,
                "parent_label": pinfo.get("label", ""),
                "n_siblings": len(children),
                **parsed,
            })
    return rows


def validate_phrase_consistency(
    df: pd.DataFrame, labels: Dict[str, Dict], out_dir: Path,
    *, model: str = "google/gemini-3-flash-preview", n_phrases: int = 30,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chat_fn = _make_chat(model)

    print("\n=== 0. Functionality verification (L2) ===")
    tech_rows = verify_tech(df, labels["L2"], chat_fn, n_phrases=n_phrases)
    pd.DataFrame(tech_rows).to_csv(out_dir / "tech_verification.csv", index=False)

    print("\n=== 1. Parent-child subsumption ===")
    sub_rows = verify_subsumption(df, labels, chat_fn)
    pd.DataFrame(sub_rows).to_csv(out_dir / "subsumption.csv", index=False)

    print("\n=== 2. Sibling exclusivity ===")
    excl_rows = verify_exclusivity(df, labels, chat_fn)
    # overlap_pairs is a list -> stringify for CSV
    df_excl = pd.DataFrame(excl_rows)
    if "overlap_pairs" in df_excl.columns:
        df_excl["overlap_pairs"] = df_excl["overlap_pairs"].apply(
            lambda v: "; ".join(v) if isinstance(v, list) else (v or "")
        )
    df_excl.to_csv(out_dir / "exclusivity.csv", index=False)

    summary = {
        "tech_verification": {
            "total": len(tech_rows),
            "verdicts": dict(Counter(r["verdict"] for r in tech_rows)),
            "pct_pass": round(
                100 * sum(1 for r in tech_rows if r["verdict"] in ("correct", "partial"))
                / max(len(tech_rows), 1), 1,
            ),
        },
        "subsumption": {
            "total": len(sub_rows),
            "verdicts": dict(Counter(r["verdict"] for r in sub_rows)),
            "pct_pass": round(
                100 * sum(1 for r in sub_rows if r["verdict"] in ("yes", "partial"))
                / max(len(sub_rows), 1), 1,
            ),
        },
        "exclusivity": {
            "total": len(excl_rows),
            "verdicts": dict(Counter(r["verdict"] for r in excl_rows)),
        },
    }
    (out_dir / "logical_consistency_phrase.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return summary
