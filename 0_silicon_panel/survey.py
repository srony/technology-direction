"""Regenerate a survey with the silicon panel: one persona-conditioned call per case and draw.

The case file needs the columns case_id, question (substitutability, complementarity, routinised or efficiency), tech,
industry, task, gender, role, organization, educbackground, education and region. A rating_survey column, if present,
is carried through to the output so that metrics.py can score the run. Answers are appended to the output CSV as they
arrive, and a rerun skips the calls already answered with a rating on the 0 to 5 scale.

    set OPENROUTER_API_KEY=...
    python survey.py --cases cases.csv --out ratings.csv                       # knee configuration, five draws
    python survey.py --cases cases.csv --out grid_T1_P075.csv --temperature 1.0 --top-p 0.75
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import llm
from prompts import CASE_FIELDS, PROFILE_FIELDS, STATEMENTS, SYSTEM_PROMPT, user_prompt

KEEP = ["case_id", "question", "rating_survey"] + CASE_FIELDS + PROFILE_FIELDS
COLS = ["call_id", "case_id", "draw", "question", "rating", "reason", "failed", "model", "temperature", "top_p",
        "rating_survey"] + CASE_FIELDS + PROFILE_FIELDS


def usable(res):
    r = pd.to_numeric(pd.Series([None if res is None else res.get("rating")]), errors="coerce").iloc[0]
    return bool(pd.notna(r) and 0 <= r <= 5)


def one(job, cases, cfg, hdrs):
    cid, idx, draw = job
    row = cases.iloc[idx]
    res = llm.ask(user_prompt(row), SYSTEM_PROMPT, cfg.model, cfg.temperature, cfg.top_p, hdrs)
    out = {"call_id": cid, "case_id": row["case_id"], "draw": draw, "question": row["question"],
           "rating": None if res is None else res.get("rating"),
           "reason": "" if res is None else str(res.get("reason", "")).replace("\n", " "),
           "failed": not usable(res), "model": cfg.model, "temperature": cfg.temperature, "top_p": cfg.top_p}
    for f in ["rating_survey"] + CASE_FIELDS + PROFILE_FIELDS:
        out[f] = row.get(f, "")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True, help="ratings CSV, appended to and resumed from")
    ap.add_argument("--model", default="deepseek/deepseek-chat-v3.1")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.3)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    cfg = ap.parse_args()

    cases = pd.read_csv(cfg.cases, dtype=str).fillna("")
    missing = [c for c in ["case_id", "question"] + CASE_FIELDS + PROFILE_FIELDS if c not in cases.columns]
    if missing:
        sys.exit(f"case file lacks columns: {missing}")
    bad = sorted(set(cases.question.str.lower()) - set(STATEMENTS))
    if bad:
        sys.exit(f"unknown question values: {bad}")
    cases["question"] = cases.question.str.lower()
    if cfg.limit:
        cases = cases.head(cfg.limit)
    cases = cases.reset_index(drop=True)
    done = set()
    if os.path.exists(cfg.out):
        prev = pd.read_csv(cfg.out, usecols=["call_id", "failed", "rating"])
        ok = ~prev.failed.astype(bool) & pd.to_numeric(prev.rating, errors="coerce").between(0, 5)
        done = set(prev.loc[ok, "call_id"])
        print(f"resume: {len(done):,} calls already answered")
    jobs = [(f"{row.case_id}|{s}", idx, s) for idx, row in cases.iterrows() for s in range(1, cfg.draws + 1)
            if f"{row.case_id}|{s}" not in done]
    print(f"{len(cases):,} cases, {len(jobs):,} calls to run with {cfg.model} at T {cfg.temperature}, top-p {cfg.top_p}", flush=True)
    hdrs = llm.headers("survey panel")
    buf, n_fail, t0 = [], 0, time.time()

    def flush():
        nonlocal buf
        if buf:
            pd.DataFrame(buf, columns=COLS).to_csv(cfg.out, mode="a", header=not os.path.exists(cfg.out), index=False)
            buf = []

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = [ex.submit(one, j, cases, cfg, hdrs) for j in jobs]
        for k, f in enumerate(as_completed(futures), 1):
            r = f.result(); buf.append(r); n_fail = n_fail + 1 if r["failed"] else 0
            if len(buf) >= 200:
                flush()
            if k % 1000 == 0:
                print(f"  {k:,}/{len(jobs):,} calls, {time.time() - t0:.0f}s", flush=True)
            if n_fail >= 50:
                flush(); sys.exit("50 consecutive failures, stopping (key, credits or rate limit)")
    flush()
    print("done")


if __name__ == "__main__":
    main()
