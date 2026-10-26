"""Run the direction panel on a grid of technology x occupation x industry triplets.

Each triplet receives 2 constructs x 3 personas x DRAWS ratings, one construct and one persona per call. Answers are
appended to the output CSV as they arrive, and a rerun skips the calls already answered with a rating on the 0 to 5
scale. The grid file needs the columns domain, tech_node, isco3, section, occ_title, sec_title, node_label.

    set OPENROUTER_API_KEY=...
    python rate.py --grid grid.csv --out my_ratings.csv
    python rate.py --grid grid.csv --out pilot.csv --limit 20        # 20 triplets, half digital and half green
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import llm
from prompts import PERSONAS, STATEMENTS, system_prompt, user_prompt

COLS = ["call_id", "batch", "domain", "tech_node", "node_label", "isco3", "occ_title", "section", "sec_title",
        "occ_exp", "ind_exp", "construct", "persona", "draw", "rating", "reason", "failed"]


def usable(res):
    r = pd.to_numeric(pd.Series([None if res is None else res.get("rating")]), errors="coerce").iloc[0]
    return bool(pd.notna(r) and 0 <= r <= 5)


def one(job, grid, cfg, hdrs):
    cid, idx, construct, pj, draw = job
    row, persona = grid.iloc[idx], PERSONAS[pj]
    res = llm.ask(user_prompt(row, construct, persona), system_prompt(row["domain"]), cfg.model, cfg.temperature, cfg.top_p, hdrs)
    return {"call_id": cid, "batch": row.get("batch", ""), "domain": row["domain"], "tech_node": row["tech_node"],
            "node_label": row["node_label"], "isco3": row["isco3"], "occ_title": row["occ_title"], "section": row["section"],
            "sec_title": row["sec_title"], "occ_exp": row.get("occ_exp", ""), "ind_exp": row.get("ind_exp", ""),
            "construct": construct, "persona": persona["name"], "draw": draw,
            "rating": None if res is None else res.get("rating"),
            "reason": "" if res is None else str(res.get("reason", "")).replace("\n", " "),
            "failed": not usable(res)}


def jobs_for(grid, draws, done):
    out = []
    for idx, row in grid.iterrows():
        for construct in STATEMENTS:
            for pj, persona in enumerate(PERSONAS):
                for s in range(1, draws + 1):
                    cid = f"{row['isco3']}|{row['section']}|{row['tech_node']}|{construct}|{persona['name']}|{s}"
                    if cid not in done:
                        out.append((cid, idx, construct, pj, s))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True, help="ratings CSV, appended to and resumed from")
    ap.add_argument("--model", default="deepseek/deepseek-chat-v3.1")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.3)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="rate only this many triplets, half per domain (seed 42)")
    cfg = ap.parse_args()

    grid = pd.read_csv(cfg.grid, dtype={"isco3": str, "section": str}).reset_index(drop=True)
    if cfg.limit:
        keep = []
        for dom in ("digital", "green"):
            cand = grid[grid.domain == dom]
            keep += list(cand.sample(n=min(cfg.limit - len(keep) if dom == "green" else cfg.limit // 2, len(cand)), random_state=42).index)
        grid = grid.loc[sorted(keep)].reset_index(drop=True)
    done = set()
    try:
        prev = pd.read_csv(cfg.out, usecols=["call_id", "failed", "rating"])
        ok = ~prev.failed.astype(bool) & pd.to_numeric(prev.rating, errors="coerce").between(0, 5)
        done = set(prev.loc[ok, "call_id"])
        print(f"resume: {len(done):,} calls already answered")
    except FileNotFoundError:
        pass
    jobs = jobs_for(grid, cfg.draws, done)
    print(f"{len(grid):,} triplets, {len(jobs):,} calls to run with {cfg.model}", flush=True)
    hdrs = llm.headers()
    buf, n_fail, t0 = [], 0, time.time()

    def flush():
        nonlocal buf
        if buf:
            pd.DataFrame(buf, columns=COLS).to_csv(cfg.out, mode="a", header=not _exists(cfg.out), index=False)
            buf = []

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = [ex.submit(one, j, grid, cfg, hdrs) for j in jobs]
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


def _exists(p):
    import os
    return os.path.exists(p)


if __name__ == "__main__":
    main()
