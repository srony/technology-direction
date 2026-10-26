"""Final digital exposure build in ONE pass from the raw patent-page matches (2026-10-02), replacing the chain of patched
builds (finetuned -> _changes -> _cloud -> _offtheme2 -> _qdrop).

Fix 1, taxonomy pages (applied to the page list, then build._build_side runs unchanged, yearly, pb_mpnet_plain_1to1):
  - page-handling plan (notes/validation/finetune/page_handling_plan.csv): MOVE puts the page in the target node;
    MOVE-NEW puts it in a new node (own L1, root of its old node), incl. Service & Domestic Robotics and
    Medical & Laboratory Robotics; DROP removes the page.
  - 'Cloud computing' added to Storage, Virtualization & File Systems (it was removed by the LLM filter as a "paradigm").
  - six word-collision pages of the Quantum Computing root removed.
  A removed page's patents carry no exposure: they are not passed on to their next-best page (as in every earlier build).
Fix 2, semantic-only cells: the off-theme zero list (cleaning.apply_digital) applied once to the result.

Check: every cell must equal the patched build pb_mpnet_finetuned_changes_cloud_offtheme2_qdrop_yearly, except in the
new Medical & Laboratory Robotics node and the L1/root above it (that node was missing from the patched builds).

    cd p3_exposure/threshold_experiment
    python -u build_final.py pb_mpnet_final_yearly
"""
import json, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from pathlib import Path
import numpy as np, pandas as pd
import core as C
import cleaning as CL
import build as B
from add_page import page_text, embed

CFG_KEY = "pb_mpnet_plain_1to1"
PLAN = Path(__file__).resolve().parents[2] / "notes" / "validation" / "finetune" / "page_handling_plan.csv"
ADD = {"Cloud computing": ("Storage, Virtualization & File Systems", "System Software & Security", "Computing Hardware & Systems")}
REMOVE = ["Cloud-based quantum computing", "Algorithmic cooling", "Quantum state discrimination", "Quantum tomography",
          "Neural network quantum states", "Pulse programming"]
OUT_NODE = "_REMOVED"
CHECK_AGAINST = C.TRI / "pb_mpnet_finetuned_changes_cloud_offtheme2_qdrop_yearly"


def page_list(pm, plan):
    """-> page levels {L2, L1, root} after the plan, with removed pages under OUT_NODE."""
    title = pm.page_title.astype(str).to_numpy()
    lv = {k: pm[k].astype(str).to_numpy().astype(object) for k in ("L2", "L1", "root")}
    up = {n: (g.L1.mode().iloc[0], g.root.mode().iloc[0]) for n, g in pm.assign(L2=lv["L2"], L1=lv["L1"], root=lv["root"]).groupby("L2")}
    for r in plan.itertuples():
        i = np.where(title == r.page)[0]
        assert len(i) == 1, f"plan page not found once: {r.page}"
        i = i[0]
        if r.action == "MOVE":
            assert r.target in up, f"MOVE target is not a node: {r.target}"
            lv["L2"][i], (lv["L1"][i], lv["root"][i]) = r.target, up[r.target]
        elif r.action == "MOVE-NEW":
            assert r.target not in up, f"MOVE-NEW target already a node: {r.target}"
            lv["L2"][i], lv["L1"][i] = r.target, r.target            # root stays that of the page's old node
        elif r.action == "DROP":
            lv["L2"][i] = lv["L1"][i] = lv["root"][i] = OUT_NODE
        else:
            raise ValueError(r.action)
    i = np.where(np.isin(title, REMOVE))[0]
    assert len(i) == len(REMOVE) and not plan.page.isin(REMOVE).any(), "quantum pages not found, or already in the plan"
    for k in lv:
        lv[k][i] = OUT_NODE
    return lv


def main(out_name):
    OUT = C.TRI / out_name
    cfg = C.CFG[CFG_KEY]
    sh = C.load_shared(C.TAGFILE[cfg["family"]], with_cit=True)
    pm = pd.read_parquet(C.PBAN / "page_meta_patentsberta.parquet")
    plan = pd.read_csv(PLAN)
    plan = plan[plan.action.isin(["MOVE", "MOVE-NEW", "DROP"])]
    lv = page_list(pm, plan)

    # added page(s): embedding with the recipe of the stored page vectors, matches with every patent at >= tb
    chk = [0, 1392, 5118]
    stored = embed(pd.read_csv(C.DROPBOX / "taxonomy" / "wiki" / "pipeline" / "07_tag_patents" / "digital_v3" / "taxonomy_used.csv").description.astype(str).iloc[chk])
    assert (np.sum(stored * sh["ppage"][chk], 1) > 0.999).all(), "page embedding recipe does not reproduce stored page vectors"
    X, _ = C.load_embeddings(mode="digital", view="abstract", embed_model=C.PB_HF)
    ppage, tags = sh["ppage"], sh["tags"]
    for page, (l2, l1, root) in ADD.items():
        assert page not in set(pm.page_title.astype(str)), f"{page} is already a page"
        e = embed([page_text(page)])
        s = np.concatenate([(C.unit(np.asarray(X[i:i + 50000])) @ e.T).ravel() for i in range(0, sh["nPat"], 50000)])
        row = len(ppage); ppage = np.vstack([ppage, e])
        hit = np.where(s >= cfg["tb"])[0]
        tags = pd.concat([tags, pd.DataFrame({"patent_col": hit, "page_row": row, "sim": s[hit].astype(np.float32)})], ignore_index=True)
        for k, v in (("L2", l2), ("L1", l1), ("root", root)):
            lv[k] = np.append(lv[k], v)
        print(f"added page '{page}' -> {l2}: matched by {len(hit):,} patents at sim >= {cfg['tb']}", flush=True)
    del X
    sh2 = dict(sh, ppage=ppage, tags=tags, levels={k: np.asarray(v, dtype=object) for k, v in lv.items()})
    print(f"pages removed: {int((lv['L2'] == OUT_NODE).sum())} (plan DROP {int((plan.action == 'DROP').sum())} + quantum {len(REMOVE)}); "
          f"L2 nodes: {len(set(lv['L2']) - {OUT_NODE})}", flush=True)

    d = OUT / "exposure"; d.mkdir(parents=True, exist_ok=True)
    rep = {}
    for side, fn in (("occ", "occupation"), ("ind", "industry")):
        df = B._build_side(cfg, side, sh2, yearly=True)
        df = df[df.tech_node != OUT_NODE].reset_index(drop=True)
        df, z = CL.apply_digital(df, side)                       # fix 2
        df.to_parquet(d / f"{fn}.parquet", index=False)
        rep[fn] = {"rows": len(df), "cells_zeroed_fix2": z}
        print(f"  {fn}: {len(df):,} rows, {z:,} zeroed by the off-theme list", flush=True)

        # check against the patched build (deleted 2026-10-16 with the other old builds: the check is skipped then)
        if not (CHECK_AGAINST / "exposure" / f"{fn}.parquet").exists():
            print(f"  {fn}: patched build not on disk, check skipped", flush=True)
            continue
        key = [df.columns[0], "level", "measure", "tech_node", "year"]
        old = pd.read_parquet(CHECK_AGAINST / "exposure" / f"{fn}.parquet")
        m = df.merge(old, on=key, how="outer", suffixes=("", "_old")).fillna({"value": 0.0, "value_old": 0.0})
        med = "Medical & Laboratory Robotics"
        expect = m.tech_node.isin([med, "Robotics & Manufacturing"])
        diff = (m.value - m.value_old).abs()
        bad = m[~expect & (diff > 1e-6 * np.maximum(1.0, m.value_old.abs()))]
        rep[fn]["check_cells_differing_outside_medical"] = len(bad)
        print(f"  check vs patched build: cells differing outside {med} and its root: {len(bad):,}", flush=True)
        if len(bad):
            print(bad.groupby(["level", "tech_node"]).size().sort_values(ascending=False).head(20).to_string(), flush=True)
        print(f"  {med}: {int((m.tech_node == med).sum()):,} rows; root Robotics & Manufacturing lin 2009-2023 "
              f"{m[(m.tech_node == 'Robotics & Manufacturing') & m.measure.str.endswith('_lin') & m.year.between(2009, 2023)].value.sum():.1f} "
              f"(patched {m[(m.tech_node == 'Robotics & Manufacturing') & m.measure.str.endswith('_lin') & m.year.between(2009, 2023)].value_old.sum():.1f})", flush=True)
    (OUT / "README.json").write_text(json.dumps({
        "script": "p3_exposure/threshold_experiment/build_final.py", "config": CFG_KEY,
        "fix1": {"plan": str(PLAN), "added_pages": ADD, "removed_pages": REMOVE},
        "fix2": "cleaning.apply_digital (OCC_OFF / IND_OFF)", "report": rep}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
