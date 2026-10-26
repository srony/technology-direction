"""LLM link review (cleaning step of the digital exposure, Paper 2): apply the reviewed verdicts to a built exposure frame.
Verdicts: notes/validation/finetune/link_review_top50.csv (one row per reviewed link: side, node_L2, code, appln_id,
patent_verdict belongs / move / not_a_digital_technology, move_target, link_verdict keep / zero); protocol in
notes/validation/finetune/link_review_protocol.md.
A patent verdict holds on both sides; zero links are side-specific. At every level (L2, L1, root) where the build
counts the patent, for both value types (plain, demeaned):
  zero link (code e, patent p): remove p's contribution to code e;
  moved patent: move all its contributions to the target L2 node (and that node's L1 and root), except its zero links;
  not a digital technology: remove all its contributions.
A patent's contribution is recomputed with build._build_side's formula (gates C >= tc on its own page, top-10 relevance
keep, A >= ta, within-year citation weight; entity mean over sentences), in its filing year.
Production: run on the built exposure of build_final.py, then re-apply the off-theme list (review_build):
    python -u review.py pb_mpnet_final_yearly pb_mpnet_final_reviewed_yearly
Removals can exceed a cell only in off-theme cells the build already set to zero (asserted); they are clipped and the
off-theme list re-applied, which equals applying the review before the off-theme list."""
from pathlib import Path
import numpy as np, pandas as pd
import core as C

VERDICTS = Path(__file__).resolve().parents[2] / "notes" / "validation" / "finetune" / "link_review_top50.csv"
LEVELS = ("L2", "L1", "root")


def _membership(sh, levels, cols, out_node, tb):
    """level -> node of each patent column in `cols` where the build counts it (node_members on the full tb-filtered tags,
    the build's own tie-breaking), '' where it counts nowhere at that level."""
    tags = sh["tags"][sh["tags"].sim >= tb]                                 # as build._build_side (same row order)
    best = tags.loc[tags.groupby("patent_col")["sim"].idxmax()]
    argmax = np.full(sh["nPat"], -1, np.int64); argmax[best.patent_col.to_numpy()] = best.page_row.to_numpy()
    want = set(int(c) for c in cols)
    mem = {}
    for k in LEVELS:
        lv = levels[k]; m = {}
        for node in sorted(set(lv[argmax[cols]])):
            if node == out_node:
                continue
            r = C.node_members(tags, np.where(lv == node)[0], argmax)
            if r is None:
                continue
            for c in r[0][r[2]]:
                if int(c) in want:
                    m[int(c)] = node
        mem[k] = np.array([m.get(int(c), "") for c in cols], dtype=object)
    return mem, argmax


def _patent_verdicts(verdicts=VERDICTS):
    """(all reviewed links, one patent verdict per (node, patent)). A patent's verdict (belongs / move / not digital) is
    about its node and holds on both sides, whichever side's links the reviewer judged it on; the most severe verdict of
    its links counts (not digital, then move, then belongs)."""
    Va = pd.read_csv(verdicts, dtype={"code": str})
    pv = (Va.assign(o=Va.patent_verdict.map({"not_a_digital_technology": 0, "move": 1, "belongs": 2})).sort_values("o")
          .drop_duplicates(["node_L2", "appln_id"]))
    return Va, pv


def _l2_parents(levels, out_node):
    """L2 node -> its L1 and root (the most frequent over its pages)."""
    return (pd.DataFrame({k: levels[k] for k in LEVELS}).query("L2 != @out_node")
            .groupby("L2").agg(L1=("L1", lambda s: s.mode().iloc[0]), root=("root", lambda s: s.mode().iloc[0])))


def apply_review(df, side, config, sh, verdicts=VERDICTS, out_node="_REMOVED", gated=False):
    """df: a _build_side(yearly=True) frame [code, level, measure, tech_node, year, value] (value = arcsinh for the
    measure, linear for measure+'_lin'). sh: the shared dict of the build (tags filtered at tb, ppage, levels, cit_rel,
    year, pids, nPat). Returns (df, report)."""
    # zero links are side-specific; patent verdicts hold on both sides (_patent_verdicts)
    Va, pv = _patent_verdicts(verdicts)
    act = pv[pv.patent_verdict != "belongs"]
    zero = Va[(Va.side == side) & (Va.link_verdict == "zero")]
    aff = pd.concat([act[["node_L2", "appln_id"]], zero[["node_L2", "appln_id"]]]).drop_duplicates().reset_index(drop=True)
    aff = aff.merge(act[["node_L2", "appln_id", "patent_verdict", "move_target"]], on=["node_L2", "appln_id"], how="left")
    aff["patent_verdict"] = aff.patent_verdict.fillna("belongs"); aff["move_target"] = aff.move_target.fillna("")
    zset = set(zip(zero.node_L2, zero.code, zero.appln_id))
    levels = {k: np.asarray(v, dtype=object) for k, v in sh["levels"].items()}
    l2s = set(levels["L2"]) - {out_node}
    bad = aff[(aff.patent_verdict == "move") & ~aff.move_target.isin(l2s)]
    assert bad.empty, f"move targets that are not L2 nodes: {sorted(bad.move_target.unique())[:5]}"
    up = _l2_parents(levels, out_node)
    pids = np.asarray(sh["pids"]).astype(np.int64)
    col_of = pd.Series(np.arange(len(pids)), index=pids)
    cols = col_of.loc[aff.appln_id.to_numpy()].to_numpy()
    mem, argmax = _membership(sh, levels, cols, out_node, config["tb"])
    assert (mem["L2"] == aff.node_L2.to_numpy()).all(), "a reviewed patent is not counted in its reviewed L2 node"
    yr = sh["year"][cols]

    fam = config["family"]; tc = config["tc"]; ta = None if config["ta"] == "adaptive" else float(config["ta"])
    codes, ents, ec, s2e, codecol = C.entity_meta(fam, side); nE = len(ents); ents_a = np.asarray(ents)
    M = np.load(C.a_dir(config["a_embed"]) / C._indf(side, C.A_FILES[side]), mmap_mode="r")
    baseline = M.mean(1).astype(np.float32)
    ptask = C.load_ptask(side)
    cnt = np.bincount(s2e, minlength=nE).astype(np.float64)
    S = np.zeros((nE, len(s2e)), np.float32); S[s2e, np.arange(len(s2e))] = 1.0 / cnt[s2e]
    vals = {"plain": np.zeros((nE, len(cols))), "dem": np.zeros((nE, len(cols)))}
    order = np.argsort(cols)
    for i0 in range(0, len(cols), 5000):
        j = order[i0:i0 + 5000]; c = cols[j]
        A = np.asarray(M[:, c]); dem = A - baseline[:, None]
        km = C.keep_single(A, baseline, s2e, nE, K=10)[s2e]
        Cc = ptask @ sh["ppage"][argmax[c]].T
        gate = (Cc >= tc) * km
        if ta is not None:
            gate = gate * (A >= ta)
        cc = sh["cit_rel"][c][None, :]
        vals["plain"][:, j] = S @ (np.clip(A, 0, None) * cc * gate).astype(np.float32)
        vals["dem"][:, j] = S @ (np.clip(dem, 0, None) * cc * gate).astype(np.float32)
    del M

    meas = {"plain": "pb_plain_mean", "dem": "pb_dem_mean"}
    rows = []
    for i, x in enumerate(aff.itertuples()):
        for vt, mn in meas.items():
            v = vals[vt][:, i]
            for e in np.nonzero(v)[0]:
                z = (x.node_L2, ents_a[e], x.appln_id) in zset
                for k in LEVELS:
                    src = mem[k][i]
                    if not src:
                        continue
                    if x.patent_verdict in ("move", "not_a_digital_technology") or z:
                        rows.append((ents_a[e], k, mn + "_lin", src, int(yr[i]), -v[e]))
                    if x.patent_verdict == "move" and not z:
                        dst = x.move_target if k == "L2" else up.loc[x.move_target, k]
                        rows.append((ents_a[e], k, mn + "_lin", dst, int(yr[i]), v[e]))
    D = pd.DataFrame(rows, columns=[codecol, "level", "measure", "tech_node", "year", "delta"])
    D = D.groupby([codecol, "level", "measure", "tech_node", "year"], as_index=False).delta.sum()
    key = [codecol, "level", "measure", "tech_node", "year"]
    lin = df[df.measure.str.endswith("_lin")].merge(D, on=key, how="outer").fillna({"value": 0.0, "delta": 0.0})
    lin["old"] = lin.value
    lin["value"] = lin.value + lin.delta
    neg = lin[lin.value < -1e-6 * np.maximum(1.0, lin.old.abs())]          # beyond float rounding of the original cell
    if gated:                                         # applied to a built (gated) exposure: removals can only exceed
        import cleaning as CL                         # the off-theme cells the build already set to zero
        g_, _ = CL.apply_digital(neg.assign(value=1.0), side)
        if not (g_.value == 0).all():
            import os
            if os.environ.get("REVIEW_DIAG"):
                bad = neg[(g_.value != 0).to_numpy()].copy()
                bad = bad.merge(df[df.measure.str.endswith("_lin")].rename(columns={"value": "build_value"}),
                                on=key, how="left")
                bad.to_csv(os.environ["REVIEW_DIAG"] + f"_{side}.csv", index=False)
        assert (g_.value == 0).all(), f"review removed more than a cell holds outside off-theme cells ({side})"
    else:
        assert neg.empty, f"review removed more than a cell holds ({side}): {float((-neg.value).max()):.3g}"
    lin = lin[lin.value > 1e-9 * np.maximum(1.0, lin.old.abs())].drop(columns=["delta", "old"])
    ash = lin.assign(measure=lin.measure.str.replace("_lin", "", regex=False), value=np.arcsinh(lin.value))
    out = pd.concat([ash, lin], ignore_index=True)[df.columns]
    rep = {"patents_reviewed_affected": int(len(aff)), "moved": int((aff.patent_verdict == "move").sum()),
           "not_digital": int((aff.patent_verdict == "not_a_digital_technology").sum()), "zero_links": int(len(zset)),
           "cells_changed": int((D.delta != 0).sum()), "negative_offtheme_cells": int(len(neg)),
           "rows_before": int(len(df)), "rows_after": int(len(out))}
    return out, rep


def final_shared():
    """(shared dict, config) of the final digital build: the build's tags and pages, with build_final.py's page list
    (plan, removed quantum pages) and its added page(s) matched to every patent at >= tb. Used by review_build and
    final_membership."""
    import build_final as BF
    from add_page import page_text, embed
    cfg = C.CFG[BF.CFG_KEY]
    sh = C.load_shared(C.TAGFILE[cfg["family"]], with_cit=True)
    pm = pd.read_parquet(C.PBAN / "page_meta_patentsberta.parquet")
    plan = pd.read_csv(BF.PLAN); plan = plan[plan.action.isin(["MOVE", "MOVE-NEW", "DROP"])]
    lv = BF.page_list(pm, plan)
    X, _ = C.load_embeddings(mode="digital", view="abstract", embed_model=C.PB_HF)
    ppage, tags = sh["ppage"], sh["tags"]
    for page, (l2, l1, root) in BF.ADD.items():
        e = embed([page_text(page)])
        s_ = np.concatenate([(C.unit(np.asarray(X[i:i + 50000])) @ e.T).ravel() for i in range(0, sh["nPat"], 50000)])
        row = len(ppage); ppage = np.vstack([ppage, e]); hit = np.where(s_ >= cfg["tb"])[0]
        tags = pd.concat([tags, pd.DataFrame({"patent_col": hit, "page_row": row, "sim": s_[hit].astype(np.float32)})], ignore_index=True)
        for k, v in (("L2", l2), ("L1", l1), ("root", root)):
            lv[k] = np.append(lv[k], v)
    del X
    return dict(sh, ppage=ppage, tags=tags, levels={k: np.asarray(v, dtype=object) for k, v in lv.items()}), cfg


def final_membership(shared=None):
    """The patents the final reviewed digital build counts, one row per patent: appln_id, year (the build's filing year;
    NaN where the patent has none, so the yearly build never counts it), L2, L1, root ('' where the build counts the
    patent nowhere at that level). Membership as in build._build_side on the final pages (final_shared): tb-filtered tags,
    argmax page, core.node_members' tie rule, one-to-one. Then the review's patent verdicts, at every level where the
    patent counts, as apply_review moves the contributions: not a digital technology -> counted nowhere; move -> the
    target L2 node and that node's L1 and root. `shared`: a final_shared() result to reuse (it loads all embeddings)."""
    import build_final as BF
    sh, cfg = shared or final_shared()
    levels = sh["levels"]
    mem, _ = _membership(sh, levels, np.arange(sh["nPat"]), BF.OUT_NODE, cfg["tb"])
    out = pd.DataFrame({"appln_id": np.asarray(sh["pids"]).astype(np.int64), "year": sh["year"], **mem})
    _, pv = _patent_verdicts()
    act = pv[pv.patent_verdict != "belongs"].reset_index(drop=True)
    assert act.appln_id.is_unique, "a reviewed patent carries verdicts in two nodes"
    rows = pd.Series(np.arange(len(out)), index=out.appln_id.to_numpy()).loc[act.appln_id.to_numpy()].to_numpy()
    assert (out.L2.to_numpy()[rows] == act.node_L2.to_numpy()).all(), "a reviewed patent is not counted in its reviewed L2 node"
    up = _l2_parents(levels, BF.OUT_NODE)
    nd = (act.patent_verdict == "not_a_digital_technology").to_numpy()
    mv = (act.patent_verdict == "move").to_numpy()
    tgt = act.move_target.to_numpy()
    for k in LEVELS:
        v = out[k].to_numpy().copy()
        has = v[rows] != ""
        v[rows[nd & has]] = ""
        dst = tgt if k == "L2" else up.reindex(tgt)[k].to_numpy()
        v[rows[mv & has]] = dst[mv & has]
        out[k] = v
    return out[(out[list(LEVELS)] != "").any(axis=1)].reset_index(drop=True)


def review_build(src, out):
    """Apply the link review to an existing build folder (e.g. pb_mpnet_final_yearly) and write it to `out`, both under
    core.TRI. Same pages, added page and levels as build_final.py; the off-theme list is re-applied after the review, so
    the result equals running the review inside the build before the off-theme list.
        python -u review.py pb_mpnet_final_yearly pb_mpnet_final_reviewed_yearly"""
    import json
    import cleaning as CL
    import build_final as BF
    src, out = C.TRI / src, C.TRI / out
    sh2, cfg = final_shared()
    (out / "exposure").mkdir(parents=True, exist_ok=True)
    rep = {}
    for side, fn in (("occ", "occupation"), ("ind", "industry")):
        df = pd.read_parquet(src / "exposure" / f"{fn}.parquet")
        df, rr = apply_review(df, side, cfg, sh2, out_node=BF.OUT_NODE, gated=True)
        df, z = CL.apply_digital(df, side)
        df.to_parquet(out / "exposure" / f"{fn}.parquet", index=False)
        rep[fn] = dict(rr, rows=len(df), cells_zeroed_offtheme=z)
        print(f"  {fn}: {rep[fn]}", flush=True)
    (out / "README.json").write_text(json.dumps({"source": str(src), "step": "review.review_build (LLM link review)",
                                                 "verdicts": str(VERDICTS), "report": rep, "script": "p3_exposure/threshold_experiment/review.py"}, indent=1))


if __name__ == "__main__":
    import sys
    review_build(sys.argv[1], sys.argv[2])
