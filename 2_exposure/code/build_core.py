"""Final digital exposure build on the final taxonomy (taxonomy_core.py; user decision 2026-10-07). Same chain as
build_final.py -> review.review_build -> cleaning.py, on the relabelled pages:
  pages: the build_final.py page list relabelled by taxonomy_core.final_levels(); pages removed as not digital (R1, S14)
         -> _REMOVED (their patents count nowhere, the build_final rule); duplicate pages (D1) are taken OUT OF THE MATCHES,
         so the kept page of each identical group receives the patents that the tie rule dropped;
  build: build._build_side unchanged; off-theme cleaning = cleaning.py lists with the node names mapped to the final names;
  review: review.apply_review logic with every verdict re-keyed to the patent's final node; move targets mapped
          (renamed/merged nodes); targets the taxonomy removes -> treated as not digital.
Writes TRI/pb_mpnet_final_reviewed_q_taxcore_yearly/{exposure/*.parquet, membership.parquet, levels.csv, page_meta.parquet,
README.json}. The engine's CLEANED digital build (analysis.DIG["cleaned"]).

    cd p3_exposure/threshold_experiment
    python -u build_core.py
"""
import os, sys, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd
import core as C, build as B, cleaning as CL, review as RV, build_final as BF
import taxonomy_core as TX
t0 = time.time(); say = lambda *a: print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)
OUT = TX.FINAL
mapn = lambda n: None if n in TX.REMOVED_NODES else TX.NAME_MAP.get(n, n)


def mapped(dct):
    out = {}
    for k, v in dct.items():
        k2 = mapn(k)
        if k2 is not None:
            out.setdefault(k2, []); out[k2] = list(dict.fromkeys(out[k2] + list(v)))
    return out


OCC_OFF, IND_OFF = mapped(CL.OCC_OFF), mapped(CL.IND_OFF)
OCC_CODES, IND_CODES = mapped(CL.OCC_OFF_CODES), mapped(CL.IND_OFF_CODES)


def clean(df, side):
    """cleaning.apply_digital with the node names mapped to the final taxonomy."""
    nm, code = CL._names(side); OFF = OCC_OFF if side == "occ" else IND_OFF
    df = df.copy(); nmcol = df[code].astype(str).map(lambda c: nm.get(c, c)); z = 0
    for node, subs in OFF.items():
        m = (df.tech_node == node) & nmcol.apply(lambda n: any(s in n for s in subs)); df.loc[m, "value"] = 0.0; z += int(m.sum())
    for node, codes in (OCC_CODES if side == "occ" else IND_CODES).items():
        m = (df.tech_node == node) & df[code].astype(str).isin(codes); df.loc[m, "value"] = 0.0; z += int(m.sum())
    return df, z


def verdicts(sh, levels, tb):
    """Review verdicts re-keyed to the final taxonomy: one patent verdict per patent at its final L2 node; zero links keyed
    (final node, code, patent)."""
    Va, pv = RV._patent_verdicts()
    pids = np.asarray(sh["pids"]).astype(np.int64); col_of = pd.Series(np.arange(len(pids)), index=pids)
    allp = pd.unique(Va.appln_id); cols = col_of.loc[allp].to_numpy()
    mem, argmax = RV._membership(sh, levels, cols, BF.OUT_NODE, tb)
    node_of = dict(zip(allp, mem["L2"]))
    pv = pv.copy(); pv["node"] = pv.appln_id.map(node_of)
    pv["move_target"] = pv.move_target.fillna("").map(lambda t: (mapn(t) or "__REMOVED__") if t else "")
    pv.loc[(pv.patent_verdict == "move") & (pv.move_target == "__REMOVED__"), "patent_verdict"] = "not_a_digital_technology"
    pv.loc[(pv.patent_verdict == "move") & (pv.move_target == pv.node), "patent_verdict"] = "belongs"
    pv = pv.assign(o=pv.patent_verdict.map({"not_a_digital_technology": 0, "move": 1, "belongs": 2})).sort_values("o")
    pv = pv.drop_duplicates("appln_id")
    l2s = set(levels["L2"]) - {BF.OUT_NODE}
    bad = pv[(pv.patent_verdict == "move") & ~pv.move_target.isin(l2s)]
    assert bad.empty, f"move targets not in the final taxonomy: {sorted(bad.move_target.unique())[:6]}"
    Va = Va.assign(node=Va.appln_id.map(node_of))
    return Va, pv


def apply_review_final(df, side, config, sh, Va, pv):
    """review.apply_review, verdicts keyed to the final node of each patent."""
    LEVELS = RV.LEVELS
    act = pv[pv.patent_verdict != "belongs"]
    zero = Va[(Va.side == side) & (Va.link_verdict == "zero") & Va.node.notna() & (Va.node != "")]
    aff = pd.concat([act[["node", "appln_id"]], zero[["node", "appln_id"]]]).drop_duplicates("appln_id").reset_index(drop=True)
    aff = aff.merge(act[["appln_id", "patent_verdict", "move_target"]], on="appln_id", how="left")
    aff["patent_verdict"] = aff.patent_verdict.fillna("belongs"); aff["move_target"] = aff.move_target.fillna("")
    zset = set(zip(zero.node, zero.code, zero.appln_id))
    levels = {k: np.asarray(v, dtype=object) for k, v in sh["levels"].items()}
    up = RV._l2_parents(levels, BF.OUT_NODE)
    pids = np.asarray(sh["pids"]).astype(np.int64); col_of = pd.Series(np.arange(len(pids)), index=pids)
    cols = col_of.loc[aff.appln_id.to_numpy()].to_numpy()
    mem, argmax = RV._membership(sh, levels, cols, BF.OUT_NODE, config["tb"])
    yr = sh["year"][cols]
    fam = config["family"]; tc = config["tc"]; ta = None if config["ta"] == "adaptive" else float(config["ta"])
    codes, ents, ec, s2e, codecol = C.entity_meta(fam, side); nE = len(ents); ents_a = np.asarray(ents)
    M = np.load(C.a_dir(config["a_embed"]) / C._indf(side, C.A_FILES[side]), mmap_mode="r")
    baseline = M.mean(1).astype(np.float32); ptask = C.load_ptask(side)
    cnt = np.bincount(s2e, minlength=nE).astype(np.float64)
    S = np.zeros((nE, len(s2e)), np.float32); S[s2e, np.arange(len(s2e))] = 1.0 / cnt[s2e]
    vals = {"plain": np.zeros((nE, len(cols))), "dem": np.zeros((nE, len(cols)))}
    order = np.argsort(cols)
    for i0 in range(0, len(cols), 5000):
        j = order[i0:i0 + 5000]; c = cols[j]
        A = np.asarray(M[:, c]); dem = A - baseline[:, None]
        km = C.keep_single(A, baseline, s2e, nE, K=10)[s2e]
        gate = (ptask @ sh["ppage"][argmax[c]].T >= tc) * km
        if ta is not None:
            gate = gate * (A >= ta)
        cc = sh["cit_rel"][c][None, :]
        vals["plain"][:, j] = S @ (np.clip(A, 0, None) * cc * gate).astype(np.float32)
        vals["dem"][:, j] = S @ (np.clip(dem, 0, None) * cc * gate).astype(np.float32)
    del M
    meas = {"plain": "pb_plain_mean", "dem": "pb_dem_mean"}; rows = []
    for i, x in enumerate(aff.itertuples()):
        for vt, mn in meas.items():
            v = vals[vt][:, i]
            for e in np.nonzero(v)[0]:
                z = (mem["L2"][i], ents_a[e], x.appln_id) in zset
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
    key = [codecol, "level", "measure", "tech_node", "year"]
    D = D.groupby(key, as_index=False).delta.sum()
    lin = df[df.measure.str.endswith("_lin")].merge(D, on=key, how="outer").fillna({"value": 0.0, "delta": 0.0})
    lin["old"] = lin.value; lin["value"] = lin.value + lin.delta
    neg = lin[lin.value < -1e-6 * np.maximum(1.0, lin.old.abs())]
    g_, _ = clean(neg.assign(value=1.0), side)
    n_bad = int((g_.value != 0).sum())
    lin = lin[lin.value > 1e-9 * np.maximum(1.0, lin.old.abs())].drop(columns=["delta", "old"])
    ash = lin.assign(measure=lin.measure.str.replace("_lin", "", regex=False), value=np.arcsinh(lin.value))
    out = pd.concat([ash, lin], ignore_index=True)[df.columns]
    rep = {"patents_affected": int(len(aff)), "moved": int((aff.patent_verdict == "move").sum()),
           "not_digital": int((aff.patent_verdict == "not_a_digital_technology").sum()), "zero_links": int(len(zset)),
           "negative_cells_offtheme": int(len(neg)) - n_bad, "negative_cells_outside_offtheme_clipped": n_bad}
    return out, rep


def main():
    titles, lv, dedup, _ = TX.final_levels()
    say("loading the build_final pages + Cloud computing (patent embeddings)")
    sh2, cfg = RV.final_shared()
    t0_, lv0 = TX.production_levels()
    assert all((np.asarray(sh2["levels"][k], dtype=object) == lv0[k]).all() for k in lv0), "review.final_shared levels differ"
    drows = [i for i, t in enumerate(titles) if t in dedup]
    tags = sh2["tags"][~sh2["tags"].page_row.isin(drows)].reset_index(drop=True)
    sh3 = dict(sh2, tags=tags, levels=lv)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "exposure").mkdir(exist_ok=True)
    lev = pd.DataFrame({"page_row": np.arange(len(titles)), "page_title": titles, **lv})
    lev.to_csv(OUT / "levels.csv", index=False)
    lev[lev.L2 != BF.OUT_NODE][["page_row", "page_title", "root", "L1", "L2"]].to_parquet(OUT / "page_meta.parquet", index=False)
    say(f"final levels: L2 {len(set(lv['L2']) - {BF.OUT_NODE})}, L1 {len(set(lv['L1']) - {BF.OUT_NODE})}, root "
        f"{len(set(lv['root']) - {BF.OUT_NODE})}; duplicate pages out of the matches: {len(drows)}")
    Va, pv = verdicts(sh3, lv, cfg["tb"])
    rep = {}
    for side, fn in (("occ", "occupation"), ("ind", "industry")):
        say(f"build {fn}")
        df = B._build_side(cfg, side, sh3, yearly=True)
        df = df[df.tech_node != BF.OUT_NODE].reset_index(drop=True)
        df, z1 = clean(df, side)
        say(f"review {fn}")
        df, rr = apply_review_final(df, side, cfg, sh3, Va, pv)
        df, z2 = clean(df, side)
        df.to_parquet(OUT / "exposure" / f"{fn}.parquet", index=False)
        rep[fn] = dict(rr, rows=len(df), cells_zeroed_before_review=z1, cells_zeroed_after_review=z2)
        say(f"  {fn}: {rep[fn]}")
    say("membership (one node per patent)")
    mem, _ = RV._membership(sh3, lv, np.arange(sh3["nPat"]), BF.OUT_NODE, cfg["tb"])
    mo = pd.DataFrame({"appln_id": np.asarray(sh3["pids"]).astype(np.int64), "year": sh3["year"], **mem})
    act = pv[pv.patent_verdict != "belongs"].reset_index(drop=True)
    rows = pd.Series(np.arange(len(mo)), index=mo.appln_id.to_numpy()).loc[act.appln_id.to_numpy()].to_numpy()
    up = RV._l2_parents(lv, BF.OUT_NODE)
    nd = (act.patent_verdict == "not_a_digital_technology").to_numpy(); mv = (act.patent_verdict == "move").to_numpy()
    for k in RV.LEVELS:
        v = mo[k].to_numpy().copy(); has = v[rows] != ""
        v[rows[nd & has]] = ""
        dst = act.move_target.to_numpy() if k == "L2" else up.reindex(act.move_target.to_numpy())[k].to_numpy()
        v[rows[mv & has]] = dst[mv & has]; mo[k] = v
    mo = mo[(mo[list(RV.LEVELS)] != "").any(axis=1)].reset_index(drop=True)
    mo.to_parquet(OUT / "membership.parquet", index=False)
    (OUT / "README.json").write_text(json.dumps({
        "build": "final digital taxonomy (user decision 2026-10-07)", "script": "p3_exposure/threshold_experiment/build_core.py",
        "taxonomy": "p3_exposure/threshold_experiment/taxonomy_core.py", "pages_live": int((lv["L2"] != BF.OUT_NODE).sum()),
        "duplicate_pages_out_of_matches": len(drows), "name_map": TX.NAME_MAP, "removed_nodes": sorted(TX.REMOVED_NODES),
        "report": rep, "membership_rows": len(mo)}, indent=1))
    say("done")


if __name__ == "__main__":
    main()
