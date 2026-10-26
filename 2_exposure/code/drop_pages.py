"""Drop taxonomy pages from an existing digital exposure build, in place of a full rebuild (page-handling DROP: a dropped
page carries no exposure, as in notes/validation/finetune/page_handling_plan.csv).

Used 2026-10-01 for six word-collision pages of the Quantum Computing root (user decision): their patents are ordinary
cloud, data-centre, information-processing and imaging patents that share words with the page titles.

The patents whose best page is a dropped page lose their contribution in every node that counts them (L2, L1, root),
computed with add_page.py's copy of the build formula; the off-theme zero gating is re-applied; every other cell is kept.
Before writing, each affected L2 node is recomputed from scratch on its remaining patents and must equal the result.

    cd p3_exposure/threshold_experiment
    python -u drop_pages.py
"""
import json, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from pathlib import Path
import numpy as np, pandas as pd
import core as C
import cleaning as CL
from build import YEARS
from add_page import CFG_KEY, MEASURES, node_C, contrib, to_entities

SRC = C.TRI / "pb_mpnet_finetuned_changes_cloud_offtheme2_yearly"     # the papers' digital build before this step
OUT = C.TRI / "pb_mpnet_finetuned_changes_cloud_offtheme2_qdrop_yearly"
CLOUD_MOVED = C.TRI / "pb_mpnet_finetuned_changes_cloud_yearly" / "moved_patents.csv"   # already out of their old pages
DROP = ["Cloud-based quantum computing", "Algorithmic cooling", "Quantum state discrimination", "Quantum tomography",
        "Neural network quantum states", "Pulse programming"]
PLAN = Path(__file__).resolve().parents[2] / "notes" / "validation" / "finetune" / "page_handling_plan.csv"


def main():
    cfg = C.CFG[CFG_KEY]
    sh = C.load_shared(C.TAGFILE[cfg["family"]], with_cit=True)
    pm = pd.read_parquet(C.PBAN / "page_meta_patentsberta.parquet")
    title = pm.page_title.astype(str).to_numpy()
    lv = {k: pm[k].astype(str).to_numpy() for k in ("L2", "L1", "root")}
    rows = np.where(np.isin(title, DROP))[0]
    assert len(rows) == len(DROP), f"pages not found: {set(DROP) - set(title[rows])}"
    plan = pd.read_csv(PLAN)
    assert not plan.page.isin(DROP).any(), "a page to drop is already handled by the page-handling plan"
    nodes_l2 = set(lv["L2"][rows])
    assert not plan.target.isin(nodes_l2).any(), "the plan moves pages into an affected node; membership would differ"
    tags = sh["tags"][sh["tags"].sim >= cfg["tb"]]
    best = tags.loc[tags.groupby("patent_col")["sim"].idxmax()]
    argmax = np.full(sh["nPat"], -1, np.int64); argmax[best.patent_col.to_numpy()] = best.page_row.to_numpy()
    pids = np.asarray(sh["pids"]).astype(np.int64)
    moved = set(pd.read_csv(CLOUD_MOVED).appln_id.astype(np.int64))
    dropped = np.where(np.isin(argmax, rows) & ~np.isin(pids, list(moved)))[0]
    old_page = argmax[dropped]
    print(f"patents whose best page is a dropped page (cloud-moved excluded): {len(dropped):,}", flush=True)
    print(pd.Series(title[old_page]).value_counts().to_string(), flush=True)

    # membership of the dropped patents in each node at each level, exactly as the build counts it (1to1 is1)
    tdr = tags[tags.patent_col.isin(dropped)]
    is1_at = {}
    for k in ("L2", "L1", "root"):
        okk = np.zeros(len(dropped), bool)
        for node in sorted(set(lv[k][old_page])):
            P, gl, i1 = C.node_members(tdr, np.where(lv[k] == node)[0], argmax); d = dict(zip(P.tolist(), i1.tolist()))
            idx = np.where(lv[k][old_page] == node)[0]
            okk[idx] = [d.get(int(dropped[j]), False) for j in idx]
        is1_at[k] = okk
        print(f"dropped patents the build does not count in their {k} node (exact page tie): {int((~okk).sum())}", flush=True)

    out = {}
    for side, fn in (("occ", "occupation"), ("ind", "industry")):
        codes, ents, ec, s2e, codecol = C.entity_meta(cfg["family"], side)
        M = np.load(C.a_dir(cfg["a_embed"]) / C._indf(side, C.A_FILES[side]), mmap_mode="r")
        base = M.mean(1).astype(np.float32)
        members = {node: C.node_members(tags, np.where(lv["L2"] == node)[0], argmax) for node in sorted(nodes_l2)}
        U = np.unique(np.concatenate([dropped] + [m[0] for m in members.values()]))
        Msub = np.asarray(M[:, U]); pos = np.full(sh["nPat"], -1, np.int64); pos[U] = np.arange(len(U))
        ctx = dict(cfg=cfg, Msub=Msub, pos=pos, keepU=C.keep_single(Msub, base, s2e, len(ents), K=10), s2e=s2e, base=base,
                   ptask=C.load_ptask(side), ppage=sh["ppage"], cit=sh["cit_rel"], nY=len(YEARS), pad=C.pad_index(s2e, len(ents)))
        df = pd.read_parquet(SRC / "exposure" / f"{fn}.parquet")
        key = [codecol, "level", "measure", "tech_node", "year"]

        # 1) the subtraction: old contribution of the dropped patents out of every node that counts them
        dl = []
        for k in ("L2", "L1", "root"):
            sk = is1_at[k]; dp, op = dropped[sk], old_page[sk]; o = np.argsort(dp); dp, op = dp[o], op[o]
            for node in sorted(set(lv[k][op])):
                sel = lv[k][op] == node
                vv = to_entities(ctx, contrib(ctx, dp[sel], node_C(ctx, op[sel], np.where(lv[k] == node)[0]), sh["year"]))
                for mn, vt in MEASURES:
                    yy, cc = np.nonzero(vv[vt])
                    dl.append(pd.DataFrame({codecol: np.asarray(ents)[cc], "level": k, "measure": mn + "_lin", "tech_node": node,
                                            "year": np.asarray(YEARS)[yy], "delta": -vv[vt][yy, cc]}))
        D = pd.concat(dl).groupby(key, as_index=False).delta.sum()
        lin = df[df.measure.str.endswith("_lin")].merge(D, on=key, how="left")
        was = lin.value.copy()
        lin["value"] = was + lin.delta.fillna(0.0)
        tol = 1e-5 * was.abs() + 1e-12
        assert (lin.value >= -tol).all(), f"negative cells after subtraction: {lin[lin.value < -tol].head()}"
        lin.loc[lin.value.abs() <= tol, "value"] = 0.0
        missing = D.merge(df[key], on=key, how="left", indicator=True)
        assert (missing._merge == "both").all() or (missing[missing._merge != "both"].delta.abs() < 1e-9).all(), \
            "a subtracted cell is absent from the source build"

        # 2) validation: each affected L2 node recomputed from scratch on its remaining patents equals the result
        worst = 0.0
        for node in sorted(nodes_l2):
            G = np.where(lv["L2"] == node)[0]; Pcols, gloc, is1 = members[node]
            keep = is1 & ~np.isin(Pcols, dropped) & ~np.isin(pids[Pcols], list(moved))
            Pc, gp = Pcols[keep], G[gloc[keep]]; o = np.argsort(Pc); Pc, gp = Pc[o], gp[o]
            v = to_entities(ctx, contrib(ctx, Pc, node_C(ctx, gp, G), sh["year"]))
            for mn, vt in MEASURES:
                f = lin[(lin.level == "L2") & (lin.tech_node == node) & (lin.measure == mn + "_lin")].set_index([codecol, "year"]).value
                yy, cc = np.nonzero(v[vt] > 0)
                mine = pd.Series(v[vt][yy, cc], index=pd.MultiIndex.from_arrays([np.asarray(ents)[cc], np.asarray(YEARS)[yy]]))
                both = pd.concat({"file": f, "mine": mine}, axis=1).fillna(0.0)
                rel = (both.file - both.mine).abs().max() / max(both.file.abs().max(), 1e-12) if len(both) else 0.0
                worst = max(worst, rel)
                assert rel < 1e-4, f"'{node}' ({mn}, {side}) after the drop != recomputed from scratch ({rel:.2e}); stop"
        print(f"  [{side}] {len(nodes_l2)} affected L2 nodes = recomputed from scratch on their remaining patents "
              f"(worst max relative difference {worst:.1e})", flush=True)

        # 3) arcsinh rows of the touched cells, zero gating, everything else identical
        touched = lin[lin.delta.notna()].drop(columns="delta").copy()
        asinh = touched.assign(measure=touched.measure.str.replace("_lin", "", regex=False), value=np.arcsinh(touched.value))
        tk = pd.concat([touched, asinh])[key].drop_duplicates()
        rest = df.merge(tk, on=key, how="left", indicator=True); rest = rest[rest._merge == "left_only"].drop(columns="_merge")
        new = pd.concat([rest, touched, asinh], ignore_index=True)
        new, z = CL.apply_digital(new, side)
        chk = rest.merge(new, on=key, suffixes=("", "_new"))
        assert len(chk) == len(rest) and (chk.value == chk.value_new).all(), "untouched cells changed"
        assert len(new) == len(df), "row count changed"
        print(f"  [{side}] cells changed {len(tk):,}; zero-gated cells {z:,}", flush=True)
        out[side] = (fn, new[df.columns.tolist()], df)

    d = OUT / "exposure"; d.mkdir(parents=True, exist_ok=True)
    rep = []
    for side, (fn, new, old) in out.items():
        new.sort_values([new.columns[0], "level", "measure", "tech_node", "year"]).to_parquet(d / f"{fn}.parquet", index=False)
        f = lambda x: x[(x.measure == "pb_plain_mean_lin") & (x.level == "L2") & x.year.between(2009, 2023)].groupby("tech_node").value.sum()
        t = pd.concat({"before": f(old), "after": f(new)}, axis=1).fillna(0.0); t["change"] = t.after - t.before
        t["change_%"] = 100 * t.change / t.before.where(t.before > 0)
        rep.append(t[t.change.abs() > 1e-9].assign(side=fn))
    R = pd.concat(rep); R.to_csv(OUT / "node_changes_2009_2023.csv")
    pd.DataFrame({"appln_id": pids[dropped], "page": title[old_page], "L2": lv["L2"][old_page]}).to_csv(OUT / "dropped_patents.csv", index=False)
    (OUT / "README.json").write_text(json.dumps({"source": str(SRC), "dropped_pages": DROP, "dropped_patents": int(len(dropped)),
                                                 "script": "p3_exposure/threshold_experiment/drop_pages.py"}, indent=1))
    print("\nL2 changes, linear value summed over filings 2009-2023:")
    print(R.round(3).to_string())


if __name__ == "__main__":
    main()
