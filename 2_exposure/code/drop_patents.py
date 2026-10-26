"""Drop individual patents from an existing digital exposure build (patent-level match check, 2026-10-01).

The patents listed in a CSV (column appln_id) lose their contribution in the node of their best page at every level
(L2, L1, root), with add_page.py's copy of the build formula. A patent's contribution depends only on its own column of
the task-patent matrix and on its own page (the task-page gate), so it is the same at the three levels and is computed
once. Node membership follows the cleaned build: the page-handling plan's MOVE / MOVE-NEW targets, pages it DROPs carry
no exposure, and the patents moved by add_page.py sit on 'Cloud computing' (their gate uses that page's embedding).

Two checks guard the result: (1) before anything is dropped, a set of test nodes recomputed from scratch must equal the
source build, which validates the node mapping; (2) after the drop, the same nodes recomputed from scratch on their
remaining patents must equal the result. Off-theme zero gating is re-applied; untouched cells are asserted identical.

    cd p3_exposure/threshold_experiment
    python -u drop_patents.py <drop_list.csv> <source build folder> <output build folder>
"""
import json, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from pathlib import Path
import numpy as np, pandas as pd
import core as C
import cleaning as CL
from build import YEARS
from add_page import CFG_KEY, MEASURES, contrib, to_entities, page_text, embed

PLAN = Path(__file__).resolve().parents[2] / "notes" / "validation" / "finetune" / "page_handling_plan.csv"
CLOUD = C.TRI / "pb_mpnet_finetuned_changes_cloud_yearly" / "moved_patents.csv"
CLOUD_PAGE, CLOUD_L2 = "Cloud computing", "Storage, Virtualization & File Systems"
TEST_NODES = ["System Security & Malware Defense", "Service & Domestic Robotics", "Medical & Laboratory Robotics",
              "Audio Measurement Instruments", CLOUD_L2, "Quantum computation theory", "Cellular Networks & Mobile Infrastructure"]
CHUNK = 20000


def hierarchy(pm, plan, src_levels):
    """page -> L2 node of the cleaned build; L2 node -> (L1, root)."""
    title = pm.page_title.astype(str).to_numpy()
    moves = plan[plan.action.isin(["MOVE", "MOVE-NEW"])].set_index("page").target.to_dict()
    l2_of_page = np.array([moves.get(t, l2) for t, l2 in zip(title, pm.L2.astype(str))], dtype=object)
    up = {}
    base = pm.assign(L2=pm.L2.astype(str), L1=pm.L1.astype(str), root=pm.root.astype(str))
    for node, g in base.groupby("L2"):
        up[node] = (g.L1.mode().iloc[0], g.root.mode().iloc[0])
    for page, tgt in moves.items():
        if tgt not in up:                                    # MOVE-NEW node: its own L1 in the build, root of its old node
            old = base[base.page_title.astype(str) == page]
            root = old.root.iloc[0]
            l1 = tgt if tgt in src_levels.get("L1", set()) else old.L1.iloc[0]
            up[tgt] = (l1, root)
    up[CLOUD_L2] = up.get(CLOUD_L2, ("System Software & Security", "Computing Hardware & Systems"))
    return l2_of_page, up


def node_values(ctx, cols, Cc, years, ents):
    """{measure_vt: (nY, nE)} summed over `cols` (sorted), processed in chunks."""
    out = {vt: np.zeros((len(YEARS), len(ents))) for _, vt in MEASURES}
    for i in range(0, len(cols), CHUNK):
        sub = cols[i:i + CHUNK]
        M, base, s2e, nE = ctx["M"], ctx["base"], ctx["s2e"], len(ents)
        Msub = np.asarray(M[:, sub])
        pos = np.full(ctx["nPat"], -1, np.int64); pos[sub] = np.arange(len(sub))
        c2 = dict(cfg=ctx["cfg"], Msub=Msub, pos=pos, keepU=C.keep_single(Msub, base, s2e, nE, K=10), s2e=s2e, base=base,
                  ptask=ctx["ptask"], ppage=None, cit=ctx["cit"], nY=len(YEARS), pad=ctx["pad"])
        v = to_entities(c2, contrib(c2, sub, Cc[:, i:i + CHUNK], years))
        for _, vt in MEASURES:
            out[vt] += v[vt]
    return out


def main(drop_csv, src_name, out_name):
    SRC, OUT = C.TRI / src_name, C.TRI / out_name
    cfg = C.CFG[CFG_KEY]
    sh = C.load_shared(C.TAGFILE[cfg["family"]], with_cit=True)
    pm = pd.read_parquet(C.PBAN / "page_meta_patentsberta.parquet")
    plan = pd.read_csv(PLAN)
    src_ind = pd.read_parquet(SRC / "exposure" / "industry.parquet", columns=["level", "tech_node"]).drop_duplicates()
    src_levels = {lv: set(g.tech_node) for lv, g in src_ind.groupby("level")}
    l2_of_page, up = hierarchy(pm, plan, src_levels)
    title = pm.page_title.astype(str).to_numpy()
    dropped_pages = set(plan[plan.action == "DROP"].page)
    tags = sh["tags"][sh["tags"].sim >= cfg["tb"]]
    best = tags.loc[tags.groupby("patent_col")["sim"].idxmax()]
    argmax = np.full(sh["nPat"], -1, np.int64); argmax[best.patent_col.to_numpy()] = best.page_row.to_numpy()
    pids = np.asarray(sh["pids"]).astype(np.int64)
    cloud = np.isin(pids, pd.read_csv(CLOUD).appln_id.astype(np.int64).to_numpy())
    cloud_cols = np.where(cloud)[0]
    # membership exactly as the build counts it: at each level, a node's page set (plan MOVEs applied, plan DROP pages
    # removed) and node_members' 1to1 rule (a patent counts where its global best page is the node's best page for it;
    # an exact tie between two pages of the node leaves it out). Cloud-moved patents count in the cloud page's node.
    page_level = {"L2": l2_of_page,
                  "L1": np.array([up[n][0] for n in l2_of_page], dtype=object),
                  "root": np.array([up[n][1] for n in l2_of_page], dtype=object)}
    live_page = ~np.isin(title, list(dropped_pages))
    scope = {lv: s for lv, s in src_levels.items()}            # nodes that carry exposure in the source build
    cloud_up = {"L2": CLOUD_L2, "L1": up[CLOUD_L2][0], "root": up[CLOUD_L2][1]}
    _mcache = {}

    def counted(level, nm):
        if (level, nm) not in _mcache:
            G = np.where((page_level[level] == nm) & live_page)[0]
            r = C.node_members(tags, G, argmax) if len(G) else None
            cols = np.array([], np.int64) if r is None else r[0][r[2]]
            cols = cols[~cloud[cols]]
            if cloud_up[level] == nm:
                cols = np.concatenate([cols, cloud_cols])
            _mcache[(level, nm)] = np.sort(np.unique(cols))
        return _mcache[(level, nm)]

    want = set(pd.read_csv(drop_csv).appln_id.astype(np.int64))
    listed = np.where(np.isin(pids, list(want)))[0]
    in_build = np.unique(np.concatenate([counted("L2", nm) for nm in sorted(scope["L2"])]))
    drop = np.intersect1d(listed, in_build)
    print(f"patents to drop: {len(want):,} listed, {len(drop):,} counted in an L2 node of the source build", flush=True)
    e_cloud = embed([page_text(CLOUD_PAGE)])
    years = sh["year"]

    def gate_matrix(ptask, cols):
        pg = argmax[cols].clip(0)
        P = sh["ppage"][pg]
        if cloud[cols].any():
            P = P.copy(); P[cloud[cols]] = e_cloud[0]
        return ptask @ P.T                                     # nSent x len(cols): each patent's own page

    out = {}
    for side, fn in (("occ", "occupation"), ("ind", "industry")):
        codes, ents, ec, s2e, codecol = C.entity_meta(cfg["family"], side)
        M = np.load(C.a_dir(cfg["a_embed"]) / C._indf(side, C.A_FILES[side]), mmap_mode="r")
        base = M.mean(1).astype(np.float32)
        ptask = C.load_ptask(side)
        ctx = dict(cfg=cfg, M=M, base=base, s2e=s2e, ptask=ptask, cit=sh["cit_rel"], nPat=sh["nPat"], pad=C.pad_index(s2e, len(ents)))
        df = pd.read_parquet(SRC / "exposure" / f"{fn}.parquet")
        key = [codecol, "level", "measure", "tech_node", "year"]
        ents_a = np.asarray(ents); yrs_a = np.asarray(YEARS)

        def frame(vals, level, node):
            rows = []
            for mn, vt in MEASURES:
                yy, cc = np.nonzero(vals[vt])
                rows.append(pd.DataFrame({codecol: ents_a[cc], "level": level, "measure": mn + "_lin", "tech_node": node,
                                          "year": yrs_a[yy], "value": vals[vt][yy, cc]}))
            return pd.concat(rows)

        def compare(f_build, vals, node, what, level="L2"):
            mine = frame(vals, level, node)
            mine, _ = CL.apply_digital(mine, side)
            worst = 0.0
            for mn, _ in MEASURES:
                a = f_build[(f_build.level == level) & (f_build.tech_node == node) & (f_build.measure == mn + "_lin")].set_index([codecol, "year"]).value
                b = mine[mine.measure == mn + "_lin"].set_index([codecol, "year"]).value
                both = pd.concat({"file": a, "mine": b}, axis=1).fillna(0.0)
                rel = (both.file - both.mine).abs().max() / max(both.file.abs().max(), 1e-12) if len(both) else 0.0
                worst = max(worst, rel)
            assert worst < 1e-4, f"[{side}] {what}: '{node}' recomputed from scratch != build ({worst:.2e}); stop"
            return worst

        # (1) mapping check on the source build (test L2 nodes, plus roots/L1 built from the mapping)
        def members(level, nm, extra=None):
            cols = counted(level, nm)
            return cols if extra is None else cols[extra[cols]]
        UPPER = [("root", "Quantum Computing"), ("L1", up["Service & Domestic Robotics"][0]), ("root", "Artificial Intelligence")]
        w1 = 0.0
        for node in TEST_NODES:
            cols = members("L2", node)
            w1 = max(w1, compare(df, node_values(ctx, cols, gate_matrix(ptask, cols), years, ents), node, "source"))
        for level, nm in UPPER:
            cols = members(level, nm)
            w1 = max(w1, compare(df, node_values(ctx, cols, gate_matrix(ptask, cols), years, ents), nm, "source", level))
        print(f"  [{side}] mapping check: {len(TEST_NODES)} test nodes recomputed from scratch = source build (worst {w1:.1e})", flush=True)

        # (2) contributions of the dropped patents in every node that counts them, level by level
        dl = []
        for level in ("L2", "L1", "root"):
            for nm in sorted(scope.get(level, set())):
                cols = np.intersect1d(counted(level, nm), drop)
                if len(cols):
                    v = node_values(ctx, cols, gate_matrix(ptask, cols), years, ents)
                    f = frame(v, level, nm); f["value"] = -f["value"]; dl.append(f)
        D = pd.concat(dl).groupby(key, as_index=False).value.sum().rename(columns={"value": "delta"})
        lin = df[df.measure.str.endswith("_lin")].merge(D, on=key, how="left")
        was = lin.value.copy()
        lin["value"] = was + lin.delta.fillna(0.0)
        tol = 1e-5 * was.abs() + 1e-9
        bad = lin.value < -tol
        assert not bad.any(), f"negative cells after subtraction ({int(bad.sum())}): {lin[bad].head()}"
        lin.loc[lin.value.abs() <= tol, "value"] = 0.0
        orphan = D.merge(df[key], on=key, how="left", indicator=True)
        orphan = orphan[(orphan._merge != "both") & (orphan.delta.abs() > 1e-9)]
        assert orphan.empty, f"subtracted cells absent from the source build: {orphan.head()}"

        # (3) arcsinh rows of touched cells, gating, everything else identical
        touched = lin[lin.delta.notna()].drop(columns="delta").copy()
        asinh = touched.assign(measure=touched.measure.str.replace("_lin", "", regex=False), value=np.arcsinh(touched.value))
        tk = pd.concat([touched, asinh])[key].drop_duplicates()
        rest = df.merge(tk, on=key, how="left", indicator=True); rest = rest[rest._merge == "left_only"].drop(columns="_merge")
        new = pd.concat([rest, touched, asinh], ignore_index=True)
        new, z = CL.apply_digital(new, side)
        chk = rest.merge(new, on=key, suffixes=("", "_new"))
        assert len(chk) == len(rest) and (chk.value == chk.value_new).all(), "untouched cells changed"
        assert len(new) == len(df), "row count changed"

        # (4) result check: test nodes recomputed on their remaining patents
        w2 = 0.0
        notdrop = ~np.isin(np.arange(sh["nPat"]), drop)
        for node in TEST_NODES:
            cols = members("L2", node, notdrop)
            w2 = max(w2, compare(new, node_values(ctx, cols, gate_matrix(ptask, cols), years, ents), node, "result"))
        for level, nm in UPPER:
            cols = members(level, nm, notdrop)
            w2 = max(w2, compare(new, node_values(ctx, cols, gate_matrix(ptask, cols), years, ents), nm, "result", level))
        print(f"  [{side}] result check: {len(TEST_NODES)} test nodes recomputed on remaining patents = result (worst {w2:.1e}); "
              f"cells changed {len(tk):,}; zero-gated {z:,}", flush=True)
        out[side] = (fn, new[df.columns.tolist()], df)

    d = OUT / "exposure"; d.mkdir(parents=True, exist_ok=True)
    rep = []
    for side, (fn, new, old) in out.items():
        new.sort_values([new.columns[0], "level", "measure", "tech_node", "year"]).to_parquet(d / f"{fn}.parquet", index=False)
        f = lambda x: x[(x.measure == "pb_plain_mean_lin") & (x.level == "L2") & x.year.between(2009, 2023)].groupby("tech_node").value.sum()
        t = pd.concat({"before": f(old), "after": f(new)}, axis=1).fillna(0.0); t["change"] = t.after - t.before
        t["change_%"] = 100 * t.change / t.before.where(t.before > 0)
        rep.append(t.assign(side=fn))
    R = pd.concat(rep); R.to_csv(OUT / "node_changes_2009_2023.csv")
    l2_drop = {int(c): nm for nm in sorted(scope["L2"]) for c in np.intersect1d(counted("L2", nm), drop)}
    pd.DataFrame({"appln_id": pids[drop], "node": [l2_drop.get(int(c), "") for c in drop]}).to_csv(OUT / "dropped_patents.csv", index=False)
    (OUT / "README.json").write_text(json.dumps({"source": str(SRC), "drop_list": str(drop_csv), "dropped_patents": int(len(drop)),
                                                 "script": "p3_exposure/threshold_experiment/drop_patents.py"}, indent=1))
    print(R.groupby("side")[["before", "after"]].sum().round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
