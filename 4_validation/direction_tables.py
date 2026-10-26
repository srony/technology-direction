"""Stability and structure of the panel's ratings, from the released grid and ratings alone.

Writes to the output folder the rating distributions, the triplet-level scatter, the persona means, the technology
ranking by substitution share, and the reliability table with the split-half correlation, the rating noise, the share
of the variance of the direction explained by technology, occupation and industry fixed effects, and the pairs that
change sign across industries. The numbers are those of the companion paper's direction sections.

    python direction_tables.py --out-dir direction_tables
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "3_direction"))
import contents as C

KEY = C.KEY
DOMS = ("digital", "green")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default="direction_tables")
    out = Path(ap.parse_args().out_dir); out.mkdir(exist_ok=True)
    J = {}
    g = pd.read_csv(HERE.parent / "3_direction" / "grid.csv", dtype={"isco3": str, "section": str})
    d = C.usable_ratings()
    per = d.groupby(KEY).size()
    J["grid"] = dict(triplets=len(g), by_domain=g.domain.value_counts().to_dict(), technologies=g.groupby("domain").tech_node.nunique().to_dict())
    J["calls"] = dict(usable=len(d), answers_per_triplet={int(k): int(v) for k, v in per.value_counts().items()}, triplets_rated=int(len(per)))

    # rating levels and the zero category
    lev = d.groupby(["domain", "construct"]).rating.agg(["mean", "std", "size"])
    J["levels"] = {f"{a}|{b}": dict(mean=float(r["mean"]), sd=float(r["std"]), n=int(r["size"])) for (a, b), r in lev.iterrows()}
    J["zero_share_pct"] = {f"{a}|{b}": float(v) for (a, b), v in (d.assign(z=d.rating == 0).groupby(["domain", "construct"]).z.mean() * 100).items()}

    # triplet level, raw ratings
    tm = d.groupby(KEY + ["construct"]).rating.mean().unstack("construct")
    J["triplet"] = {}
    for dom in DOMS:
        x = tm.xs(dom, level="domain")
        J["triplet"][dom] = dict(n=len(x), sub_above_comp=int((x.substitutability > x.complementarity).sum()),
                                 corr=float(x.substitutability.corr(x.complementarity)),
                                 mean_gap=float((x.complementarity - x.substitutability).mean()))

    # personas
    pmean = d.pivot_table(index=["domain", "construct"], columns="persona", values="rating", aggfunc="mean")
    J["persona_means"] = {f"{a}|{b}": {k: float(v) for k, v in r.items()} for (a, b), r in pmean.iterrows()}
    vd = []
    for (dom, c), x in d.groupby(["domain", "construct"]):
        y = x.rating; tot = ((y - y.mean()) ** 2).sum()
        m_t = x.groupby(KEY).rating.transform("mean"); m_tp = x.groupby(KEY + ["persona"]).rating.transform("mean")
        vd.append(dict(domain=dom, construct=c, triplet=float(((m_t - y.mean()) ** 2).sum() / tot * 100),
                       persona=float(((m_tp - m_t) ** 2).sum() / tot * 100), draw=float(((y - m_tp) ** 2).sum() / tot * 100)))
    J["variance_parts_pct"] = vd

    # fractions and the technology ranking
    fr = C.fractions(d)
    J["fractions"] = {}
    for dom, x in fr.groupby("domain"):
        J["fractions"][dom] = dict(sigma_mean=float(x.sigma.mean()), sigma_zero_pct=float((x.sigma == 0).mean() * 100),
                                   kappa_mean=float(x.kappa.mean()), kappa_zero_pct=float((x.kappa == 0).mean() * 100),
                                   spearman=float(x.sigma.corr(x.kappa, method="spearman")), sigma_gt_kappa_pct=float((x.sigma > x.kappa).mean() * 100))
    nlab = d.drop_duplicates(["domain", "tech_node"]).set_index(["domain", "tech_node"]).node_label
    node = fr.groupby(["domain", "tech_node"]).agg(sigma=("sigma", "mean"), kappa=("kappa", "mean"), n=("sigma", "size"))
    node["share"] = node.sigma / (node.sigma + node.kappa)
    node["label"] = [nlab.get(i, i[1]) for i in node.index]
    node.reset_index().sort_values(["domain", "share"], ascending=[True, False]).to_csv(out / "technology_ranking.csv", index=False)

    # reliability and the determinants of direction
    dd = d.copy(); dd["f"] = dd.rating.round().astype(int).map(C.FRACTION)
    rng = np.random.default_rng(20261007); dd["half"] = rng.integers(0, 2, len(dd))
    full = dd.groupby(KEY + ["construct"]).f.mean().unstack("construct"); full["d"] = full.complementarity - full.substitutability
    halves = dd.groupby(KEY + ["construct", "half"]).f.mean().unstack(["construct", "half"])
    dh = pd.DataFrame({h: halves[("complementarity", h)] - halves[("substitutability", h)] for h in (0, 1)}).dropna()
    TT = full.reset_index().merge(dd.drop_duplicates(KEY)[KEY + ["occ_title", "sec_title"]], on=KEY)
    R = {}
    for dom in DOMS:
        t = TT[TT.domain == dom].copy(); h = dh.loc[dom]
        r = float(np.corrcoef(h[0], h[1])[0, 1]); sb = 2 * r / (1 + r); noise = float((h[0] - h[1]).std() / 2)
        t["no"] = t.tech_node + "|" + t.isco3; t["ns"] = t.tech_node + "|" + t.section; t["os"] = t.isco3 + "|" + t.section
        tv = float(t.d.var(ddof=0)); r2 = {}
        for name, fe in (("technology", "tech_node"), ("occupation", "isco3"), ("industry", "section"),
                         ("additive", "tech_node + isco3 + section"), ("technology_x_occupation", "no"),
                         ("technology_x_industry", "ns"), ("occupation_x_industry", "os")):
            r2[name] = 1 - float(np.var(pf.feols(f"d ~ 1 | {fe}", t).resid(), ddof=0)) / tv
        gp = t.groupby("no").filter(lambda x: len(x) >= 2); within = gp.groupby("no").d.agg(["std", "min", "max", "size"])
        flips = float(((within["max"] > 2 * noise) & (within["min"] < -2 * noise)).mean() * 100)
        R[dom] = dict(triplets=len(t), mean_direction=float(t.d.mean()), share_positive=float((t.d > 0).mean()), split_half_r=r,
                      spearman_brown=sb, noise_sd=noise, **{f"r2_{k}": v for k, v in r2.items()}, pairs_two_or_more_industries=int(len(within)),
                      within_pair_sd=float(within["std"].mean()), sign_flip_pct=flips)
    J["reliability"] = R
    pd.DataFrame(R).to_csv(out / "reliability.csv")

    # figures
    COL = {"substitutability": "#c1443c", "complementarity": "#2f6f9f"}; CLAB = {"substitutability": "substitution", "complementarity": "complementarity"}
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
    for i, dom in enumerate(DOMS):
        for j, c in enumerate(("substitutability", "complementarity")):
            s = d[(d.domain == dom) & (d.construct == c)].rating.value_counts(normalize=True).sort_index() * 100
            a = ax[i, j]; a.bar(s.index, s.values, color=COL[c], width=.7); a.set_title(f"{dom.capitalize()}, {CLAB[c]}", fontsize=10)
            a.set_xticks(range(6)); a.set_xlabel("rating (0 = do not know)" if i == 1 else ""); a.set_ylabel("% of answers" if j == 0 else "")
            a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "rating_distributions.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    for i, dom in enumerate(DOMS):
        x = tm.xs(dom, level="domain")
        ax[i].scatter(x.complementarity, x.substitutability, s=7, alpha=.25, color="#2f4858", edgecolors="none")
        ax[i].plot([0, 5], [0, 5], color="#c1443c", lw=1); ax[i].set_xlim(0, 5); ax[i].set_ylim(0, 5)
        ax[i].set_xlabel("complementarity"); ax[i].set_ylabel("substitution" if i == 0 else "")
        ax[i].set_title(f"{dom.capitalize()} ({len(x):,} triplets)", fontsize=10); ax[i].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "triplet_scatter.png", dpi=200); plt.close(fig)
    json.dump(J, open(out / "direction_numbers.json", "w"), indent=1, default=str)
    print(json.dumps({k: J[k] for k in ("grid", "calls", "levels", "triplet", "fractions")}, indent=1, default=str))
    print(pd.DataFrame(R).round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
