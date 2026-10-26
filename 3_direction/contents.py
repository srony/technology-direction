"""Substitution and complementarity content of occupations and industries from the panel's ratings and the exposure
release, for any window of patent filing years.

Each rating maps to the fraction of the reached task content its scale label names, 3 to one half, 4 to three
quarters, 5 to all, and 0, 1 and 2 to none. The mean over the fifteen ratings of a triplet gives sigma (substitution)
and kappa (complementarity). For a four-digit occupation e in group o(e), with x_en its exposure to technology n over
the window and N the number of base technologies of the domain,

    Sigma_e = (1/N) sum_n sum_i w_{o(e) i} x_en sigma_{n, o(e), i},    Gamma_e likewise with kappa,

where w_{oi} is the pooled employment share of section i among the group's workers. Cells outside the grid contribute
nothing. The industry side mirrors this with the section's occupation shares. Zero content is content: every code of
the exposure release is kept.

    python contents.py --window 2011 2023 --out-dir contents_2011_2023
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent / "2_exposure" / "release"
sys.path.insert(0, str(RELEASE))
import make_exposure as X

FRACTION = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.5, 4: 0.75, 5: 1.0}
KEY = ["domain", "tech_node", "isco3", "section"]
PARTS = ["digital_sub", "digital_comp", "green_sub", "green_comp"]
NACE_SECTIONS = [(1, 3, "A"), (5, 9, "B"), (10, 33, "C"), (35, 35, "D"), (36, 39, "E"), (41, 43, "F"), (45, 47, "G"),
                 (49, 53, "H"), (55, 56, "I"), (58, 63, "J"), (64, 66, "K"), (68, 68, "L"), (69, 75, "M"), (77, 82, "N"),
                 (84, 84, "O"), (85, 85, "P"), (86, 88, "Q"), (90, 93, "R"), (94, 96, "S"), (97, 98, "T"), (99, 99, "U")]


def nace_section(code):
    try:
        d = int(str(code).split(".")[0])
    except Exception:
        return "?"
    for lo, hi, s in NACE_SECTIONS:
        if lo <= d <= hi:
            return s
    return "?"


def usable_ratings():
    """The ratings of the grid's triplets, one row per answered call, rating on the 0 to 5 scale."""
    g = pd.read_csv(HERE / "grid.csv", dtype={"isco3": str, "section": str}, usecols=KEY)
    d = pd.read_parquet(HERE / "ratings.parquet")
    d = d.merge(g, on=KEY, how="inner")
    d = d[~d.failed.astype(str).eq("True")].drop_duplicates("call_id", keep="last").copy()
    d["rating"] = pd.to_numeric(d.rating, errors="coerce")
    return d[d.rating.between(0, 5)]


def fractions(d=None):
    """-> [domain, tech_node, isco3, section, sigma, kappa], the mean fraction over the ratings of each construct."""
    d = usable_ratings() if d is None else d
    f = d.assign(f=d.rating.round().astype(int).map(FRACTION))
    cell = f.groupby(KEY + ["construct"]).f.mean().unstack("construct")
    return cell.rename(columns={"substitutability": "sigma", "complementarity": "kappa"}).reset_index()


def shares():
    return pd.read_csv(HERE / "pair_employment_shares.csv", dtype={"isco3": str, "section": str})


def _side(kind, fr, start, end):
    """Per (entity, cell) content for both domains. kind='occupation' -> [isco4, section, parts]; 'industry' -> [nace, isco3, parts]."""
    frames = []
    for dom in ("digital", "green"):
        m = X.base_scores(kind, dom, start, end)
        N = m.shape[1]
        s = m.stack().rename("x").reset_index()
        s.columns = ["code", "tech_node", "x"]
        f = fr[fr.domain == dom][["tech_node", "isco3", "section", "sigma", "kappa"]]
        if kind == "occupation":
            s["isco4"] = s.code.astype(str).str.zfill(4); s["isco3"] = s.isco4.str[:3]
            mm = s.merge(f, on=["tech_node", "isco3"], how="inner"); keys = ["isco4", "section"]
        else:
            s["nace"] = s.code.astype(str); s["section"] = s.nace.map(nace_section)
            mm = s[s.section != "?"].merge(f, on=["tech_node", "section"], how="inner"); keys = ["nace", "isco3"]
        g = mm.assign(sub=mm.x * mm.sigma, comp=mm.x * mm.kappa).groupby(keys)[["sub", "comp"]].sum() / N
        frames.append(g.rename(columns={"sub": f"{dom}_sub", "comp": f"{dom}_comp"}))
    return pd.concat(frames, axis=1).fillna(0.0).reset_index()


def occupation_contents(start, end, fr=None):
    """-> [isco4, digital, green, digital_sub, digital_comp, green_sub, green_comp] for every occupation of the release."""
    fr = fractions() if fr is None else fr
    side = _side("occupation", fr, start, end)
    w = shares().rename(columns={"share_in_occupation": "w"})[["isco3", "section", "w"]]
    codes = pd.DataFrame({"isco4": X.base_scores("occupation", "digital", start, end).index.astype(str)})
    codes["isco3"] = codes.isco4.str[:3]
    full = codes.merge(w, on="isco3", how="left").merge(side, on=["isco4", "section"], how="left")
    for c in PARTS:
        full[c] = full[c].fillna(0.0) * full.w.fillna(0.0)
    out = full.groupby("isco4")[PARTS].sum().reindex(codes.isco4).fillna(0.0)
    exp = pd.concat([X.domain_exposure("occupation", dm, start, end).set_index("isco4") for dm in ("digital", "green")], axis=1)
    exp.columns = ["digital", "green"]
    return exp.join(out).reset_index()


def industry_contents(start, end, fr=None):
    """-> [nace, digital, green, parts] for every three-digit industry of the release."""
    fr = fractions() if fr is None else fr
    side = _side("industry", fr, start, end)
    w = shares().rename(columns={"share_in_section": "w"})[["isco3", "section", "w"]]
    codes = pd.DataFrame({"nace": X.base_scores("industry", "digital", start, end).index.astype(str)})
    codes["section"] = codes.nace.map(nace_section)
    full = codes.merge(w, on="section", how="left").merge(side, on=["nace", "isco3"], how="left")
    for c in PARTS:
        full[c] = full[c].fillna(0.0) * full.w.fillna(0.0)
    out = full.groupby("nace")[PARTS].sum().reindex(codes.nace).fillna(0.0)
    exp = pd.concat([X.domain_exposure("industry", dm, start, end).set_index("nace") for dm in ("digital", "green")], axis=1)
    exp.columns = ["digital", "green"]
    return exp.join(out).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", nargs=2, type=int, required=True, metavar=("START", "END"))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    fr = fractions()
    occ = occupation_contents(*a.window, fr); ind = industry_contents(*a.window, fr)
    out = Path(a.out_dir or f"contents_{a.window[0]}_{a.window[1]}"); out.mkdir(exist_ok=True)
    fr.to_csv(out / "triplet_fractions.csv", index=False)
    occ.to_csv(out / "contents_occupation.csv", index=False)
    ind.to_csv(out / "contents_industry.csv", index=False)
    print(f"{len(fr):,} rated triplets, {len(occ)} occupations, {len(ind)} industries -> {out}")
    for dm in ("digital", "green"):
        print(f"  {dm}: occupations corr(sub, comp) {occ[f'{dm}_sub'].corr(occ[f'{dm}_comp']):.2f}, "
              f"industries {ind[f'{dm}_sub'].corr(ind[f'{dm}_comp']):.2f}")


if __name__ == "__main__":
    main()
