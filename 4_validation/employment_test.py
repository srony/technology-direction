"""Substitution and complementarity content and employment change, on the public Eurostat table of employment by
two-digit occupation (lfsa_egai2d).

For occupation o in country c over a period t0 to t1,
    ln E_{co,t1} - ln E_{co,t0} = b_S Sigma_o + b_G Gamma_o + phi ln E_{co,t0} + mu_ISCO1 + delta_c + e_{co},
with the contents built from the patents filed in the years of the period, averaged over the four-digit codes of the
two-digit occupation and divided by their standard deviation across cells, the digital and the green pair in separate
regressions, weighted by base-year employment, standard errors clustered by occupation, on the cells observed in all
three periods. Employment at the start of a period that crosses 2020 is multiplied by Eurostat's national correction
for the 2021 redefinition of employment (lfsi_emp_a against lfsi_emp_a_h, extended to ages 65 and over with
lfst_r_lfe2emp). The default countries are the 21 of the companion paper's microdata, so that the two sources compare.

    python employment_test.py
    python employment_test.py --countries EU27
"""
import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "3_direction"))
import contents as C

CACHE = HERE / "eurostat"
URL = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{code}?format=TSV&compressed=true"
EU27 = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
        "NL", "PL", "PT", "RO", "SE", "SI", "SK"]
MICRO21 = [c for c in EU27 if c not in ("BG", "SI", "MT", "DE", "AT", "NL")]
PERIODS = ((2011, 2019), (2020, 2023), (2011, 2023))
CC = C.PARTS
st = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def load(code):
    """Eurostat TSV -> long frame [dims..., year, value]."""
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / f"{code}.tsv"
    if not fp.exists():
        req = urllib.request.Request(URL.format(code=code), headers={"User-Agent": "direction-loop/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            raw = r.read()
        fp.write_bytes(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    t = pd.read_csv(fp, sep="\t", dtype=str)
    dims = t.columns[0].split("\\")[0].split(",")
    keys = t[t.columns[0]].str.split(",", expand=True); keys.columns = dims
    d = pd.concat([keys, t[t.columns[1:]]], axis=1).melt(id_vars=dims, var_name="year", value_name="raw")
    d["year"] = pd.to_numeric(d.year.str.strip(), errors="coerce")
    d["value"] = pd.to_numeric(d.raw.str.strip().str.split(" ").str[0].replace(":", np.nan), errors="coerce")
    return d.dropna(subset=["year"]).astype({"year": int}).drop(columns="raw")


def public_cells():
    P = load("lfsa_egai2d")
    age = "Y15-74" if "Y15-74" in set(P.age) else "Y15-64"
    P = P[(P.sex == "T") & (P.age == age) & (P.unit == "THS_PER") & P.isco08.str.fullmatch(r"OC\d\d") & P.geo.isin(EU27)]
    return P.assign(isco2=P.isco08.str[2:], country_id=P.geo).pivot_table(index=["country_id", "isco2"], columns="year", values="value")


def break_factor(countries, y0):
    """National employment-level factor for year y0 (<= 2020): corrected / old-definition employment 15-64, extended to
    15+ with the old-definition share of 15-64 in 15+ employment. 1 where Eurostat reports no break."""
    def ann(code):
        d = load(code)
        d = d[(d.indic_em == "EMP_LFS") & (d.sex == "T") & (d.age == "Y15-64") & (d.year == y0) & d.unit.isin(["PC_POP", "THS_PER"])]
        return d.pivot_table(index="geo", columns="unit", values="value")
    c, o = ann("lfsi_emp_a"), ann("lfsi_emp_a_h")
    f = (c / o).dropna()
    e = load("lfst_r_lfe2emp")
    e = e[(e.geo.str.len() == 2) & (e.sex == "T") & (e.year == y0) & e.age.isin(["Y15-64", "Y_GE15"])].pivot_table(index="geo", columns="age", values="value")
    s = e["Y15-64"] / e["Y_GE15"]
    out = {}
    for g in countries:
        if g not in f.index:
            raise SystemExit(f"no 2021-break correction for {g} in {y0}")
        if f.loc[g, "PC_POP"] == 1.0 and f.loc[g, "THS_PER"] == 1.0:
            out[g] = 1.0
        else:
            out[g] = 1.0 + (f.loc[g, "THS_PER"] - 1.0) * s[g]
    return pd.Series(out)


def frame(CY, X, y0, y1, countries):
    f = break_factor(countries, y0).reindex(CY.index.get_level_values("country_id")).to_numpy() if y1 > 2020 else 1.0
    d = pd.DataFrame({"dlog_emp": np.log(CY[y1]) - np.log(f * CY[y0]), "log_e0": np.log(CY[y0]), "e0": CY[y0]})
    d = d.replace([np.inf, -np.inf], np.nan).dropna().reset_index().rename(columns={"isco2": "occ"}).merge(X.reset_index(), on="occ")
    d = d[d.country_id.isin(countries)].copy()
    d["isco1"] = d.occ.str[0]
    for c in CC:
        s = d[c].std(ddof=0); d[c] = d[c] / (s if s and s > 0 else 1.0)
    return d


def fit(d, pair):
    return pf.feols(f"dlog_emp ~ {' + '.join(pair)} + log_e0 | isco1 + country_id", d, vcov={"CRV1": "occ"}, weights="e0")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--countries", default="micro21", help="micro21 (default), EU27, or a comma-separated list")
    ap.add_argument("--out", default="employment_test.csv")
    a = ap.parse_args()
    cs = MICRO21 if a.countries == "micro21" else EU27 if a.countries == "EU27" else a.countries.split(",")
    P = public_cells()
    fr = C.fractions()
    X = {}
    for per in PERIODS:
        occ = C.occupation_contents(*per, fr)
        X[per] = occ.assign(occ=occ.isco4.str[:2]).groupby("occ")[CC].mean()
    F = {per: frame(P, X[per], *per, cs) for per in PERIODS}
    common = set.intersection(*(set(map(tuple, d[["country_id", "occ"]].values)) for d in F.values()))
    rows = []
    for per, d in F.items():
        d = d[[tuple(v) in common for v in d[["country_id", "occ"]].values]]
        for pair in (["digital_sub", "digital_comp"], ["green_sub", "green_comp"]):
            m = fit(d, pair)
            for v in pair:
                rows.append(dict(period=f"{per[0]}-{per[1]}", term=v, b=float(m.coef()[v]), se=float(m.se()[v]), p=float(m.pvalue()[v]),
                                 n=int(m._N), clusters=int(d.occ.nunique()), countries=int(d.country_id.nunique())))
    T = pd.DataFrame(rows); T.to_csv(a.out, index=False)
    q = T.iloc[0]
    print(f"public table lfsa_egai2d, {q.n} cells per period, {q.clusters} occupations, {q.countries} countries")
    for term in T.term.unique():
        r = T[T.term == term].set_index("period")
        print(f"  {term:13s} " + "   ".join(f"{per}: {r.loc[per, 'b']:+.3f}{st(r.loc[per, 'p']):<3s}({r.loc[per, 'se']:.3f})" for per in r.index))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
