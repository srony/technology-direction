"""Exposure scores of occupations (ISCO-08 four-digit) and industries (NACE Rev. 2 three-digit) to digital and green
technologies, for any window of patent filing years.

Standalone: needs only Python 3, pandas, numpy and pyarrow, and the files of this folder.

The exposure of an occupation or industry e to a base technology n (184 digital base technologies, 66 green ENV-TECH
nodes) over the window [start, end] is
    x_en = asinh( sum over the years start..end of the yearly value of (e, n) ),
the inverse hyperbolic sine applied once to the summed yearly values. Codes without a value for a technology have
exposure zero. Every aggregate is the mean of x_en over its base technologies, zeros included: a digital intermediate
technology or root over the base technologies it contains, a green group over its nodes, and the domain exposure of e
over ALL base technologies of the domain.

Examples
    python make_exposure.py --side occupation --domain digital --window 2009 2023 --average
    python make_exposure.py --side occupation --domain digital --window 2018 2023 --technology "Robot Manipulators and Arm Systems"
    python make_exposure.py --side industry --domain green --window 2009 2023 --level group --out green_groups.csv
    python make_exposure.py --list --domain green --level group
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = {"occupation": "isco4", "industry": "nace"}
BASE = {"digital": "base", "green": "node"}


def technologies(domain=None, level=None):
    t = pd.read_csv(HERE / "technologies.csv", dtype=str, keep_default_na=False)
    if domain:
        t = t[t.domain == domain]
    if level:
        t = t[t.level == level]
    return t


def yearly(side, domain):
    d = pd.read_parquet(HERE / f"exposure_yearly_{side}.parquet")
    return d[d.domain == domain]


def base_scores(side, domain, start, end):
    """Code x base technology matrix of x_en over the window, every code of the side, zero where nothing was filed."""
    d = yearly(side, domain)
    codes = sorted(pd.read_parquet(HERE / f"exposure_yearly_{side}.parquet", columns=[CODE[side]])[CODE[side]].unique())
    base = technologies(domain, BASE[domain]).technology
    s = d[d.year.between(start, end)].groupby([CODE[side], "technology"])["value"].sum()
    m = s.unstack("technology").reindex(index=codes, columns=base).fillna(0.0)
    return np.arcsinh(m)


def members(domain, level):
    """Technology of the level -> the base technologies it averages (itself at the base level)."""
    base = technologies(domain, BASE[domain])
    if level == BASE[domain]:
        return {t: [t] for t in base.technology}
    key = "root" if level == "root" else "parent"
    out = {t: list(base.technology[base[key] == t]) for t in technologies(domain, level).technology}
    assert all(out.values()), f"{domain} {level}: a technology without base technologies"
    return out


def exposure(side, domain, start, end, level=None, techs=None):
    """Long table [code, technology, name, exposure] for every code of the side and every selected technology."""
    level = level or BASE[domain]
    tech = technologies(domain, level)
    if tech.empty:
        sys.exit(f"{domain} has no level '{level}'; digital: base, intermediate, root. green: node, group")
    if techs:
        want = {s.strip().lower() for s in techs}
        tech = tech[tech.technology.str.lower().isin(want) | tech.name.str.lower().isin(want)]
        if tech.empty:
            sys.exit(f"no {domain} technology matches {techs}; run with --list to see the names")
    m = base_scores(side, domain, start, end)
    mem = members(domain, level)
    out = pd.DataFrame({t: m[mem[t]].mean(axis=1) for t in tech.technology})
    out.index.name = CODE[side]
    out = out.reset_index().melt(id_vars=CODE[side], var_name="technology", value_name="exposure")
    out = out.sort_values(CODE[side], kind="stable").reset_index(drop=True)
    return out.merge(tech[["technology", "name"]], on="technology")[[CODE[side], "technology", "name", "exposure"]]


def domain_exposure(side, domain, start, end):
    """Mean over all base technologies of the domain, zeros included."""
    m = base_scores(side, domain, start, end)
    return (m.sum(axis=1) / m.shape[1]).rename(f"{domain}_exposure").rename_axis(CODE[side]).reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--side", choices=list(CODE), help="occupation (ISCO-08 four-digit) or industry (NACE three-digit)")
    p.add_argument("--domain", choices=["digital", "green"])
    p.add_argument("--window", nargs=2, type=int, metavar=("START", "END"), help="first and last patent filing year, 2000-2024")
    p.add_argument("--level", help="digital: base, intermediate or root. green: node or group. Default: base / node")
    p.add_argument("--technology", nargs="+", help="technology names or identifiers (see --list). Default: all of the level")
    p.add_argument("--average", action="store_true", help="the domain exposure: mean over all base technologies")
    p.add_argument("--list", action="store_true", help="list the technologies of --domain (and --level) and stop")
    p.add_argument("--out", help="output CSV. Default: printed to the screen")
    a = p.parse_args()
    if a.list:
        print(technologies(a.domain, a.level).to_string(index=False))
        return
    if not (a.side and a.domain and a.window):
        p.error("--side, --domain and --window are required")
    start, end = a.window
    if start > end:
        p.error("the window must run from START to END with START <= END")
    res = domain_exposure(a.side, a.domain, start, end) if a.average else exposure(a.side, a.domain, start, end, a.level, a.technology)
    if a.out:
        res.to_csv(a.out, index=False)
        print(f"{len(res):,} rows written to {a.out}")
    else:
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
