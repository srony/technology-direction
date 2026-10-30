"""Score a silicon panel against the human panel: Earth Mover's similarity with its floor, ceiling and normalization.

Similarity is one minus the Earth Mover's distance between the two rating distributions on the 0 to 5 support, as a
share of the width of the scale. The floor is the similarity a uniform-random rater reaches against the human
distribution, the ceiling the mean similarity between two bootstrap resamples of the human ratings, and the
normalized similarity places the panel between them (0 random, 1 a second human panel). Within-persona variance
(unbiased, as in the calibration paper's Table 1; its variance-decomposition table uses the population variance) and
Shannon entropy (bits) are computed over the draws of each case and averaged.

Two inputs are accepted.

    python metrics.py --ratings ratings.csv [--human human.csv]
        ratings.csv from survey.py (case_id, question, draw, rating, failed and, if carried through, rating_survey);
        human.csv with case_id, question, rating_survey when the ratings file does not carry the human rating.
    python metrics.py --histograms counts.csv
        counts.csv with columns question, rating (0 to 5), human, silicon, the pooled counts per statement. The
        ceiling is bootstrapped from the human counts, so the per-statement table needs no respondent-level file.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.spatial import distance as spdist
from scipy.stats import wasserstein_distance

CATS = np.arange(6, dtype=float)
SPAN = 5.0
B_BOOT, SEED = 200, 48
QORDER = ["efficiency", "substitutability", "complementarity", "routinised"]


def counts(ratings) -> np.ndarray:
    r = pd.to_numeric(pd.Series(ratings), errors="coerce").dropna().round().astype(int)
    r = r[(r >= 0) & (r <= 5)]
    return np.bincount(r.to_numpy(), minlength=6)[:6].astype(float)


def _prob(c):
    s = c.sum()
    return c / s if s > 0 else None


def emd_sim(hc, mc) -> float:
    ph, pm = _prob(hc), _prob(mc)
    if ph is None or pm is None:
        return np.nan
    return float(np.clip(1.0 - wasserstein_distance(CATS, CATS, u_weights=ph, v_weights=pm) / SPAN, 0.0, 1.0))


def jsd_sim(hc, mc) -> float:
    ph, pm = _prob(hc), _prob(mc)
    if ph is None or pm is None:
        return np.nan
    return float(1.0 - spdist.jensenshannon(ph, pm, base=2))


def floor(hc) -> float:
    return emd_sim(hc, np.ones(6))


def ceiling(hc, b=B_BOOT, seed=SEED) -> float:
    """Mean similarity between two bootstrap resamples of the human ratings, drawn from their counts."""
    rng = np.random.default_rng(seed)
    n, p = int(hc.sum()), _prob(hc)
    sims = [emd_sim(rng.multinomial(n, p).astype(float), rng.multinomial(n, p).astype(float)) for _ in range(b)]
    return float(np.mean(sims))


def normalized(sim, flo, ceil) -> float:
    return (sim - flo) / (ceil - flo) if ceil > flo else np.nan


def entropy_bits(ratings) -> float:
    p = _prob(counts(ratings))
    if p is None:
        return np.nan
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def per_statement(human: pd.Series, silicon: pd.Series, draws: pd.DataFrame | None = None) -> dict:
    hc, mc = counts(human), counts(silicon)
    out = {"n_human": int(hc.sum()), "n_silicon": int(mc.sum()), "emd_sim": emd_sim(hc, mc), "jsd_sim": jsd_sim(hc, mc),
           "floor": floor(hc), "ceiling": ceiling(hc)}
    out["normalized"] = normalized(out["emd_sim"], out["floor"], out["ceiling"])
    if draws is not None:
        g = draws.groupby("case_id")["rating"]
        out["within_persona_var"] = float(g.var(ddof=1).mean())
        out["entropy_bits"] = float(g.apply(entropy_bits).mean())
    return out


def from_ratings(path, human_path=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[~df.failed.astype(bool)]
    df["rating"] = pd.to_numeric(df.rating, errors="coerce")
    df = df.dropna(subset=["rating"])
    df["question"] = df.question.str.lower()
    if human_path:
        h = pd.read_csv(human_path)[["case_id", "question", "rating_survey"]]
        h["question"] = h.question.str.lower()
        df = df.drop(columns=["rating_survey"], errors="ignore").merge(h, on=["case_id", "question"], how="left")
    if "rating_survey" not in df.columns:
        raise SystemExit("no human ratings: pass --human or carry rating_survey in the ratings file")
    df["rating_survey"] = pd.to_numeric(df.rating_survey, errors="coerce")
    rows = []
    for q in [q for q in QORDER if q in set(df.question)]:
        g = df[df.question == q]
        human = g.drop_duplicates("case_id").rating_survey.dropna()
        rows.append({"question": q, **per_statement(human, g.rating, g[["case_id", "rating"]])})
    return pd.DataFrame(rows)


def from_histograms(path) -> pd.DataFrame:
    c = pd.read_csv(path)
    c["question"] = c.question.str.lower()
    rows = []
    for q in [q for q in QORDER if q in set(c.question)]:
        g = c[c.question == q].set_index("rating").reindex(range(6)).fillna(0)
        hc, mc = g.human.to_numpy(float), g.silicon.to_numpy(float)
        row = {"question": q, "n_human": int(hc.sum()), "n_silicon": int(mc.sum()), "emd_sim": emd_sim(hc, mc),
               "jsd_sim": jsd_sim(hc, mc), "floor": floor(hc), "ceiling": ceiling(hc)}
        row["normalized"] = normalized(row["emd_sim"], row["floor"], row["ceiling"])
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ratings", help="ratings CSV from survey.py")
    ap.add_argument("--human", help="human ratings CSV (case_id, question, rating_survey)")
    ap.add_argument("--histograms", help="pooled counts per statement (question, rating, human, silicon)")
    ap.add_argument("--out", help="write the table to this CSV")
    a = ap.parse_args()
    if not (a.ratings or a.histograms):
        ap.error("pass --ratings or --histograms")
    table = from_ratings(a.ratings, a.human) if a.ratings else from_histograms(a.histograms)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    if a.out:
        table.to_csv(a.out, index=False)
