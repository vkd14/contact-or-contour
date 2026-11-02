#!/usr/bin/env python3
"""Merged-touching rate, contact minus contour (pooled over nuclei per seed, then averaged over seeds),
with an image-paired bootstrap and a two-way bootstrap that also resamples each arm's seeds independently
(10,000 draws each, numpy default_rng(0)).  BBBC039 test: results/bbbc_merge_split/per_image.csv
(column merged_touching); external DSB2018: results/dsb2018_external/per_image.csv (contact_cp, contour_cp).
Writes results/merge_robustness.json."""
import csv, json
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]


def arms(path, names):
    rows = list(csv.DictReader(open(path))); ids = sorted({r["id"] for r in rows})
    def mm(m, k):
        A = np.zeros((10, len(ids)))
        for r in rows:
            if r["method"] == m: A[int(r["seed"]), ids.index(r["id"])] = float(r[k])
        return A
    return mm(names[0], "merged_touching"), mm(names[1], "merged_touching"), mm(names[0], "touching")[0], len(ids)


def analyse(Mc, Mo, T, n):
    rng = np.random.default_rng(0)
    pool = lambda M, s, i: 100 * M[np.ix_(s, i)].sum(1).mean() / T[i].sum()
    all_s, all_i = np.arange(10), np.arange(n)
    point = pool(Mc, all_s, all_i) - pool(Mo, all_s, all_i)
    img = [pool(Mc, all_s, i) - pool(Mo, all_s, i) for i in (rng.integers(0, n, n) for _ in range(10000))]
    two = []
    for _ in range(10000):
        i = rng.integers(0, n, n)
        two.append(pool(Mc, rng.integers(0, 10, 10), i) - pool(Mo, rng.integers(0, 10, 10), i))
    seed_rates = lambda M: [float(100 * M[s].sum() / T.sum()) for s in range(10)]
    return {"contact_pct": round(pool(Mc, all_s, all_i), 2), "contour_pct": round(pool(Mo, all_s, all_i), 2),
            "difference_points": round(float(point), 2),
            "image_bootstrap_ci": [round(float(x), 3) for x in np.percentile(img, [2.5, 97.5])],
            "two_way_ci": [round(float(x), 3) for x in np.percentile(two, [2.5, 97.5])],
            "seed_sd_points": {"contact": round(float(np.std(seed_rates(Mc), ddof=1)), 2),
                               "contour": round(float(np.std(seed_rates(Mo), ddof=1)), 2)}, "n_images": n}


out = {"bbbc039": analyse(*arms(R / "results/bbbc_merge_split/per_image.csv", ("contact", "contour"))),
       "external_dsb2018": analyse(*arms(R / "results/dsb2018_external/per_image.csv", ("contact_cp", "contour_cp")))}
(R / "results/merge_robustness.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
