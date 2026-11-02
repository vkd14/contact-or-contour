#!/usr/bin/env python3
"""Two-way (seed x image) bootstrap for the headline contrasts on AJI and object F1 (pre-specified endpoint).

Same inputs and contrasts as scripts/analyze_boundary_vs_target.py (BBBC039 test with the component-preserving
decoder and the validation-selected semantic watershed; external DSB2018 53 images).  Each draw resamples images
with replacement and, independently for each arm, its ten seeds with replacement; an arm's per-image value is the
mean over its resampled seeds.  The contour arm uses one seed draw in head_contour_minus_target = 2*contour -
semantic - contact.  10,000 draws, numpy default_rng(0) per contrast.  Exploratory.
Writes results/twoway_headline.json."""
import csv, glob, json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
METRICS = ("aji", "caicedo_af1")
CONTRASTS = {"head_contact": {"contact": 1, "semantic_ws": -1}, "head_contour": {"contour": 1, "semantic_ws": -1},
             "target": {"contact": 1, "contour": -1},
             "head_contour_minus_target": {"contour": 2, "semantic_ws": -1, "contact": -1}}


def seed_matrix(rows, method, ids, key):
    A = np.full((10, len(ids)), np.nan)
    for r in rows:
        if r["method"] == method:
            A[int(r["seed"]), ids.index(r["id"])] = float(r[key])
    assert not np.isnan(A).any()
    return A


def twoway(arms, weights):
    rng = np.random.default_rng(0); n = next(iter(arms.values())).shape[1]
    point = sum(w * arms[a].mean(0) for a, w in weights.items()).mean()
    draws = []
    for _ in range(10000):
        i = rng.integers(0, n, n)
        draws.append(sum(w * arms[a][np.ix_(rng.integers(0, 10, 10), i)].mean(0) for a, w in weights.items()).mean())
    return [round(float(point), 4), [round(float(x), 4) for x in np.percentile(draws, [2.5, 97.5])]]


def main():
    cp = list(csv.DictReader(open(R / "results/bbbc_component_preserving/per_image.csv")))
    ids = sorted({r["id"] for r in cp})
    sem = [list(csv.DictReader(open(p))) for p in
           sorted(glob.glob(str(R / "results/posthoc_bbbc_decoding/bbbc_semantic_ws_10seed__*__plain__selected_test_per_image.csv")))]
    assert len(sem) == 10
    ext = list(csv.DictReader(open(R / "results/dsb2018_external/per_image.csv")))
    eids = sorted({r["id"] for r in ext})
    out = {}
    for k in METRICS:
        bbbc = {m: seed_matrix(cp, m, ids, k) for m in ("contact", "contour")}
        bbbc["semantic_ws"] = np.array([[{r["id"]: float(r[k]) for r in s}[i] for i in ids] for s in sem])
        external = {m: seed_matrix(ext, src, eids, k) for m, src in
                    {"contact": "contact_cp", "contour": "contour_cp", "semantic_ws": "semantic_ws"}.items()}
        for name, arms in (("bbbc039", bbbc), ("external_dsb2018", external)):
            out.setdefault(name, {})[k] = {c: twoway(arms, w) for c, w in CONTRASTS.items()}
    (R / "results/twoway_headline.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
