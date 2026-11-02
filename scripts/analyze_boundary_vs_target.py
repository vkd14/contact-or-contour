#!/usr/bin/env python3
"""Boundary-head effect versus boundary-target effect, on AJI and object F1 (pre-specified endpoint).

BBBC039 test (50 images): two-head contact / full-contour models with the component-preserving decoder
(results/bbbc_component_preserving/per_image.csv) and the semantic network with validation-selected distance
watershed (results/posthoc_bbbc_decoding/bbbc_semantic_ws_10seed__*__plain__selected_test_per_image.csv).
External DSB2018 (53 images): results/dsb2018_external/per_image.csv.  Seeds are averaged within each image;
95% percentile CIs from 10,000 image-paired bootstrap draws (numpy default_rng(0), one generator per contrast).
Contrasts: head_contact = contact - semantic_ws, head_contour = contour - semantic_ws,
target = contact - contour, head_minus_target = head_contour - target (the smaller head effect vs. the target effect).
Exploratory.  Writes results/boundary_vs_target.json."""
import csv, glob, json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
METRICS = ("aji", "caicedo_af1")


def seed_mean(rows, method, ids, key):
    by = {}
    for r in rows:
        if r["method"] == method:
            by.setdefault(r["id"], []).append(float(r[key]))
    return np.array([np.mean(by[i]) for i in ids])


def boot(d):
    rng = np.random.default_rng(0); n = len(d)
    draws = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
    return [round(float(d.mean()), 4), [round(float(x), 4) for x in np.percentile(draws, [2.5, 97.5])], int((d > 0).sum()), n]


def contrasts(arms):
    out = {}
    for k in METRICS:
        c, b, s = arms["contact"][k], arms["contour"][k], arms["semantic_ws"][k]
        out[k] = {"means": {m: round(float(arms[m][k].mean()), 4) for m in arms},
                  "head_contact": boot(c - s), "head_contour": boot(b - s), "target": boot(c - b),
                  "head_contour_minus_target": boot((b - s) - (c - b))}
    return out


def main():
    cp = list(csv.DictReader(open(R / "results/bbbc_component_preserving/per_image.csv")))
    ids = sorted({r["id"] for r in cp})
    sem = [list(csv.DictReader(open(p))) for p in
           sorted(glob.glob(str(R / "results/posthoc_bbbc_decoding/bbbc_semantic_ws_10seed__*__plain__selected_test_per_image.csv")))]
    assert len(sem) == 10
    bbbc = {m: {k: seed_mean(cp, m, ids, k) for k in METRICS} for m in ("contact", "contour")}
    bbbc["semantic_ws"] = {k: np.array([[{r["id"]: float(r[k]) for r in s}[i] for i in ids] for s in sem]).mean(0) for k in METRICS}
    ext = list(csv.DictReader(open(R / "results/dsb2018_external/per_image.csv")))
    eids = sorted({r["id"] for r in ext})
    names = {"contact": "contact_cp", "contour": "contour_cp", "semantic_ws": "semantic_ws"}
    external = {m: {k: seed_mean(ext, src, eids, k) for k in METRICS} for m, src in names.items()}
    out = {"bbbc039": contrasts(bbbc), "external_dsb2018": contrasts(external)}
    (R / "results/boundary_vs_target.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
