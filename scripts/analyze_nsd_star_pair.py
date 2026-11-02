#!/usr/bin/env python3
"""Matched Contact* vs Contour* on expanded NucleusSegData: both re-decoded from their
final-epoch checkpoints with the widened training-selected grid + TTA (component-preserving
decoder; scripts/posthoc_nucleusseg_decoding.py).  Pooled Bilkent nucleus F1 per subset
(mean, seed SD over ten seeds) and the paired image bootstrap of seed-averaged AJI and
pooled nucleus F1 (10,000 draws).  Writes results/nucleusseg_star_pair.json."""
import csv, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SRC = {"contact": ("posthoc_nucleusseg_decoding_cp", "standard_contact_b7"),
       "contour": ("posthoc_nucleusseg_decoding_cp_contour", "standard_contour_b7")}
ADD = ("n_one_to_one", "n_reference", "n_prediction")

def load(t, s):
    d, m = SRC[t]
    return list(csv.DictReader(open(ROOT / "results" / d / f"{m}_s{s}__tta8__selected_test_per_image.csv")))

def main():
    rng = np.random.default_rng(0); out = {}
    for sub in ("Huh7", "HepG2"):
        data = {t: [[r for r in load(t, s) if r["subset"] == sub] for s in range(10)] for t in SRC}
        ids = [r["id"] for r in data["contact"][0]]; n = len(ids)
        def mat(t, k):
            return np.array([[float({r["id"]: r for r in run}[i][k]) for i in ids] for run in data[t]])
        A = {t: {k: mat(t, "bilkent_" + k) for k in ADD} for t in SRC}
        aji = {t: mat(t, "aji") for t in SRC}
        def f1(t, idx):
            a = A[t]; return 2 * a["n_one_to_one"][:, idx].sum(1) / (a["n_reference"][:, idx].sum(1) + a["n_prediction"][:, idx].sum(1))
        e = {"n_images": n}
        for t in SRC:
            f = f1(t, np.arange(n)); e[t] = {"nucleus_f1": [float(f.mean()), float(f.std(ddof=1))], "aji": float(aji[t].mean())}
        d = aji["contact"].mean(0) - aji["contour"].mean(0)
        dr = [d[rng.integers(0, n, n)].mean() for _ in range(10000)]
        e["aji_diff"] = [float(d.mean()), np.percentile(dr, [2.5, 97.5]).tolist(), int((d > 0).sum())]
        fd = []
        for _ in range(10000):
            idx = rng.integers(0, n, n); fd.append(f1("contact", idx).mean() - f1("contour", idx).mean())
        e["f1_diff"] = [float(f1("contact", np.arange(n)).mean() - f1("contour", np.arange(n)).mean()), np.percentile(fd, [2.5, 97.5]).tolist()]
        out[sub] = e; print(sub, json.dumps(e))
    (ROOT / "results/nucleusseg_star_pair.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
