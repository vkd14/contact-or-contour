#!/usr/bin/env python3
"""Image-paired comparisons on the 53 external DSB2018 images (results/dsb2018_external/per_image.csv).

Per-image values are averaged over the ten seeds; 10,000 paired image bootstrap draws (RNG seed 0).
Endpoints: AJI, PQ, absolute count error, signed count bias (predicted - reference), and the
percentages of reference nuclei and of predicted objects without an IoU>0.5 match (matched pairs
recovered exactly from DQ = TP / (0.5 n_pred + 0.5 n_ref)); pooled rates are also reported.
Writes results/dsb2018_external/paired.json.
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "results/dsb2018_external"
def main():
    acc = defaultdict(lambda: defaultdict(list))
    pooled = defaultdict(lambda: np.zeros(3))
    for r in csv.DictReader(open(D / "per_image.csv")):
        n_pred, n_ref = float(r["n_pred"]), float(r["n_ref"])
        v = {"aji": float(r["aji"]), "pq": float(r["pq"]), "count_abs": abs(n_pred - n_ref),
             "count_bias": n_pred - n_ref, "merged_touching": float(r["merged_touching"]) / max(float(r["touching"]), 1)}
        tp = float(r["dq"]) * (n_pred + n_ref) / 2
        v["unmatched_ref_pct"] = 100 * (1 - tp / n_ref) if n_ref else 0.0
        v["unmatched_pred_pct"] = 100 * (1 - tp / n_pred) if n_pred else 0.0
        pooled[r["method"]] += np.array([tp, n_ref, n_pred])
        for k, x in v.items():
            acc[(r["method"], k)][r["id"]].append(x)
    ids = sorted(acc[("cellpose", "aji")])
    mean = {key: np.array([np.mean(acc[key][i]) for i in ids]) for key in acc}
    rng = np.random.default_rng(0); out = {}
    for a, b in (("contour_cp", "semantic_ws"), ("contact_cp", "contour_cp"), ("contact_cp", "semantic_ws"),
                 ("cellpose", "contact_cp")):
        name = f"{a}_minus_{b}"; out[name] = {}
        for k in ("aji", "pq", "count_abs", "count_bias", "unmatched_ref_pct", "unmatched_pred_pct"):
            d = mean[(a, k)] - mean[(b, k)]
            draws = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
            out[name][k] = [round(float(d.mean()), 4), [round(float(x), 4) for x in np.percentile(draws, [2.5, 97.5])],
                            int((d > 0).sum())]
        print(name, out[name])
    out["pooled_unmatched_pct"] = {m: {"reference": round(100 * (1 - t / r), 2), "prediction": round(100 * (1 - t / p), 2)}
                                   for m, (t, r, p) in pooled.items()}
    print(out["pooled_unmatched_pct"])
    (D / "paired.json").write_text(json.dumps(out, indent=2))
if __name__ == "__main__":
    main()
