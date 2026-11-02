#!/usr/bin/env python3
"""Effect of the decoder alone on the same two-head networks (BBBC039 test, base protocol, ten
seeds): component-preserving decoder (results/bbbc_component_preserving/per_image.csv) minus
the original global nearest-marker decoder (results/bbbc_<target>_10seed/*/test_per_image.csv),
same checkpoints and thresholds.  Image-paired bootstrap of seed-averaged AJI (10,000 draws).
Writes results/decoder_fix_effect.json."""
import csv, json
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]; rng = np.random.default_rng(0)
long = list(csv.DictReader(open(R / "results/bbbc_component_preserving/per_image.csv")))
ids = sorted({r["id"] for r in long}); out = {}
for t in ("contour", "contact"):
    new = {(int(r["seed"]), r["id"]): float(r["aji"]) for r in long if r["method"] == t}
    old = {(s, r["id"]): float(r["aji"]) for s in range(10)
           for r in csv.DictReader(open(R / f"results/bbbc_{t}_10seed/vanilla_{t}_b7_s{s}/test_per_image.csv"))}
    d = np.array([[new[s, i] - old[s, i] for i in ids] for s in range(10)]).mean(0)
    dr = [d[rng.integers(0, 50, 50)].mean() for _ in range(10000)]
    out[t] = [float(d.mean()), np.percentile(dr, [2.5, 97.5]).tolist(), int((d > 0).sum())]
    print(t, out[t])
(R / "results/decoder_fix_effect.json").write_text(json.dumps(out, indent=1))
