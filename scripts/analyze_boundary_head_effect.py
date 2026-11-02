#!/usr/bin/env python3
"""Adding a boundary head: two-head contact model (component-preserving decoder) minus the
semantic network with validation-selected distance watershed (Table 1 family), BBBC039 test,
seed-averaged per-image AJI, image-paired bootstrap (10,000 draws); external value copied from
results/dsb2018_external/paired.json.  Writes results/boundary_head_effect.json."""
import csv, glob, json
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]; rng = np.random.default_rng(0)
long = list(csv.DictReader(open(R / "results/bbbc_component_preserving/per_image.csv"))); ids = sorted({r["id"] for r in long})
c = {(int(r["seed"]), r["id"]): float(r["aji"]) for r in long if r["method"] == "contact"}
C = np.array([[c[s, i] for i in ids] for s in range(10)]).mean(0)
W = np.array([[{r["id"]: float(r["aji"]) for r in csv.DictReader(open(p))}[i] for i in ids] for p in
              sorted(glob.glob(str(R / "results/posthoc_bbbc_decoding/bbbc_semantic_ws_10seed__*__plain__selected_test_per_image.csv")))]).mean(0)
d = C - W; dr = [d[rng.integers(0, 50, 50)].mean() for _ in range(10000)]
out = {"bbbc039_contact_minus_semantic_ws": [float(d.mean()), np.percentile(dr, [2.5, 97.5]).tolist(), int((d > 0).sum())],
       "external_contact_minus_semantic_ws": json.loads((R / "results/dsb2018_external/paired.json").read_text())["contact_cp_minus_semantic_ws"]["aji"]}
(R / "results/boundary_head_effect.json").write_text(json.dumps(out, indent=1)); print(out)
