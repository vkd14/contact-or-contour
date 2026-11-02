#!/usr/bin/env python3
"""Decoder contrast on the 53 external DSB2018 images: the same BBBC039-trained semantic
networks (bbbc_semantic_ws_10seed, ten seeds) decoded by connected components and by the
BBBC039-validation-selected distance watershed (no external tuning).  Image-paired bootstrap
(10,000 draws) of seed-averaged AJI and PQ.  The watershed arm must reproduce the semantic_ws
rows of results/dsb2018_external/per_image.csv.  Writes results/dsb2018_external/decoder.json."""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from evaluate_dsb_external import RES, load_cohort, load_unet, semantic_run, unet_probabilities  # noqa: E402
from posthoc_bbbc_decoding import semantic_instances  # noqa: E402
from breastseg.instance_metrics import published_metrics  # noqa: E402

def main():
    cases = load_cohort(); ids = [c["id"] for c in cases]
    saved = {(int(r["seed"]), r["id"]): float(r["aji"]) for r in csv.DictReader(open(RES / "dsb2018_external/per_image.csv"))
             if r["method"] == "semantic_ws"}
    A = {k: np.zeros((10, len(ids))) for k in ("cc_aji", "ws_aji", "cc_pq", "ws_pq")}; worst = 0.0
    for seed in range(10):
        run = semantic_run(seed)
        cfg = json.loads((RES / "posthoc_bbbc_decoding" / f"{run.replace('/', '__')}.json").read_text())["settings"]["plain"]["selected"]["config"]
        model = load_unet(ROOT / "checkpoints" / run / "best.pt", 1)
        for j, c in enumerate(cases):
            fg = unet_probabilities(model, c["image"])[0] > .5
            for k, lab in (("cc", semantic_instances(fg, "components")), ("ws", semantic_instances(fg, "watershed", cfg["h"], cfg["sigma"]))):
                m = published_metrics(c["truth"], lab); A[f"{k}_aji"][seed, j] = m["aji"]; A[f"{k}_pq"][seed, j] = m["pq"]
            worst = max(worst, abs(A["ws_aji"][seed, j] - saved[seed, c["id"]]))
        del model
        print("seed", seed, "cc", round(A["cc_aji"][seed].mean(), 4), "ws", round(A["ws_aji"][seed].mean(), 4), "max|dAJI| vs saved", round(worst, 6), flush=True)
    rng = np.random.default_rng(0); out = {"max_abs_diff_vs_saved_ws": worst}
    for met in ("aji", "pq"):
        d = A[f"ws_{met}"].mean(0) - A[f"cc_{met}"].mean(0)
        dr = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        out[met] = {"cc": float(A[f"cc_{met}"].mean()), "ws": float(A[f"ws_{met}"].mean()),
                    "ws_minus_cc": [float(d.mean()), np.percentile(dr, [2.5, 97.5]).tolist(), int((d > 0).sum()), len(d)]}
    (RES / "dsb2018_external/decoder.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out))

if __name__ == "__main__":
    main()
