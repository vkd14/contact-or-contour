#!/usr/bin/env python3
"""Apply Astra's decoder grid (scripts/astra_utils.candidates) to the ten existing
native-resolution contact checkpoints (bbbc_contact_native_10seed).

Per seed: eight-fold TTA probabilities on the 50 validation and 50 test images
(reflect-padded to 544x704), the 25-candidate decoder chosen by mean validation
object F1, and the test images decoded once with that decoder.  Writes per-image
test metrics, predictions, and a summary to results/native_astra_decoder/.
"""
import csv
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import posthoc_bbbc_decoding as phd  # noqa: E402
from astra_utils import candidates, decode  # noqa: E402
from breastseg.core import Net  # noqa: E402
from breastseg.data import load_bbbc039  # noqa: E402
from breastseg.instance_metrics import published_metrics  # noqa: E402
from train_bbbc_contact import instance_annotations  # noqa: E402

OUT = ROOT / "results/native_astra_decoder"
METRICS = ("dice1", "aji", "dq", "sq", "pq", "caicedo_af1")


def score(job):
    truth, probability, config = job
    return published_metrics(truth, decode(probability, config))


def main():
    phd.NATIVE_PAD = (544, 704)
    cv2.setNumThreads(1)
    cases = load_bbbc039(ROOT / "data/bbbc039", ("val", "test"))
    ann = instance_annotations(ROOT / "data/bbbc039")
    for c in cases:
        c["instances"] = ann[c["id"]]
    val = [c for c in cases if c["split"] == "val"]
    test = [c for c in cases if c["split"] == "test"]
    OUT.mkdir(parents=True, exist_ok=True)
    grid = candidates()
    with get_context("spawn").Pool(12) as pool:
        for seed in range(10):
            run = f"bbbc_contact_native_10seed/vanilla_contact_b7_s{seed}"
            dest = OUT / f"s{seed}"
            if (dest / "summary.json").exists():
                print("SKIP", seed, flush=True)
                continue
            dest.mkdir(parents=True, exist_ok=True)
            state = torch.load(ROOT / "checkpoints" / run / "best.pt", map_location="cpu", weights_only=True)
            model = Net("vanilla_b7", pretrained=False, classes=2).cuda()
            model.load_state_dict(state["model_state_dict"], strict=True)
            vp = phd.probabilities(model, val, False, True)
            tp = phd.probabilities(model, test, False, True)
            del model; torch.cuda.empty_cache()
            scores = []
            for config in grid:
                rows = pool.map(score, [(c["instances"], vp[c["id"]], config) for c in val], chunksize=2)
                scores.append({**config, "val_caicedo_af1": float(np.mean([r["caicedo_af1"] for r in rows])),
                               "val_aji": float(np.mean([r["aji"] for r in rows]))})
            best = max(scores, key=lambda g: g["val_caicedo_af1"])
            config = {k: v for k, v in best.items() if not k.startswith("val_")}
            rows = pool.map(score, [(c["instances"], tp[c["id"]], config) for c in test], chunksize=2)
            (dest / "predictions").mkdir(exist_ok=True)
            for c in test:
                cv2.imwrite(str(dest / "predictions" / f"{c['id']}_instances.tif"),
                            np.asarray(decode(tp[c["id"]], config)).astype(np.uint16))
            with open(dest / "test_per_image.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id"] + list(rows[0]), lineterminator="\n")
                writer.writeheader()
                for c, r in zip(test, rows):
                    writer.writerow({"id": c["id"], **r})
            summary = {"seed": seed, "selected": best, "validation_grid": scores,
                       "test_mean": {m: float(np.mean([r[m] for r in rows])) for m in METRICS}}
            (dest / "summary.json").write_text(json.dumps(summary, indent=2))
            print("COMPLETE", seed, json.dumps({"config": config, "test": summary["test_mean"]}), flush=True)


if __name__ == "__main__":
    main()
