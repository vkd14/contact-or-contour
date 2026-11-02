#!/usr/bin/env python3
"""Save BBBC039 test instance predictions of the Table 1 semantic + watershed family.

bbbc_semantic_ws_10seed checkpoints (selected by validation object F1 after
watershed), decoded with each seed's validation-selected (h, sigma), exactly as in
posthoc_bbbc_decoding.py (no TTA).  Writes
results/bbbc_semantic_ws_10seed_predictions/s<seed>/<id>_instances.tif and checks
per-image AJI against the saved selected-decoder CSV.
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import posthoc_bbbc_decoding as phd  # noqa: E402
from breastseg.core import Net  # noqa: E402
from breastseg.data import load_bbbc039  # noqa: E402
from breastseg.instance_metrics import published_metrics  # noqa: E402
from train_bbbc_contact import instance_annotations  # noqa: E402

OUT = ROOT / "results/bbbc_semantic_ws_10seed_predictions"
POST = ROOT / "results/posthoc_bbbc_decoding"


def main():
    test = load_bbbc039(ROOT / "data/bbbc039", ("test",))
    ann = instance_annotations(ROOT / "data/bbbc039")
    worst = 0.0
    for seed in range(10):
        run = f"bbbc_semantic_ws_10seed/vanilla_semantic_ws_b7_s{seed}"
        stem = run.replace("/", "__")
        config = json.loads((POST / f"{stem}.json").read_text())["settings"]["plain"]["selected"]["config"]
        with open(POST / f"{stem}__plain__selected_test_per_image.csv") as handle:
            saved = {r["id"]: float(r["aji"]) for r in csv.DictReader(handle)}
        state = torch.load(ROOT / "checkpoints" / run / "best.pt", map_location="cpu", weights_only=True)
        model = Net("vanilla_b7", pretrained=False, classes=1).cuda()
        model.load_state_dict(state["model_state_dict"], strict=True)
        probs = phd.probabilities(model, test, False, False)
        del model; torch.cuda.empty_cache()
        dest = OUT / f"s{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        for c in test:
            labels = phd.decode(probs[c["id"]], config)
            cv2.imwrite(str(dest / f"{c['id']}_instances.tif"), np.asarray(labels).astype(np.uint16))
            worst = max(worst, abs(published_metrics(ann[c["id"]], labels)["aji"] - saved[c["id"]]))
        print("seed", seed, config, "max |dAJI| so far", round(worst, 6), flush=True)


if __name__ == "__main__":
    main()
