#!/usr/bin/env python3
"""Wider, training-image-selected post-processing for expanded NucleusSegData.

No model is retrained.  The original runs selected the contact threshold and
minimum object size on the 25 official training images from a grid whose
chosen values often sat at its edge (threshold 0.1 = minimum, size 128 =
maximum).  This script re-decodes the saved final-epoch checkpoints with a
grid that extends past both edges, optionally adds 8-fold dihedral test-time
augmentation, again selecting only on the
training images (pooled Bilkent nucleus F1, as in the original protocol and
in AttentionBoost), then decodes the 36 test images once.  Ties are broken by
grid order (smaller threshold, then smaller minimum size).
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch

from breastseg.core import Net, features
from breastseg.data import load_nucleusseg_extended
from breastseg.instance_metrics import bilkent_metrics, published_metrics
from posthoc_bbbc_decoding import watershed_contact
from train_bbbc_contact import contact_instances
from train_nucleusseg_contact import aggregate_bilkent

ROOT = Path(__file__).resolve().parents[1]
STUDY = "nucleusseg_extended_contact_10seed"
THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
MIN_SIZES = (16, 64, 128, 192, 256, 384)
DECODERS = ("nearest_marker",)  # the locked decoder; only the grid is widened
ADDITIVE = ("n_reference", "n_prediction", "n_one_to_one", "pixel_true_positive",
            "n_reference_pixels", "n_prediction_pixels")


def decode(probability, config):
    foreground = probability[0] > .5
    if config["decoder"] == "component_preserving":
        from analyze_component_preserving import component_preserving
        return component_preserving(foreground, probability[1] > config["threshold"], min_size=config["min_size"])
    if config["decoder"] == "nearest_marker":
        return contact_instances(foreground, probability[1] > config["threshold"], min_size=config["min_size"])
    return watershed_contact(foreground, probability[1], config["threshold"], min_size=config["min_size"])


def bilkent_job(job):
    truth, probability, config = job
    return bilkent_metrics(truth, decode(probability, config))


def full_job(job):
    truth, probability, config = job
    prediction = decode(probability, config)
    return {**published_metrics(truth, prediction),
            **{f"bilkent_{k}": v for k, v in bilkent_metrics(truth, prediction).items()}}


@torch.inference_mode()
def probabilities(model, cases, edge, tta):
    model.eval()
    output = {}
    for case in cases:
        x = features(case["image"], edge=edge, size=512)[None].cuda()
        views = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
        total = 0
        for k, flip in views:
            view = torch.rot90(x, k, (-2, -1))
            view = view.flip(-1) if flip else view
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = model(view).float().sigmoid()
            value = value.flip(-1) if flip else value
            total = total + torch.rot90(value, -k, (-2, -1))
        value = (total / len(views))[0].cpu().numpy()
        height, width = case["instances"].shape
        output[case["id"]] = np.stack([cv2.resize(c, (width, height), interpolation=cv2.INTER_LINEAR)
                                       for c in value]).astype(np.float32)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["standard_contact_b7", "gabor_contact_b7"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--output", type=Path, default=ROOT / "results/posthoc_nucleusseg_decoding")
    parser.add_argument("--decoder", default="nearest_marker",
                        help="nearest_marker (original, global) or component_preserving (corrected)")
    parser.add_argument("--study", default=STUDY,
                        help="checkpoint study; nucleusseg_extended_contour_10seed gives the matched Contour* arm")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cases = load_nucleusseg_extended(ROOT / "data/nucleusseg_extended")
    train = [c for c in cases if c["split"] == "train"]
    test = [c for c in cases if c["split"] == "test"]
    variants = {"standard_contact_b7": "vanilla_b7", "gabor_contact_b7": "gabor_b7", "standard_contour_b7": "vanilla_b7"}
    args.output.mkdir(parents=True, exist_ok=True)
    grid_configs = [{"decoder": d, "threshold": t, "min_size": m}
                    for d in (args.decoder,) for t in THRESHOLDS for m in MIN_SIZES]
    with Pool(20) as pool:
        for seed in args.seeds:
            for model_name in args.models:
                run = f"{model_name}_s{seed}"
                if (args.output / f"{run}.json").exists():
                    print("SKIP", run, flush=True)
                    continue
                state = torch.load(ROOT / "checkpoints" / args.study / run / "final.pt", map_location="cpu",
                                   weights_only=True)
                model = Net(variants[model_name], pretrained=False, classes=2).cuda()
                model.load_state_dict(state["model_state_dict"], strict=True)
                record = {"run": run, "selection": "official training images, pooled Bilkent nucleus F1",
                          "settings": {}}
                for tta in (False, True):
                    key = "tta8" if tta else "plain"
                    train_p = probabilities(model, train, model.use_edge, tta)
                    grid = []
                    for config in grid_configs:
                        rows = pool.map(bilkent_job, [(c["instances"], train_p[c["id"]], config) for c in train])
                        grid.append({**config, "train_nucleus_f1": aggregate_bilkent(rows)["nucleus_f1"]})
                    candidates = {"selected": max(grid, key=lambda g: g["train_nucleus_f1"])}
                    test_p = probabilities(model, test, model.use_edge, tta)
                    record["settings"][key] = {"training_grid": grid}
                    for label, chosen in candidates.items():
                        config = {k: chosen[k] for k in ("decoder", "threshold", "min_size")}
                        rows = pool.map(full_job, [(c["instances"], test_p[c["id"]], config) for c in test])
                        out_rows = []
                        for case, row in zip(test, rows):
                            out_rows.append({"id": case["id"], "subset": case["subset"], **row})
                        path = args.output / f"{run}__{key}__{label}_test_per_image.csv"
                        with path.open("w", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]), lineterminator="\n")
                            writer.writeheader(); writer.writerows(out_rows)
                        subsets = {}
                        for subset in ("Huh7", "HepG2"):
                            sub = [r for r in out_rows if r["subset"] == subset]
                            pooled = aggregate_bilkent([{k: r[f"bilkent_{k}"] for k in ADDITIVE} for r in sub])
                            subsets[subset] = {"nucleus_f1": pooled["nucleus_f1"],
                                               "aji": float(np.mean([r["aji"] for r in sub])),
                                               "pq": float(np.mean([r["pq"] for r in sub]))}
                        record["settings"][key][label] = {"config": config,
                                                          "train_nucleus_f1": chosen["train_nucleus_f1"],
                                                          "test": subsets}
                    print(json.dumps({"run": run, key: {k: v for k, v in record["settings"][key].items()
                                                        if k != "training_grid"}}), flush=True)
                (args.output / f"{run}.json").write_text(json.dumps(record, indent=2))
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
