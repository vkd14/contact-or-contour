#!/usr/bin/env python3
"""Validation-selected post-hoc decoding for trained BBBC039 checkpoints.

No model is retrained.  For each saved checkpoint this script computes
validation and test probabilities (plain and with 8-fold dihedral test-time
augmentation), selects the decoding configuration on the 50 official
validation images only (Caicedo object F1, the declared selection criterion),
and decodes the 50 official test images once with (i) the locked configuration
used in the original run and (ii) the validation-selected configuration.

Semantic checkpoints (one sigmoid head) get the standard instance baseline that
a reviewer would expect: distance-transform marker watershed with the h-maxima
depth selected on validation, alongside plain connected components.
Contact/contour checkpoints (two heads) are decoded with the locked
nearest-marker rule or with a marker-controlled watershed on the predicted
contact probability.
"""
import argparse
import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation

from breastseg.core import Net, features
from breastseg.data import load_bbbc039
from breastseg.instance_metrics import published_metrics, remap_labels
from train_bbbc_contact import contact_instances, instance_annotations, pad_reflect

NATIVE_PAD = None  # (544, 704) when checkpoints were trained at native resolution

ROOT = Path(__file__).resolve().parents[1]
CONTACT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
WATERSHED_H = (0.5, 1.0, 2.0, 3.0)
WATERSHED_SIGMA = (0.0, 1.0, 2.0)
METRICS = ("dice1", "aji", "dq", "sq", "pq", "caicedo_af1")


def fill_unmarked(labels, binary):
    """Keep foreground components that received no marker as their own object."""
    labels = np.asarray(labels).copy()
    missing = measure.label(binary & (labels == 0), connectivity=1)
    next_label = int(labels.max()) + 1
    for region in measure.regionprops(missing):
        labels[tuple(region.coords.T)] = next_label
        next_label += 1
    return labels


def watershed_contact(foreground, contact_probability, threshold, min_size=16):
    foreground = morphology.remove_small_holes(np.asarray(foreground, bool), max_size=min_size - 1)
    foreground = morphology.remove_small_objects(foreground, max_size=min_size - 1)
    marker_mask = morphology.remove_small_objects(foreground & (contact_probability <= threshold),
                                                  max_size=min_size - 1)
    markers = measure.label(marker_mask, connectivity=1)
    if markers.max() == 0:
        return remap_labels(measure.label(foreground, connectivity=1))
    labels = segmentation.watershed(contact_probability, markers, mask=foreground)
    return remap_labels(fill_unmarked(labels, foreground))


def semantic_instances(foreground, method, h=None, sigma=0.0, min_size=16):
    binary = morphology.remove_small_holes(np.asarray(foreground, bool), max_size=min_size - 1)
    binary = morphology.remove_small_objects(binary, max_size=min_size - 1)
    if not binary.any():
        return np.zeros(binary.shape, np.int32)
    if method == "components":
        return remap_labels(measure.label(binary, connectivity=1))
    distance = ndi.distance_transform_edt(binary)
    if sigma:
        distance = ndi.gaussian_filter(distance, sigma)
    markers = measure.label(morphology.h_maxima(distance, h) & binary, connectivity=1)
    if markers.max() == 0:
        return remap_labels(measure.label(binary, connectivity=1))
    labels = segmentation.watershed(-distance, markers, mask=binary)
    return remap_labels(fill_unmarked(labels, binary))


def decode(probability, config):
    kind = config["decoder"]
    if kind == "components":
        return semantic_instances(probability[0] > .5, "components")
    if kind == "distance_watershed":
        return semantic_instances(probability[0] > .5, "watershed", config["h"], config["sigma"])
    if kind == "nearest_marker":
        return contact_instances(probability[0] > .5, probability[1] > config["threshold"])
    if kind == "contact_watershed":
        return watershed_contact(probability[0] > .5, probability[1], config["threshold"])
    raise ValueError(kind)


def score_job(job):
    truth, probability, config = job
    return published_metrics(truth, decode(probability, config))


@torch.inference_mode()
def probabilities(model, cases, edge, tta):
    model.eval()
    output = {}
    for case in cases:
        if NATIVE_PAD:
            x = pad_reflect(features(case["image"], edge=edge, native=True), NATIVE_PAD)[None].cuda()
        else:
            x = features(case["image"], edge=edge, size=512)[None].cuda()
        views = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
        total = None
        for k, flip in views:
            view = torch.rot90(x, k, (-2, -1))
            if flip:
                view = view.flip(-1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = model(view).float().sigmoid()
            if flip:
                value = value.flip(-1)
            value = torch.rot90(value, -k, (-2, -1))
            total = value if total is None else total + value
        value = (total / len(views))[0].cpu().numpy()
        height, width = case["mask"].shape
        if NATIVE_PAD:
            output[case["id"]] = value[:, :height, :width].astype(np.float32)
            continue
        output[case["id"]] = np.stack([cv2.resize(c, (width, height), interpolation=cv2.INTER_LINEAR)
                                       for c in value]).astype(np.float32)
    return output


def candidate_configs(kind):
    if kind == "semantic":
        return [{"decoder": "components"}] + [{"decoder": "distance_watershed", "h": h, "sigma": g}
                                          for h in WATERSHED_H for g in WATERSHED_SIGMA]
    return ([{"decoder": "nearest_marker", "threshold": t} for t in CONTACT_THRESHOLDS]
            + [{"decoder": "contact_watershed", "threshold": t} for t in CONTACT_THRESHOLDS])


def evaluate(pool, cases, probs, config):
    jobs = [(case["instances"], probs[case["id"]], config) for case in cases]
    return pool.map(score_job, jobs, chunksize=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("semantic", "contact"), required=True)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="study/run pairs, e.g. bbbc_contact_10seed/vanilla_contact_b7_s0")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--locked", default=None,
                        help="locked decoder for the original protocol (components or nearest_marker)")
    parser.add_argument("--native-pad", nargs=2, type=int, default=None)
    args = parser.parse_args()
    global NATIVE_PAD
    NATIVE_PAD = tuple(args.native_pad) if args.native_pad else None
    cv2.setNumThreads(1)
    cases = load_bbbc039(ROOT / "data/bbbc039", ("val", "test"))
    annotations = instance_annotations(ROOT / "data/bbbc039")
    for case in cases:
        case["instances"] = annotations[case["id"]]
    validation = [c for c in cases if c["split"] == "val"]
    test = [c for c in cases if c["split"] == "test"]
    args.output.mkdir(parents=True, exist_ok=True)
    classes = 1 if args.kind == "semantic" else 2
    configs = candidate_configs(args.kind)
    with Pool(20) as pool:
        for run in args.runs:
            name = run.replace("/", "__")
            if (args.output / f"{name}.json").exists():
                print("SKIP", run, flush=True)
                continue
            checkpoint = torch.load(ROOT / "checkpoints" / run / "best.pt", map_location="cpu", weights_only=True)
            model = Net(args.variant, pretrained=False, classes=classes).cuda()
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            record = {"run": run, "variant": args.variant, "kind": args.kind,
                      "selection": "official validation split, mean Caicedo object F1", "settings": {}}
            for tta in (False, True):
                if args.kind == "semantic" and tta:
                    continue
                start = time.time()
                val_p = probabilities(model, validation, model.use_edge, tta)
                test_p = probabilities(model, test, model.use_edge, tta)
                seconds = time.time() - start
                grid = []
                for config in configs:
                    rows = evaluate(pool, validation, val_p, config)
                    grid.append({**config, "val_caicedo_af1": float(np.mean([r["caicedo_af1"] for r in rows])),
                                 "val_aji": float(np.mean([r["aji"] for r in rows]))})
                selected = max(grid, key=lambda g: g["val_caicedo_af1"])
                chosen = {k: v for k, v in selected.items() if not k.startswith("val_")}
                to_test = {"selected": chosen}
                if args.locked and not tta:
                    locked_grid = [g for g in grid if g["decoder"] == args.locked]
                    best_locked = max(locked_grid, key=lambda g: g["val_caicedo_af1"])
                    to_test["locked"] = {k: v for k, v in best_locked.items() if not k.startswith("val_")}
                key = "tta8" if tta else "plain"
                record["settings"][key] = {"inference_seconds_100_images": seconds, "validation_grid": grid}
                for label, config in to_test.items():
                    rows = evaluate(pool, test, test_p, config)
                    path = args.output / f"{name}__{key}__{label}_test_per_image.csv"
                    with path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=["id"] + list(rows[0]), lineterminator="\n")
                        writer.writeheader()
                        for case, row in zip(test, rows):
                            writer.writerow({"id": case["id"], **row})
                    record["settings"][key][label] = {
                        "config": config,
                        "test_mean": {m: float(np.mean([r[m] for r in rows])) for m in METRICS}}
                print(json.dumps({"run": run, key: {k: v for k, v in record["settings"][key].items()
                                                    if k != "validation_grid"}}), flush=True)
            (args.output / f"{name}.json").write_text(json.dumps(record, indent=2))
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
