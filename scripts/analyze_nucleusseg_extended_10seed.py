#!/usr/bin/env python3
"""Aggregate the expanded NucleusSegData study and compare AttentionBoost."""
import csv
import json
import statistics
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, wilcoxon
from skimage.segmentation import find_boundaries

from breastseg.data import load_nucleusseg_extended


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/nucleusseg_extended_contact_10seed"
FIGURE_DIR = ROOT / "results/nucleusseg_extended_figures"
MODELS = ("standard_contact_b7", "gabor_contact_b7")
SUBSETS = ("Huh7", "HepG2")
PUBLISHED_ATTENTIONBOOST = {
    "Huh7": {"nucleus_f1_mean": 0.954, "run_sd": 0.005, "n_runs": 5},
    "HepG2": {"nucleus_f1_mean": 0.919, "run_sd": 0.008, "n_runs": 5},
}


def summarize(values):
    mean = statistics.mean(values)
    seed_sd = statistics.stdev(values)
    half_width = float(t.ppf(.975, len(values) - 1) * seed_sd / np.sqrt(len(values)))
    return {
        "mean": mean,
        "seed_sd": seed_sd,
        "mean_95ci": [mean - half_width, mean + half_width],
        "minimum": min(values),
        "maximum": max(values),
    }


def normalized_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    low, high = np.percentile(gray, (1, 99.5))
    return np.clip((gray - low) / (high - low + 1e-7), 0, 1)


def boundary_overlay(image, labels, color):
    output = np.repeat(normalized_gray(image)[..., None], 3, axis=2)
    boundary = find_boundaries(labels, mode="inner")
    boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    output[boundary] = color
    return output


def read_rows(model, seed):
    path = STUDY / f"{model}_s{seed}" / "test_per_image.csv"
    with path.open() as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def make_example_figure(runs):
    standard_scores = {
        summary["seed"]: statistics.mean(
            summary["subsets"][subset]["bilkent_micro"]["nucleus_f1"]
            for subset in SUBSETS)
        for summary in runs["standard_contact_b7"]
    }
    ordered_seeds = sorted(standard_scores, key=standard_scores.get)
    display_seed = ordered_seeds[len(ordered_seeds) // 2]
    standard_rows = read_rows("standard_contact_b7", display_seed)
    gabor_rows = read_rows("gabor_contact_b7", display_seed)
    cases = {case["id"]: case for case in load_nucleusseg_extended(
        ROOT / "data/nucleusseg_extended") if case["split"] == "test"}

    selected = {}
    for subset in SUBSETS:
        candidates = [row for row in standard_rows.values() if row["subset"] == subset]
        median = float(np.median([float(row["bilkent_nucleus_f1"]) for row in candidates]))
        selected[subset] = min(
            candidates, key=lambda row: abs(float(row["bilkent_nucleus_f1"]) - median))["id"]

    figure, axes = plt.subplots(2, 4, figsize=(7.1, 4.15), constrained_layout=True)
    for row_index, subset in enumerate(SUBSETS):
        identifier = selected[subset]
        case = cases[identifier]
        predictions = {}
        for model in MODELS:
            path = STUDY / f"{model}_s{display_seed}" / "predictions" / f"{identifier}_instances.tif"
            predictions[model] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if predictions[model] is None:
                raise RuntimeError(f"Unreadable prediction: {path}")
        panels = (
            (normalized_gray(case["image"]), f"({chr(97 + 4 * row_index)}) {subset} input", "gray"),
            (boundary_overlay(case["image"], case["instances"], (1.0, .15, .05)),
             f"({chr(98 + 4 * row_index)}) reference", None),
            (boundary_overlay(case["image"], predictions["standard_contact_b7"], (.05, .9, .25)),
             f"({chr(99 + 4 * row_index)}) Standard F1={float(standard_rows[identifier]['bilkent_nucleus_f1']):.3f}", None),
            (boundary_overlay(case["image"], predictions["gabor_contact_b7"], (1.0, .7, .05)),
             f"({chr(100 + 4 * row_index)}) Gabor F1={float(gabor_rows[identifier]['bilkent_nucleus_f1']):.3f}", None),
        )
        for axis, (panel, title, cmap) in zip(axes[row_index], panels):
            axis.imshow(panel, cmap=cmap, vmin=0, vmax=1)
            axis.set_title(title, fontsize=7.2)
            axis.axis("off")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / "nucleusseg_extended_examples.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "nucleusseg_extended_examples.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"display_seed": display_seed, "cases": selected,
            "selection": "median standard seed; within-subset median standard per-image nucleus F1"}


def main():
    runs = {}
    for model in MODELS:
        runs[model] = []
        for seed in range(10):
            path = STUDY / f"{model}_s{seed}" / "summary.json"
            if not path.exists():
                raise RuntimeError(f"Missing completed run: {path}")
            runs[model].append(json.loads(path.read_text()))

    output = {
        "protocol": "Official expanded 25-train/11-Huh7-test/25-HepG2-test split",
        "n_seeds": 10,
        "published_attentionboost": PUBLISHED_ATTENTIONBOOST,
        "models": {},
    }
    metric_names = ("nucleus_precision", "nucleus_recall", "nucleus_f1",
                    "pixel_precision", "pixel_recall", "pixel_f1")
    for model, summaries in runs.items():
        output["models"][model] = {}
        for subset in SUBSETS:
            output["models"][model][subset] = {
                metric: summarize([
                    summary["subsets"][subset]["bilkent_micro"][metric]
                    for summary in summaries
                ])
                for metric in metric_names
            }
            ours = output["models"][model][subset]["nucleus_f1"]["mean"]
            published = PUBLISHED_ATTENTIONBOOST[subset]["nucleus_f1_mean"]
            output["models"][model][subset]["nucleus_f1_vs_attentionboost_pp"] = (
                100 * (ours - published)
            )

    output["paired_gabor_minus_standard"] = {}
    for subset in SUBSETS:
        standard_values = [
            summary["subsets"][subset]["bilkent_micro"]["nucleus_f1"]
            for summary in runs["standard_contact_b7"]]
        gabor_values = [
            summary["subsets"][subset]["bilkent_micro"]["nucleus_f1"]
            for summary in runs["gabor_contact_b7"]]
        deltas = [
            gabor - standard for standard, gabor in zip(standard_values, gabor_values)
        ]
        test = wilcoxon(standard_values, gabor_values, alternative="two-sided", method="exact")
        output["paired_gabor_minus_standard"][subset] = {
            **summarize(deltas),
            "gabor_seed_wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "paired_two_sided_wilcoxon_statistic": float(test.statistic),
            "paired_two_sided_wilcoxon_p": float(test.pvalue),
        }

    output["qualitative_figure"] = make_example_figure(runs)

    (STUDY / "aggregate.json").write_text(json.dumps(output, indent=2) + "\n")
    with (STUDY / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "subset", "metric", "mean", "seed_sd", "minimum", "maximum"))
        for model in MODELS:
            for subset in SUBSETS:
                for metric in metric_names:
                    values = output["models"][model][subset][metric]
                    writer.writerow((model, subset, metric, values["mean"], values["seed_sd"],
                                     values["minimum"], values["maximum"]))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
