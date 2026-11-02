#!/usr/bin/env python3
"""Ten-seed contact-aware training on the fixed Bilkent NucleusSegData split."""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from breastseg.core import Net, features, sha256
from breastseg.data import load_nucleusseg, load_nucleusseg_extended
from breastseg.instance_metrics import bilkent_metrics, published_metrics
from train_bbbc_contact import contact_instances, contact_loss, full_contour_target, touching_contact_target
from train_public_study import save_json, seed_all, state_digest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = "nucleusseg_contact_10seed"


class Cached(Dataset):
    def __init__(self, cases, augment=False):
        self.cases = cases
        self.augment = augment

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        image, target = case["x"], case["target"]
        if self.augment:
            rotations = int(np.random.randint(4))
            image = torch.rot90(image, rotations, (-2, -1))
            target = torch.rot90(target, rotations, (-2, -1))
            if np.random.random() < .5:
                image, target = image.flip(-1), target.flip(-1)
            if np.random.random() < .5:
                image, target = image.flip(-2), target.flip(-2)
        return image, target, index


def loader(cases, batch_size, augment, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Cached(cases, augment), batch_size=batch_size, shuffle=augment,
        generator=generator if augment else None, num_workers=0,
        pin_memory=True, drop_last=False,
    )


def aggregate_bilkent(rows):
    matched = sum(row["n_one_to_one"] for row in rows)
    references = sum(row["n_reference"] for row in rows)
    predictions = sum(row["n_prediction"] for row in rows)
    nucleus_precision = matched / predictions if predictions else float(references == 0)
    nucleus_recall = matched / references if references else float(predictions == 0)
    nucleus_f1 = (2 * nucleus_precision * nucleus_recall /
                  (nucleus_precision + nucleus_recall)) if nucleus_precision + nucleus_recall else 0.
    pixel_tp = sum(row["pixel_true_positive"] for row in rows)
    reference_pixels = sum(row["n_reference_pixels"] for row in rows)
    prediction_pixels = sum(row["n_prediction_pixels"] for row in rows)
    pixel_precision = pixel_tp / prediction_pixels if prediction_pixels else float(reference_pixels == 0)
    pixel_recall = pixel_tp / reference_pixels if reference_pixels else float(prediction_pixels == 0)
    pixel_f1 = (2 * pixel_precision * pixel_recall /
                (pixel_precision + pixel_recall)) if pixel_precision + pixel_recall else 0.
    return {
        "nucleus_precision": nucleus_precision,
        "nucleus_recall": nucleus_recall,
        "nucleus_f1": nucleus_f1,
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_f1": pixel_f1,
        "n_reference": references,
        "n_prediction": predictions,
        "n_one_to_one": matched,
    }


@torch.inference_mode()
def predict_probabilities(model, data_loader, cases):
    model.eval()
    output = {}
    for image, _, indices in data_loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probabilities = model(image.cuda(non_blocking=True)).float().sigmoid().cpu().numpy()
        for probability, index in zip(probabilities, indices.tolist()):
            case = cases[index]
            height, width = case["instances"].shape
            output[case["id"]] = np.stack([
                cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
                for channel in probability
            ])
    return output


def evaluate_probabilities(probabilities, cases, foreground_threshold,
                           contact_threshold, min_size, output=None):
    if output is not None:
        prediction_dir = output / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        probability = probabilities[case["id"]]
        foreground = probability[0] > foreground_threshold
        contact = probability[1] > contact_threshold
        prediction = contact_instances(foreground, contact, min_size=min_size)
        values = {
            "id": case["id"],
            "subset": case["subset"],
            **published_metrics(case["instances"], prediction),
            **{f"bilkent_{key}": value for key, value in bilkent_metrics(case["instances"], prediction).items()},
        }
        # Keep unprefixed additive components for exact subset-level aggregation.
        for key in ("n_reference", "n_prediction", "n_one_to_one", "pixel_true_positive",
                    "n_reference_pixels", "n_prediction_pixels"):
            values[key] = values[f"bilkent_{key}"]
        rows.append(values)
        if output is not None and not cv2.imwrite(
                str(prediction_dir / f"{case['id']}_instances.tif"), prediction.astype(np.uint16)):
            raise IOError(f"Failed to write prediction for {case['id']}")
    return rows


def run_one(config, model_name, variant, seed, cases, protocol_hash, study_name):
    run_name = f"{model_name}_s{seed}"
    output = ROOT / "results" / study_name / run_name
    checkpoint_dir = ROOT / "checkpoints" / study_name / run_name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous["protocol_sha256"] != protocol_hash:
            raise RuntimeError(f"Protocol differs from completed run: {run_name}")
        print("SKIP completed", run_name, flush=True)
        return
    if (output / "history.json").exists():
        raise RuntimeError(f"Partial run exists: {output}")

    seed_all(seed)
    train = [case for case in cases if case["split"] == "train"]
    test = [case for case in cases if case["split"] == "test"]
    model = Net(variant, pretrained=True, classes=2).cuda()
    initial_hash = state_digest(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["learning_rate"] * .01)
    train_loader = loader(train, config["batch_size"], True, seed)
    start, history = time.time(), []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses, epoch_start = [], time.time()
        for image, target, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = contact_loss(
                    model(image.cuda(non_blocking=True)), target.cuda(non_blocking=True),
                    config["contact_weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append((loss.item(), len(image)))
        record = {
            "epoch": epoch,
            "train_loss": sum(value * count for value, count in losses) / sum(count for _, count in losses),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
        }
        history.append(record)
        save_json(output / "history.json", history)
        save_json(ROOT / "results" / study_name / "status.json", {"state": "training", "run": run_name, **record})
        print(json.dumps({"run": run_name, **record}), flush=True)
        scheduler.step()

    checkpoint_path = checkpoint_dir / "final.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "model": model_name, "variant": variant,
        "seed": seed, "epoch": config["epochs"], "protocol_sha256": protocol_hash,
    }, checkpoint_path)
    train_probabilities = predict_probabilities(
        model, loader(train, config["batch_size"], False, seed), train)
    selection_scores = []
    for contact_threshold in config["contact_thresholds"]:
        for min_size in config["min_sizes"]:
            candidate_rows = evaluate_probabilities(
                train_probabilities, train, config["foreground_threshold"],
                contact_threshold, min_size)
            score = aggregate_bilkent(candidate_rows)
            selection_scores.append({
                "contact_threshold": contact_threshold,
                "min_size": min_size,
                "nucleus_f1": score["nucleus_f1"],
                "pixel_f1": score["pixel_f1"],
            })
    selected_parameters = max(
        selection_scores, key=lambda row: (row["nucleus_f1"], row["pixel_f1"]))
    test_probabilities = predict_probabilities(
        model, loader(test, config["batch_size"], False, seed), test)
    rows = evaluate_probabilities(
        test_probabilities, test, config["foreground_threshold"],
        selected_parameters["contact_threshold"], selected_parameters["min_size"], output)
    with (output / "test_per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    subsets = {}
    for subset in ("Huh7", "HepG2"):
        subset_rows = [row for row in rows if row["subset"] == subset]
        subsets[subset] = {
            "n_images": len(subset_rows),
            "bilkent_micro": aggregate_bilkent(subset_rows),
            "macro_mean": {
                key: float(np.mean([row[key] for row in subset_rows]))
                for key in ("dice1", "aji", "pq", "caicedo_af1", "symmetric_best_dice")
            },
        }
    summary = {
        "model": model_name, "variant": variant, "seed": seed,
        "n_train": len(train), "n_test": len(test), "epochs": config["epochs"],
        "selection": "fixed final epoch; post-processing selected on official training images only",
        "foreground_threshold": config["foreground_threshold"],
        "contact_threshold": selected_parameters["contact_threshold"],
        "min_size": selected_parameters["min_size"],
        "training_selection_score": selected_parameters,
        "training_selection_grid": selection_scores,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_state_sha256": initial_hash, "checkpoint_sha256": sha256(checkpoint_path),
        "protocol_sha256": protocol_hash, "seconds": time.time() - start, "subsets": subsets,
    }
    save_json(summary_path, summary)
    print("COMPLETE " + json.dumps(summary), flush=True)
    del model, optimizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-version", choices=("original", "extended"), default="original")
    parser.add_argument("--study")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    protocol_hash = sha256(args.config)
    study_name = args.study or (DEFAULT_STUDY if args.dataset_version == "original"
                                else "nucleusseg_extended_contact_10seed")
    cv2.setNumThreads(1)
    torch.set_num_threads(8)
    loader_function = load_nucleusseg if args.dataset_version == "original" else load_nucleusseg_extended
    cases = loader_function(args.data)
    for case in cases:
        foreground = case["instances"] > 0
        if config.get("target", "contact") == "contact":
            contact = touching_contact_target(case["instances"], radius=config["contact_radius"])
        elif config["target"] == "full_contour":
            contact = full_contour_target(case["instances"], radius=config["contact_radius"])
        else:
            raise ValueError(config["target"])
        case["target"] = torch.from_numpy(np.stack([
            cv2.resize(channel.astype(np.uint8), (config["image_size"], config["image_size"]),
                       interpolation=cv2.INTER_NEAREST)
            for channel in (foreground, contact)
        ]).astype(np.float32))
    study = ROOT / "results" / study_name
    study.mkdir(parents=True, exist_ok=True)
    manifest = [
        {key: value for key, value in case.items()
         if key not in ("image", "mask", "instances", "x", "target")}
        for case in cases
    ]
    save_json(study / "split_manifest.json", manifest)
    archive_name = "NucleusSegData.zip" if args.dataset_version == "original" else "NucleusSegDataExtended.zip"
    archive = args.data / archive_name
    save_json(study / "environment.json", {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "protocol_sha256": protocol_hash,
        "dataset_version": args.dataset_version,
        "config": config, "archive_sha256": sha256(archive) if archive.exists() else None,
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (Path(__file__), ROOT / "src/breastseg/core.py",
                         ROOT / "src/breastseg/data.py", ROOT / "src/breastseg/instance_metrics.py")
        },
    })
    seeds = args.seeds if args.seeds is not None else config["seeds"]
    for model_name, variant in config["models"].items():
        use_edge = variant in ("gabor_b7", "gabor_boundary_b7")
        for case in cases:
            case["x"] = features(case["image"], edge=use_edge, size=config["image_size"])
        for seed in seeds:
            run_one(config, model_name, variant, seed, cases, protocol_hash, study_name)
    save_json(study / "status.json", {"state": "complete", "protocol_sha256": protocol_hash})


if __name__ == "__main__":
    main()
