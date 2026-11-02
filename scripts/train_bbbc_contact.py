#!/usr/bin/env python3
"""Train a foreground model with an auxiliary touching-contact boundary head."""
import argparse
import csv
import json
import os
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from breastseg.core import Net, features, sha256, soft_dice
from breastseg.data import load_bbbc039
from breastseg.instance_metrics import bbbc039_reference_instances, published_metrics, remap_labels
from train_public_study import save_json, seed_all, state_digest


ROOT = Path(__file__).resolve().parents[1]
STUDY = os.environ.get("BREASTSEG_BBBC_CONTACT_STUDY", "bbbc_contact_10seed")


def touching_contact_target(instances, radius=1):
    """Mark only interfaces between adjacent non-zero instance IDs."""
    labels = np.asarray(instances)
    if labels.ndim != 2 or (labels < 0).any():
        raise ValueError("Expected a nonnegative 2D instance map")
    contact = np.zeros(labels.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        source_y = slice(max(0, -dy), labels.shape[0] - max(0, dy))
        source_x = slice(max(0, -dx), labels.shape[1] - max(0, dx))
        target_y = slice(max(0, dy), labels.shape[0] - max(0, -dy))
        target_x = slice(max(0, dx), labels.shape[1] - max(0, -dx))
        first, second = labels[source_y, source_x], labels[target_y, target_x]
        differs = (first > 0) & (second > 0) & (first != second)
        contact[source_y, source_x] |= differs
        contact[target_y, target_x] |= differs
    if radius:
        contact = morphology.dilation(contact, morphology.disk(radius))
    return contact & (labels > 0)


def full_contour_target(instances, radius=1):
    """Control target: every nucleus contour, not only nucleus-nucleus interfaces."""
    labels = np.asarray(instances)
    contour = segmentation.find_boundaries(labels, mode="inner", connectivity=1)
    if radius:
        contour = morphology.dilation(contour, morphology.disk(radius))
    return contour & (labels > 0)


def instance_annotations(root, case_ids=None):
    output = {}
    with zipfile.ZipFile(Path(root) / "masks.zip") as archive:
        for name in archive.namelist():
            if (name.startswith("masks/") and name.endswith(".png")
                    and (case_ids is None or Path(name).stem in case_ids)):
                raw = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_UNCHANGED)
                output[Path(name).stem] = bbbc039_reference_instances(raw)
    return output


CROP = None  # set from config["train_crop"] when training at native resolution
UNION_DECODE = False  # carved-foreground control: decode foreground as (fg OR contact)
SEMANTIC = False  # fair semantic baseline: one head, watershed decoding, selected by validation object F1
SEMANTIC_WS = (0.5, 2.0)  # (h, sigma) used during checkpoint selection
INTENSITY_JITTER = False
NATIVE = False


def pad_reflect(tensor, shape):
    height, width = tensor.shape[-2:]
    return F.pad(tensor[None], (0, shape[1] - width, 0, shape[0] - height), mode="reflect")[0]


class Cached(Dataset):
    def __init__(self, cases, edge, augment=False):
        self.cases, self.edge, self.augment = cases, edge, augment

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        x = case["x4"] if self.edge else case["x3"]
        target = case["target"]
        if self.augment and CROP:
            y0 = int(np.random.randint(x.shape[-2] - CROP + 1))
            x0 = int(np.random.randint(x.shape[-1] - CROP + 1))
            x, target = x[:, y0:y0 + CROP, x0:x0 + CROP], target[:, y0:y0 + CROP, x0:x0 + CROP]
        if self.augment:
            k = int(np.random.randint(4))
            x, target = torch.rot90(x, k, (-2, -1)), torch.rot90(target, k, (-2, -1))
            if np.random.random() < .5:
                x, target = x.flip(-1), target.flip(-1)
            if INTENSITY_JITTER:
                if self.edge:
                    raise ValueError("Intensity jitter currently requires a three-channel model")
                mean = x.new_tensor([.485, .456, .406])[:, None, None]
                std = x.new_tensor([.229, .224, .225])[:, None, None]
                raw = (x * std + mean).clamp(0, 1)
                raw = (raw.pow(float(np.random.uniform(.8, 1.2))) *
                       float(np.random.uniform(.85, 1.15))).clamp(0, 1)
                x = (raw - mean) / std
        return x, target, index


def contact_loss(logits, target, contact_weight, focal_gamma=0.0):
    logits, target = logits.float(), target.float()
    foreground_loss = .5 * soft_dice(logits[:, :1].sigmoid(), target[:, :1]) + \
        .5 * F.binary_cross_entropy_with_logits(logits[:, :1], target[:, :1])
    contact = target[:, 1:2]
    ratio = contact.mean()
    positive_weight = torch.where(
        ratio > 0,
        ((1 - ratio) / ratio.clamp_min(1e-7)).clamp(1, 50),
        torch.ones_like(ratio),
    )
    bce = F.binary_cross_entropy_with_logits(
        logits[:, 1:2], contact, pos_weight=positive_weight, reduction="none")
    if focal_gamma:
        p = logits[:, 1:2].sigmoid()
        pt = contact * p + (1 - contact) * (1 - p)
        bce = bce * (1 - pt).pow(focal_gamma)
    boundary_loss = .5 * soft_dice(logits[:, 1:2].sigmoid(), contact) + .5 * bce.mean()
    loss = foreground_loss + contact_weight * boundary_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite contact-head loss")
    return loss


def contact_instances(foreground, contact, min_size=16):
    foreground = morphology.remove_small_holes(np.asarray(foreground, bool), max_size=min_size - 1)
    foreground = morphology.remove_small_objects(foreground, max_size=min_size - 1)
    markers_binary = morphology.remove_small_objects(foreground & ~np.asarray(contact, bool), max_size=min_size - 1)
    markers = remap_labels(measure.label(markers_binary, connectivity=1))
    if markers.max() == 0:
        return remap_labels(measure.label(foreground, connectivity=1))
    _, indices = ndi.distance_transform_edt(markers == 0, return_indices=True)
    output = markers[tuple(indices)]
    output[~foreground] = 0
    return remap_labels(output)


def loader(cases, edge, batch_size, augment, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Cached(cases, edge, augment), batch_size=batch_size, shuffle=augment,
                      generator=generator if augment else None, num_workers=0, pin_memory=True)


@torch.inference_mode()
def predict_probabilities(model, data_loader, cases):
    model.eval()
    output = {}
    for x, _, indices in data_loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = model(x.cuda(non_blocking=True)).float().sigmoid().cpu().numpy()
        for probability, index in zip(values, indices.tolist()):
            case = cases[index]
            height, width = case["mask"].shape
            if NATIVE or CROP:
                output[case["id"]] = probability[:, :height, :width].copy()
                continue
            output[case["id"]] = np.stack([
                cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
                for channel in probability
            ])
    return output


def evaluate(probabilities, cases, contact_threshold=.5, output=None, decoder=None):
    rows = []
    if output is not None:
        (output / "predictions").mkdir(parents=True, exist_ok=True)
    for case in cases:
        probability = probabilities[case["id"]]
        if SEMANTIC:
            from posthoc_bbbc_decoding import semantic_instances
            foreground = probability[0] > .5
            instances = semantic_instances(foreground, "watershed", *SEMANTIC_WS)
            rows.append({"id": case["id"], **published_metrics(case["instances"], instances)})
            if output is not None:
                cv2.imwrite(str(output / "predictions" / f"{case['id']}_foreground.png"), foreground.astype(np.uint8) * 255)
                cv2.imwrite(str(output / "predictions" / f"{case['id']}_instances.tif"), instances.astype(np.uint16))
            continue
        foreground, contact = probability[0] > .5, probability[1] > contact_threshold
        if UNION_DECODE:
            foreground = foreground | contact
        if decoder is None:
            instances = contact_instances(foreground, contact)
        else:
            from astra_utils import decode
            instances = decode(probability, {**decoder, "threshold": contact_threshold})
        rows.append({"id": case["id"], **published_metrics(case["instances"], instances)})
        if output is not None:
            cv2.imwrite(str(output / "predictions" / f"{case['id']}_foreground.png"), foreground.astype(np.uint8) * 255)
            cv2.imwrite(str(output / "predictions" / f"{case['id']}_contact.png"), contact.astype(np.uint8) * 255)
            cv2.imwrite(str(output / "predictions" / f"{case['id']}_instances.tif"), instances.astype(np.uint16))
    return rows


def run_one(config, name, variant, seed, cases, protocol_hash):
    training_code_sha256 = None
    if config.get("astra_study"):
        from astra_utils import code_digest
        training_code_sha256 = code_digest()
    run_name = f"{name}_s{seed}"
    output = ROOT / "results" / STUDY / run_name
    checkpoint_dir = ROOT / "checkpoints" / STUDY / run_name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        previous = json.loads((output / "summary.json").read_text())
        if previous["protocol_sha256"] != protocol_hash:
            raise RuntimeError(f"Protocol differs from completed run: {run_name}")
        if training_code_sha256 and previous.get("training_code_sha256") != training_code_sha256:
            raise RuntimeError(f"Training code differs from completed run: {run_name}")
        print("SKIP completed", run_name, flush=True)
        return
    if (output / "history.json").exists():
        raise RuntimeError(f"Partial run exists: {output}")

    seed_all(seed)
    train = [case for case in cases if case["split"] == "train"]
    validation = [case for case in cases if case["split"] == "val"]
    test = [case for case in cases if case["split"] == "test"]
    model = Net(variant, pretrained=True, classes=1 if SEMANTIC else 2).cuda()
    initial_hash = state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["learning_rate"] * .01)
    train_loader = loader(train, model.use_edge, config["batch_size"], True, seed)
    validation_loader = loader(validation, model.use_edge, config["batch_size"], False, seed)
    start, best, best_epoch, history = time.time(), -1., None, []
    for epoch in range(1, config["epochs"] + 1):
        model.train(); losses = []; epoch_start = time.time()
        for x, target, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.cuda(non_blocking=True)); target = target.cuda(non_blocking=True)
                if SEMANTIC:
                    loss = .5 * soft_dice(logits[:, :1].float().sigmoid(), target[:, :1]) + \
                        .5 * F.binary_cross_entropy_with_logits(logits[:, :1].float(), target[:, :1])
                else:
                    loss = contact_loss(logits, target, config["contact_weight"],
                                        config.get("contact_focal_gamma", 0.0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step(); losses.append((loss.item(), len(x)))
        probabilities = predict_probabilities(model, validation_loader, validation)
        rows = evaluate(probabilities, validation, contact_threshold=.5,
                        decoder=config.get("checkpoint_decoder"))
        score = float(np.mean([row["caicedo_af1"] for row in rows]))
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model_state_dict": model.state_dict(), "model": name, "variant": variant,
                        "seed": seed, "epoch": epoch, "val_caicedo_af1": best,
                        "training_code_sha256": training_code_sha256,
                        "protocol_sha256": protocol_hash}, checkpoint_dir / "best.pt")
        record = {"epoch": epoch, "train_loss": sum(v*n for v,n in losses)/sum(n for _,n in losses),
                  "val_caicedo_af1": score, "best_val_caicedo_af1": best,
                  "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.time()-epoch_start}
        history.append(record); save_json(output / "history.json", history)
        save_json(ROOT / "results" / STUDY / "status.json", {"state": "training", "run": run_name, **record})
        print(json.dumps({"run": run_name, **record}), flush=True)
        scheduler.step()

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    validation_probabilities = predict_probabilities(model, validation_loader, validation)
    candidates = config["contact_thresholds"]
    threshold_scores = {}
    for threshold in candidates:
        rows = evaluate(validation_probabilities, validation, contact_threshold=threshold,
                        decoder=config.get("checkpoint_decoder"))
        threshold_scores[str(threshold)] = float(np.mean([row["caicedo_af1"] for row in rows]))
    selected_threshold = max(candidates, key=lambda value: threshold_scores[str(value)])
    validation_rows = evaluate(validation_probabilities, validation, selected_threshold,
                               decoder=config.get("checkpoint_decoder"))
    validation_mean = {key: float(np.mean([row[key] for row in validation_rows]))
                       for key in validation_rows[0] if key != "id"}
    if config.get("validation_only", False):
        save_json(output / "summary.json", {
            "model": name, "variant": variant, "seed": seed, "validation_only": True,
            "epochs": config["epochs"], "best_epoch": best_epoch, "best_val_caicedo_af1": best,
            "selected_contact_threshold": selected_threshold, "validation_threshold_scores": threshold_scores,
            "validation_mean": validation_mean,
            "training_code_sha256": training_code_sha256,
            "protocol_sha256": protocol_hash, "seconds": time.time() - start})
        print("VALIDATION-ONLY COMPLETE", run_name, threshold_scores, flush=True)
        del model, optimizer; torch.cuda.empty_cache()
        return
    test_probabilities = predict_probabilities(
        model, loader(test, model.use_edge, config["batch_size"], False, seed), test)
    locked_decoder = config.get("locked_decoder")
    rows = evaluate(test_probabilities, test,
                    contact_threshold=locked_decoder["threshold"] if locked_decoder else selected_threshold,
                    output=output, decoder=locked_decoder or config.get("checkpoint_decoder"))
    with (output / "test_per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    metric_keys = [key for key in rows[0] if key not in ("id", "n_reference", "n_prediction")]
    summary = {"model": name, "variant": variant, "seed": seed, "n_test": 50,
               "epochs": config["epochs"], "best_epoch": best_epoch, "best_val_caicedo_af1": best,
               "selected_contact_threshold": locked_decoder["threshold"] if locked_decoder else selected_threshold,
               "validation_selected_contact_threshold": selected_threshold,
               "validation_threshold_scores": threshold_scores,
               "locked_decoder": locked_decoder, "validation_mean": validation_mean,
               "training_code_sha256": training_code_sha256,
               "parameters": sum(p.numel() for p in model.parameters()), "initial_state_sha256": initial_hash,
               "checkpoint_sha256": sha256(checkpoint_dir / "best.pt"), "protocol_sha256": protocol_hash,
               "seconds": time.time()-start,
               "mean": {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}}
    save_json(output / "summary.json", summary); print("COMPLETE " + json.dumps(summary), flush=True)
    del model, optimizer; torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args(); config = json.loads(args.config.read_text()); protocol_hash = sha256(args.config)
    if config.get("astra_study") and STUDY != config["astra_study"]:
        raise ValueError("Astra configs require their dedicated BREASTSEG_BBBC_CONTACT_STUDY namespace")
    cv2.setNumThreads(1); torch.set_num_threads(8)
    splits = ("train", "val") if config.get("validation_only", False) else ("train", "val", "test")
    cases = load_bbbc039(args.data, splits)
    annotations = instance_annotations(args.data, {case["id"] for case in cases})
    global CROP, UNION_DECODE, INTENSITY_JITTER, NATIVE
    native = config.get("native_resolution", False)
    NATIVE = native
    CROP = None
    INTENSITY_JITTER = config.get("intensity_jitter", False)
    UNION_DECODE = config.get("target", "contact") == "contact_carved"
    global SEMANTIC
    SEMANTIC = config.get("target", "contact") == "semantic"
    if native:
        CROP = config["train_crop"]
        pad_shape = tuple(config["pad_to"])
    for case in cases:
        if native:
            case["x3"] = pad_reflect(features(case["image"], edge=False, native=True), pad_shape)
            case["x4"] = pad_reflect(features(case["image"], edge=True, native=True), pad_shape)
        else:
            case["x3"] = features(case["image"], edge=False, size=config["image_size"])
            case["x4"] = features(case["image"], edge=True, size=config["image_size"])
        case["instances"] = annotations[case["id"]]
        foreground = case["instances"] > 0
        if config.get("target", "contact") == "contact":
            contact = touching_contact_target(case["instances"], radius=config["contact_radius"])
        elif config["target"] == "contact_carved":
            # Closest prior art (2018 DSB winners): contacts are removed from the foreground target.
            contact = touching_contact_target(case["instances"], radius=config["contact_radius"])
            foreground = foreground & ~contact
        elif config["target"] == "semantic":
            contact = touching_contact_target(case["instances"], radius=config["contact_radius"])  # unused
        elif config["target"] == "full_contour":
            contact = full_contour_target(case["instances"], radius=config["contact_radius"])
        else:
            raise ValueError(config["target"])
        target = np.stack([foreground, contact]).astype(np.uint8)
        if native:
            case["target"] = pad_reflect(torch.from_numpy(target.astype(np.float32)), pad_shape)
            continue
        case["target"] = torch.from_numpy(np.stack([
            cv2.resize(channel, (config["image_size"], config["image_size"]), interpolation=cv2.INTER_NEAREST)
            for channel in target
        ]).astype(np.float32))
    study = ROOT / "results" / STUDY; study.mkdir(parents=True, exist_ok=True)
    save_json(study / "environment.json", {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "protocol_sha256": protocol_hash, "config": config})
    models = args.models or list(config["models"]); seeds = args.seeds if args.seeds is not None else config["seeds"]
    for seed in seeds:
        for name in models:
            run_one(config, name, config["models"][name], seed, cases, protocol_hash)
    save_json(study / "status.json", {"state": "complete", "protocol_sha256": protocol_hash})


if __name__ == "__main__":
    main()
