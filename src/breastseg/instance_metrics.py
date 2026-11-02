"""Published BBBC039 instance metrics with explicit post-processing contracts.

The definitions follow the public Caicedo et al. and ASW-Net evaluation code.
They are kept here (rather than importing either archived repository) so the
exact behavior is tested under the current Python environment.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation


def remap_labels(labels: np.ndarray) -> np.ndarray:
    """Map non-zero labels to contiguous positive integers."""
    labels = np.asarray(labels)
    out = np.zeros(labels.shape, np.int32)
    for new, old in enumerate(np.unique(labels)[1:], 1):
        out[labels == old] = new
    return out


def bbbc039_reference_instances(mask_bgra: np.ndarray, min_size: int = 16) -> np.ndarray:
    """Reproduce the published BBBC039 raw-annotation conversion.

    BBBC039 PNGs encode background/interior/boundary classes in the red
    channel.  The ASW-Net/Caicedo evaluation reads that channel, removes small
    labeled regions, and calls connected-component labeling on the class map.
    OpenCV returns BGRA, hence index 2 below.
    """
    mask_bgra = np.asarray(mask_bgra)
    if mask_bgra.ndim != 3 or mask_bgra.shape[2] < 3:
        raise ValueError("Expected a BBBC039 color annotation")
    classes = mask_bgra[..., 2]
    cleaned = classes.copy()
    for value in np.unique(cleaned)[1:]:
        if np.count_nonzero(cleaned == value) < min_size:
            cleaned[cleaned == value] = 0
    return remap_labels(measure.label(cleaned, connectivity=1))


def semantic_to_instances(mask: np.ndarray, method: str = "components",
                          min_size: int = 16, h: float = 2.0) -> np.ndarray:
    """Convert binary semantic output to instances using a declared rule."""
    binary = morphology.remove_small_holes(np.asarray(mask, bool), max_size=min_size - 1)
    binary = morphology.remove_small_objects(binary, max_size=min_size - 1)
    if not binary.any():
        return np.zeros(binary.shape, np.int32)
    if method == "components":
        return remap_labels(measure.label(binary, connectivity=1))
    if method != "watershed":
        raise ValueError(f"Unknown instance post-processing method: {method}")
    distance = ndi.distance_transform_edt(binary)
    maxima = morphology.h_maxima(distance, h)
    markers = measure.label(maxima, connectivity=1)
    if markers.max() == 0:
        markers = measure.label(binary, connectivity=1)
    labels = segmentation.watershed(-distance, markers, mask=binary, watershed_line=True)
    return remap_labels(labels)


def classmap_to_instances(classes: np.ndarray, expand_boundary: bool = False,
                          min_size: int = 16) -> np.ndarray:
    """Convert background/interior/boundary classes to instance labels.

    The non-expanded path matches Caicedo's ``pred_to_label`` convention by
    labeling only class 1 (interior). The expanded path assigns predicted
    boundary pixels to their nearest interior marker without joining markers.
    """
    classes = np.asarray(classes)
    if classes.ndim != 2 or not np.isin(classes, (0, 1, 2)).all():
        raise ValueError("Expected a 2D class map with values 0, 1, and 2")
    interiors = morphology.remove_small_holes(classes == 1, max_size=min_size - 1)
    interiors = morphology.remove_small_objects(interiors, max_size=min_size - 1)
    markers = remap_labels(measure.label(interiors, connectivity=1))
    if not expand_boundary or markers.max() == 0:
        return markers
    foreground = classes > 0
    _, indices = ndi.distance_transform_edt(markers == 0, return_indices=True)
    expanded = markers[tuple(indices)]
    expanded[~foreground] = 0
    return remap_labels(expanded)


def _pairwise(true: np.ndarray, pred: np.ndarray):
    true, pred = remap_labels(true), remap_labels(pred)
    nt, npred = int(true.max()), int(pred.max())
    intersections = np.zeros((nt, npred), np.float64)
    true_areas = np.bincount(true.ravel(), minlength=nt + 1)[1:].astype(np.float64)
    pred_areas = np.bincount(pred.ravel(), minlength=npred + 1)[1:].astype(np.float64)
    overlap = (true > 0) & (pred > 0)
    if overlap.any() and nt and npred:
        pairs = (true[overlap] - 1) * npred + (pred[overlap] - 1)
        intersections = np.bincount(pairs, minlength=nt * npred).reshape(nt, npred).astype(np.float64)
    unions = true_areas[:, None] + pred_areas[None, :] - intersections
    iou = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    return intersections, unions, iou, true_areas, pred_areas


def foreground_dice(true: np.ndarray, pred: np.ndarray) -> float:
    true, pred = np.asarray(true) > 0, np.asarray(pred) > 0
    denominator = true.sum() + pred.sum()
    return float(2 * np.logical_and(true, pred).sum() / denominator) if denominator else 1.0


def ensemble_dice(true: np.ndarray, pred: np.ndarray) -> float:
    """DICE2 as implemented by ASW-Net."""
    intersections, _, _, true_areas, pred_areas = _pairwise(true, pred)
    rows, cols = np.nonzero(intersections)
    if not len(rows):
        return 1.0 if not true_areas.size and not pred_areas.size else 0.0
    total = np.sum(true_areas[rows] + pred_areas[cols])
    return float(2 * intersections[rows, cols].sum() / total) if total else 0.0


def aggregated_jaccard(true: np.ndarray, pred: np.ndarray) -> float:
    """AJI with the same per-reference best-overlap pairing used by ASW-Net."""
    intersections, unions, iou, true_areas, pred_areas = _pairwise(true, pred)
    if not true_areas.size:
        return 1.0 if not pred_areas.size else 0.0
    if not pred_areas.size:
        return 0.0
    best = np.argmax(iou, axis=1)
    valid = iou[np.arange(len(best)), best] > 0
    rows, cols = np.nonzero(valid)[0], best[valid]
    overall_intersection = intersections[rows, cols].sum()
    overall_union = unions[rows, cols].sum()
    # Match the distributed ASW-Net/MoNuSeg code: every reference object
    # without any overlapping prediction contributes its full area to the
    # union.  Without this term, a complete miss would not be penalized.
    overall_union += true_areas[~valid].sum()
    used = set(cols.tolist())
    overall_union += pred_areas[[i for i in range(len(pred_areas)) if i not in used]].sum()
    return float(overall_intersection / overall_union) if overall_union else 0.0


def panoptic_quality(true: np.ndarray, pred: np.ndarray, threshold: float = 0.5):
    """Return detection quality, segmentation quality, and PQ at IoU > 0.5."""
    _, _, iou, true_areas, pred_areas = _pairwise(true, pred)
    matched = iou > threshold
    rows, cols = np.nonzero(matched)
    tp = len(rows)
    fp, fn = len(pred_areas) - tp, len(true_areas) - tp
    denominator = tp + .5 * fp + .5 * fn
    dq = float(tp / denominator) if denominator else 1.0
    sq = float(iou[rows, cols].sum() / tp) if tp else (1.0 if denominator == 0 else 0.0)
    return dq, sq, dq * sq


def average_object_f1(true: np.ndarray, pred: np.ndarray):
    """Caicedo average object F1 over IoU thresholds 0.50:0.05:0.90."""
    _, _, iou, true_areas, pred_areas = _pairwise(true, pred)
    values = []
    for threshold in np.arange(.5, .95, .05):
        matches = iou > threshold
        tp = int((matches.sum(axis=1) == 1).sum())
        fp = int((matches.sum(axis=0) == 0).sum())
        fn = int((matches.sum(axis=1) == 0).sum())
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 1.0)
    return float(np.mean(values)), values


def symmetric_best_dice(true: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric best Dice (SBD), used by biological instance benchmarks."""
    intersections, _, _, true_areas, pred_areas = _pairwise(true, pred)
    if not true_areas.size and not pred_areas.size:
        return 1.0
    if not true_areas.size or not pred_areas.size:
        return 0.0
    dice = 2 * intersections / (true_areas[:, None] + pred_areas[None, :])
    reference_to_prediction = float(dice.max(axis=1).mean())
    prediction_to_reference = float(dice.max(axis=0).mean())
    return min(reference_to_prediction, prediction_to_reference)


def bilkent_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Nucleus- and pixel-level scores used by Arslan and Koyuncu et al.

    Each predicted object is assigned to the reference object containing at
    least half of its area, and vice versa.  A true-positive nucleus is a
    mutual, unique assignment.  Pixel true positives are intersections of
    those one-to-one pairs; all predicted/reference foreground contributes to
    the corresponding precision/recall denominator.
    """
    intersections, _, _, true_areas, pred_areas = _pairwise(true, pred)
    n_true, n_pred = len(true_areas), len(pred_areas)
    if n_true and n_pred:
        true_fraction = intersections / true_areas[:, None]
        pred_fraction = intersections / pred_areas[None, :]
        true_to_pred = np.argmax(true_fraction, axis=1)
        pred_to_true = np.argmax(pred_fraction, axis=0)
        pairs = [
            (true_index, pred_index)
            for true_index, pred_index in enumerate(true_to_pred)
            if true_fraction[true_index, pred_index] >= .5
            and pred_fraction[true_index, pred_index] >= .5
            and pred_to_true[pred_index] == true_index
        ]
    else:
        pairs = []
    true_positives = len(pairs)
    nucleus_precision = true_positives / n_pred if n_pred else float(n_true == 0)
    nucleus_recall = true_positives / n_true if n_true else float(n_pred == 0)
    nucleus_f1 = (2 * nucleus_precision * nucleus_recall /
                  (nucleus_precision + nucleus_recall)) if nucleus_precision + nucleus_recall else 0.0
    pixel_tp = float(sum(intersections[i, j] for i, j in pairs))
    predicted_pixels, reference_pixels = pred_areas.sum(), true_areas.sum()
    pixel_precision = pixel_tp / predicted_pixels if predicted_pixels else float(reference_pixels == 0)
    pixel_recall = pixel_tp / reference_pixels if reference_pixels else float(predicted_pixels == 0)
    pixel_f1 = (2 * pixel_precision * pixel_recall /
                (pixel_precision + pixel_recall)) if pixel_precision + pixel_recall else 0.0
    return {
        "nucleus_precision": float(nucleus_precision),
        "nucleus_recall": float(nucleus_recall),
        "nucleus_f1": float(nucleus_f1),
        "pixel_precision": float(pixel_precision),
        "pixel_recall": float(pixel_recall),
        "pixel_f1": float(pixel_f1),
        "n_reference": n_true,
        "n_prediction": n_pred,
        "n_one_to_one": true_positives,
        "pixel_true_positive": int(pixel_tp),
        "n_reference_pixels": int(reference_pixels),
        "n_prediction_pixels": int(predicted_pixels),
    }


def published_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    dq, sq, pq = panoptic_quality(true, pred)
    af1, _ = average_object_f1(true, pred)
    return {
        "dice1": foreground_dice(true, pred),
        "dice2": ensemble_dice(true, pred),
        "aji": aggregated_jaccard(true, pred),
        "dq": dq,
        "sq": sq,
        "pq": pq,
        "caicedo_af1": af1,
        "symmetric_best_dice": symmetric_best_dice(true, pred),
        "n_reference": int(np.max(true)),
        "n_prediction": int(np.max(pred)),
    }
