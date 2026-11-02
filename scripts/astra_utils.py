"""Opt-in contact decoders and validation diagnostics for the Astra study."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation

from breastseg.instance_metrics import _pairwise, remap_labels

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('baseline40', 'long200', 'lowlr200', 'focal200', 'radius0_200', 'intensity200')
METRICS = ('caicedo_af1', 'aji', 'pq', 'dq', 'sq', 'dice1')


def config_path(arm):
    if arm not in ARMS:
        raise ValueError(f'Unknown Astra arm: {arm}')
    return ROOT / 'configs' / f'bbbc_astra_{arm}.json'


def study_name(arm):
    config_path(arm)
    return f'bbbc_astra_search_{arm}'


def code_digest():
    names = ('scripts/train_bbbc_contact.py', 'scripts/astra_utils.py',
             'scripts/posthoc_astra.py', 'scripts/summarize_astra_search.py',
             'scripts/prepare_astra_final.py', 'src/breastseg/core.py',
             'src/breastseg/data.py', 'src/breastseg/instance_metrics.py',
             'scripts/posthoc_bbbc_decoding.py', 'scripts/train_public_study.py')
    return hashlib.sha256(b''.join((ROOT / n).read_bytes() for n in names)).hexdigest()


def decode(probability, config):
    kind = config['decoder']
    threshold = config.get('threshold', .5)
    min_size = config.get('min_size', 16)
    foreground = probability[0] > config.get('foreground_threshold', .5)
    if kind == 'nearest_marker':
        from train_bbbc_contact import contact_instances
        return contact_instances(foreground, probability[1] > threshold, min_size)
    if kind not in ('component_nearest', 'combined_watershed'):
        raise ValueError(kind)
    foreground = morphology.remove_small_holes(foreground, max_size=min_size - 1)
    foreground = morphology.remove_small_objects(foreground, max_size=min_size - 1)
    marker_mask = morphology.remove_small_objects(
        foreground & (probability[1] <= threshold),
        max_size=config.get('marker_min_size', 16) - 1)
    components = measure.label(foreground, connectivity=1)
    out = np.zeros(foreground.shape, np.int32)
    next_id = 1
    elevation = (1 - probability[0]) + config.get('contact_scale', 1.) * probability[1]
    for region in measure.regionprops(components):
        sl = region.slice
        inside = components[sl] == region.label
        markers = measure.label(marker_mask[sl] & inside, connectivity=1)
        if not markers.max():
            out[sl][inside] = next_id
            next_id += 1
            continue
        if kind == 'component_nearest':
            _, idx = ndi.distance_transform_edt(markers == 0, return_indices=True)
            labels = markers[tuple(idx)]
        else:
            labels = segmentation.watershed(elevation[sl], markers, mask=inside)
        out[sl][inside] = labels[inside] + next_id - 1
        next_id += int(markers.max())
    return out


def candidates():
    configs = []
    for threshold in (.3, .4, .5, .6, .7):
        for decoder in ('nearest_marker', 'component_nearest'):
            configs.append(dict(decoder=decoder, threshold=threshold))
        for scale in (.5, 1.):
            configs.append(dict(decoder='combined_watershed', threshold=threshold,
                                contact_scale=scale))
        configs.append(dict(decoder='combined_watershed', threshold=threshold,
                            contact_scale=1., marker_min_size=4))
    return configs


def diagnostics(truth, pred):
    """Counts pooled over images; fixed small-object cutoff, no test-derived bins."""
    from train_bbbc_contact import touching_contact_target
    truth, pred = remap_labels(truth), remap_labels(pred)
    intersections, _, iou, areas, pred_areas = _pairwise(truth, pred)
    nt, npred = len(areas), len(pred_areas)
    owners = np.full(nt, -1, dtype=int)
    reverse = np.full(npred, -1, dtype=int)
    if nt and npred:
        best = intersections.argmax(1)
        valid = intersections[np.arange(nt), best] > areas / 2
        owners[valid] = best[valid]
        best_reverse = intersections.argmax(0)
        valid_reverse = intersections[best_reverse, np.arange(npred)] > pred_areas / 2
        reverse[valid_reverse] = best_reverse[valid_reverse]
    counts = np.bincount(owners[owners >= 0], minlength=npred)
    merged = np.zeros(nt, bool)
    merged[owners >= 0] = counts[owners[owners >= 0]] >= 2
    split = np.bincount(reverse[reverse >= 0], minlength=nt) >= 2
    touch_ids = np.unique(truth[touching_contact_target(truth, radius=0)])
    touching = np.isin(np.arange(1, nt + 1), touch_ids)
    matched = (iou > .5).any(1)
    small = areas < 128
    return dict(n_ref=nt, n_pred=npred, merged=int(merged.sum()), split=int(split.sum()),
                n_touching=int(touching.sum()), merged_touching=int((merged & touching).sum()),
                merged_isolated=int((merged & ~touching).sum()), n_small=int(small.sum()),
                missed_small=int((small & ~matched).sum()), missed_large=int((~small & ~matched).sum()),
                count_absolute_error=abs(nt - npred))


def write_new_or_equal(path, value):
    text = json.dumps(value, indent=2, sort_keys=True) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f'Refusing to replace a different artifact: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle:
            handle.write(text)
