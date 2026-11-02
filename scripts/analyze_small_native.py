#!/usr/bin/env python3
"""Small-nucleus (<128 px) unmatched rate (no IoU>0.5 match) on BBBC039 test for the
native-resolution contact model vs the 512^2 contact model (base protocol, no TTA,
component-preserving decoder), pooled over images per seed, mean (SD) over ten seeds.
Same definitions as analyze_stratified_comparison.py.  Writes results/small_native.json."""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from analyze_component_preserving import component_preserving
from analyze_stratified_comparison import per_nucleus, load_pred  # noqa
import analyze_stratified_comparison as asc
import zipfile
from breastseg.data import load_bbbc039
from breastseg.instance_metrics import bbbc039_reference_instances
STUDIES = {"contact": "bbbc_contact_10seed/vanilla_contact_b7", "native": "bbbc_contact_native_10seed/vanilla_contact_b7"}

def pred(method, seed, cid):
    d = ROOT / "results" / f"{STUDIES[method]}_s{seed}" / "predictions"
    fg = cv2.imread(str(d / f"{cid}_foreground.png"), 0) > 0
    bd = cv2.imread(str(d / f"{cid}_contact.png"), 0) > 0
    return np.asarray(component_preserving(fg, bd), np.int64)

def main():
    cases = load_bbbc039(ROOT / "data/bbbc039", ("test",))
    out = {}
    with zipfile.ZipFile(ROOT / "data/bbbc039/masks.zip") as z:
        truths = {c["id"]: np.asarray(bbbc039_reference_instances(cv2.imdecode(np.frombuffer(z.read(f"masks/{c['id']}.png"), np.uint8), cv2.IMREAD_UNCHANGED)), np.int64) for c in cases}
    for m in STUDIES:
        rates, all_rates = [], []
        for s in range(10):
            small = missed = n = nm = 0
            for cid, t in truths.items():
                p = pred(m, s, cid)
                area = np.bincount(t.ravel(), minlength=int(t.max()) + 1)
                ids = np.arange(1, int(t.max()) + 1)
                inter = np.zeros((int(t.max()) + 1, int(p.max()) + 1)); np.add.at(inter, (t.ravel(), p.ravel()), 1)
                inter = inter[1:, 1:]; pa = np.bincount(p.ravel(), minlength=int(p.max()) + 1)[1:]
                iou = inter / (area[ids][:, None] + pa[None, :] - inter + 1e-9)
                miss = iou.max(1) <= .5 if pa.size else np.ones(len(ids), bool)
                sm = area[ids] < 128
                small += sm.sum(); missed += (miss & sm).sum(); n += len(ids); nm += miss.sum()
            rates.append(100 * missed / small); all_rates.append(100 * nm / n)
        out[m] = {"small_unmatched_pct": [float(np.mean(rates)), float(np.std(rates, ddof=1))],
                  "all_unmatched_pct": [float(np.mean(all_rates)), float(np.std(all_rates, ddof=1))], "n_small": int(small)}
        print(m, out[m], flush=True)
    (ROOT / "results/small_native.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
