#!/usr/bin/env python3
"""Statistics added for the v6 revision: (a) image-paired bootstrap of the pooled merged-touching
rate, contact minus contour (base protocol); (b) two-way seed x image bootstrap of object F1,
contact minus contour, base and TTA; (c) t-intervals for the NucleusSegData contact-contour AJI
differences; (d) locked (global nearest-marker) decoder AJI for the two targets.
Writes results/panel_v6_stats.json."""
import csv, glob, json
from pathlib import Path
import numpy as np
from scipy import stats
R = Path(__file__).resolve().parents[1]; rng = np.random.default_rng(0); out = {}
rows = list(csv.DictReader(open(R / "results/bbbc_merge_split/per_image.csv")))
ids = sorted({r["id"] for r in rows})
def mm(m, k):
    A = np.zeros((10, len(ids)))
    for r in rows:
        if r["method"] == m: A[int(r["seed"]), ids.index(r["id"])] = float(r[k])
    return A
Mc, Mo, T = mm("contact", "merged_touching"), mm("contour", "merged_touching"), mm("contact", "touching")
pool = lambda M, i: 100 * M[:, i].sum(1).mean() / T[0, i].sum()
dr = [pool(Mc, i) - pool(Mo, i) for i in (rng.integers(0, 50, 50) for _ in range(10000))]
out["merged_touching_contact_minus_contour"] = [pool(Mc, np.arange(50)) - pool(Mo, np.arange(50)), np.percentile(dr, [2.5, 97.5]).tolist(),
                                               int((Mc.mean(0) < Mo.mean(0)).sum())]
long = list(csv.DictReader(open(R / "results/bbbc_component_preserving/per_image.csv")))
def mat(m):
    d = {(int(r["seed"]), r["id"]): float(r["caicedo_af1"]) for r in long if r["method"] == m}
    return np.array([[d[s, i] for i in ids] for s in range(10)])
def matc(p):
    return np.array([[{r["id"]: float(r["caicedo_af1"]) for r in csv.DictReader(open(f))}[i] for i in ids] for f in sorted(glob.glob(p))])
def twoway(a, b):
    d = [a[rng.integers(0, 10, 10)][:, ii].mean() - b[rng.integers(0, 10, 10)][:, ii].mean() for ii in (rng.integers(0, 50, 50) for _ in range(10000))]
    return [float(a.mean() - b.mean()), np.percentile(d, [2.5, 97.5]).tolist()]
out["twoway_objf1_base"] = twoway(mat("contact"), mat("contour"))
out["twoway_objf1_tta"] = twoway(matc(str(R / "results/posthoc_tta_cp/bbbc_contact_10seed__s*.csv")), matc(str(R / "results/posthoc_tta_cp/bbbc_contour_10seed__s*.csv")))
for sub in ("Huh7", "HepG2"):
    m = {}
    for t in ("contact", "contour"):
        runs = [{r["id"]: float(r["aji"]) for r in csv.DictReader(open(R / f"results/nucleusseg_cp/{t}_s{s}.csv")) if r["subset"] == sub} for s in range(10)]
        k = sorted(runs[0]); m[t] = np.array([[r[i] for i in k] for r in runs]).mean(0)
    d = m["contact"] - m["contour"]; n = len(d); h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    out[f"nsd_{sub}_aji_t_interval"] = [float(d.mean()), [float(d.mean() - h), float(d.mean() + h)]]
for t, m in (("contact", "vanilla_contact_b7"), ("contour", "vanilla_contour_b7")):
    v = [json.load(open(p))["mean"]["aji"] for p in sorted(glob.glob(str(R / f"results/bbbc_{t}_10seed/{m}_s*/summary.json")))]
    out[f"locked_decoder_aji_{t}"] = [float(np.mean(v)), len(v)]
(R / "results/panel_v6_stats.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out))
