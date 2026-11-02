#!/usr/bin/env python3
"""Realized batchwise positive weight (1-r)/r clipped to [1,50] for the contact and full-contour targets on BBBC039 training images at 512^2 (batches of 8, 5000 random draws). Writes results/positive_weight.json."""
import sys, json, numpy as np, cv2
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'scripts')]
from train_bbbc_contact import touching_contact_target, full_contour_target, instance_annotations
from breastseg.data import load_bbbc039
from pathlib import Path
R=Path(__file__).resolve().parents[1]
cases=load_bbbc039(R/'data/bbbc039',('train',)); ann=instance_annotations(R/'data/bbbc039')
rs={'contact':[], 'contour':[]}
for c in cases:
    ins=ann[c['id']]
    for k,f in (('contact',touching_contact_target),('contour',full_contour_target)):
        t=cv2.resize(f(ins,radius=1).astype(np.uint8),(512,512),interpolation=cv2.INTER_NEAREST)
        rs[k].append(t.mean())
rng=np.random.default_rng(0); out={}
for k,v in rs.items():
    v=np.array(v); w=[]
    for _ in range(5000):
        r=v[rng.choice(len(v),8,replace=False)].mean()
        w.append(np.clip((1-r)/r,1,50) if r>0 else 1)
    w=np.array(w); out[k]={'pixel_fraction':float(v.mean()),'weight_mean':float(w.mean()),'clipped_share':float((w>=50).mean())}; print(k,'pixel fraction %.4f'%v.mean(),'pos weight mean %.1f median %.1f  p5 %.1f p95 %.1f clipped@50 %.2f'%(w.mean(),np.median(w),*np.percentile(w,[5,95]),(w>=50).mean()))
(R/'results/positive_weight.json').write_text(json.dumps(out,indent=1))
