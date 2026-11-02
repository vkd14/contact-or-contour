import json,csv,numpy as np
from pathlib import Path
R=Path('results'); P=R/'posthoc_bbbc_decoding'
def csvm(paths,key):
    runs=[{r['id']:float(r[key]) for r in csv.DictReader(open(p))} for p in paths]
    ids=sorted(runs[0]); return ids,np.array([[r[i] for i in ids] for r in runs])
def astra(key):
    runs=[]
    for s in range(10):
        d=json.load(open(R/f'bbbc_astra_final_locked_10seed/vanilla_contact_b7_s{s}/astra_test_tta8.json'))
        assert d['settings']['tta8']['diagnostic_config']=={'decoder':'component_nearest','threshold':0.3}
        runs.append({r['id']:r[key] for r in d['settings']['tta8']['diagnostic_per_image']})
    ids=sorted(runs[0]); return ids,np.array([[r[i] for i in ids] for r in runs])
rng=np.random.default_rng(0)
def eff(a,b):
    d=a-b; dr=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(10000)])
    return round(d.mean(),4),[round(x,4) for x in np.percentile(dr,[2.5,97.5])],int((d>0).sum())
out={}
for key in ['aji','pq','caicedo_af1','dice1']:
    ia,A=astra(key)
    inat,N=csvm([R/f'posthoc_tta_cp/bbbc_contact_native_10seed__s{s}.csv' for s in range(10)],key)
    ic,C=csvm([R/f'cellpose_bbbc_scratch/cellpose3_scratch_s{s}/test_per_image.csv' for s in range(10)],key)
    assert ia==inat==ic
    a,n,c=A.mean(0),N.mean(0),C.mean(0)
    print(key,'astra %.4f (%.4f)'%(A.mean(),A.mean(1).std(ddof=1)),'native %.4f'%N.mean(),'cellpose %.4f'%C.mean())
    print('  astra-native',eff(a,n),' cellpose-astra',eff(c,a))
    out[key]={'astra_vs_native':eff(a,n),'cellpose_vs_astra':eff(c,a)}
json.dump(out,open(R/'astra_final_effects.json','w'),indent=2)
