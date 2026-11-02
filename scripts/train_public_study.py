#!/usr/bin/env python3
"""Locked, matched-budget exploratory public-data study. Test only after selection."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from breastseg.core import Net, SegmentationLoss, features, metrics, sha256
from breastseg.data import load_bbbc039

ROOT=Path(__file__).resolve().parents[1]

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

class Cached(Dataset):
    def __init__(self,cases,edge,augment=False,sobel=False): self.cases,self.edge,self.augment,self.sobel=cases,edge,augment,sobel
    def __len__(self): return len(self.cases)
    def __getitem__(self,i):
        c=self.cases[i];x=c['xsobel'] if self.sobel else (c['x4'] if self.edge else c['x4'][:3]);y=c['y']
        if self.augment:
            k=random.randrange(4);x,y=torch.rot90(x,k,(-2,-1)),torch.rot90(y,k,(-2,-1))
            if random.random()<.5: x,y=x.flip(-1),y.flip(-1)
        return x,y,i

@torch.inference_mode()
def evaluate(model,loader,cases,full=False,out=None):
    model.eval();rows=[]
    if out: (out/'predictions').mkdir(parents=True,exist_ok=True)
    for x,_,indices in loader:
        with torch.autocast('cuda',dtype=torch.bfloat16): z=model(x.cuda(non_blocking=True))
        probs=z.float().sigmoid().cpu().numpy()[:,0]
        for p,i in zip(probs,indices.tolist()):
            c=cases[i];g=c['mask'].astype(bool)
            p=cv2.resize(p,(g.shape[1],g.shape[0]),interpolation=cv2.INTER_LINEAR)>.5
            if full: m=metrics(p,g)
            else:
                denom=p.sum()+g.sum();m={'dice':float(2*(p&g).sum()/denom) if denom else 1.}
            rows.append(dict(id=c['id'],group=c['group'],**m))
            if out and not cv2.imwrite(str(out/'predictions'/f"{c['id']}.png"),p.astype(np.uint8)*255):
                raise IOError('Failed prediction write')
    if len(rows)!=len(cases): raise RuntimeError('Incomplete evaluation')
    return rows

def state_digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items(): h.update(k.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

def run_one(cfg,variant,seed,cases,protocol_hash,study='public_study'):
    out=ROOT/'results'/study/f'{variant}_s{seed}';out.mkdir(parents=True,exist_ok=True)
    ckptdir=ROOT/'checkpoints'/study/f'{variant}_s{seed}';ckptdir.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():
        old=json.loads((out/'summary.json').read_text())
        if old['protocol_sha256']!=protocol_hash: raise RuntimeError('Protocol differs from completed run')
        print('SKIP completed',variant,seed,flush=True);return
    if (out/'history.json').exists(): raise RuntimeError(f'Partial run exists: {out}; inspect before restarting')
    start=time.time();seed_all(seed)
    model=Net(variant,pretrained=True).cuda()
    initial_hash=state_digest(model)
    tr=[c for c in cases if c['split']=='train'];va=[c for c in cases if c['split']=='val'];te=[c for c in cases if c['split']=='test']
    gen=torch.Generator().manual_seed(seed)
    def loader(cs,train=False): return DataLoader(Cached(cs,model.use_edge,train,sobel='sobel' in variant),batch_size=cfg['batch_size'],shuffle=train,
        generator=gen if train else None,num_workers=0,pin_memory=True,drop_last=False)
    tl,vl=loader(tr,True),loader(va)
    loss_fn=SegmentationLoss(.1 if 'boundary' in variant else 0.)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'],eta_min=cfg['learning_rate']*.01)
    history=[];best=-1.;bestepoch=None
    for epoch in range(1,cfg['epochs']+1):
        model.train();losses=[];t0=time.time()
        for x,y,_ in tl:
            opt.zero_grad(set_to_none=True);x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=loss_fn(model(x),y)
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            opt.step();losses.append((loss.item(),len(x)))
        rows=evaluate(model,vl,va);score=float(np.mean([r['dice'] for r in rows]))
        if score>best:
            best,bestepoch=score,epoch
            torch.save({'model_state_dict':model.state_dict(),'variant':variant,'seed':seed,'epoch':epoch,
                'val_dice':best,'protocol_sha256':protocol_hash},ckptdir/'best.pt')
        record=dict(epoch=epoch,train_loss=sum(l*n for l,n in losses)/sum(n for l,n in losses),val_dice=score,
            best_val_dice=best,learning_rate=opt.param_groups[0]['lr'],seconds=time.time()-t0)
        history.append(record);save_json(out/'history.json',history)
        save_json(ROOT/'results'/study/'status.json',dict(state='training',run=out.name,**record))
        print(json.dumps(dict(run=out.name,**record)),flush=True);scheduler.step()
    model.load_state_dict(torch.load(ckptdir/'best.pt',map_location='cpu',weights_only=True)['model_state_dict'],strict=True)
    rows=evaluate(model,loader(te),te,full=True,out=out)
    with (out/'test_per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(variant=variant,seed=seed,n_test=len(rows),epochs=cfg['epochs'],best_epoch=bestepoch,best_val_dice=best,
        parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial_hash,
        checkpoint_sha256=sha256(ckptdir/'best.pt'),protocol_sha256=protocol_hash,seconds=time.time()-start,
        edge_gain=float(model.edge_gain.tanh().detach().cpu()) if model.residual else None,
        mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('id','group')})
    save_json(out/'summary.json',summary);print('COMPLETE '+json.dumps(summary),flush=True)
    del model,opt;torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--variants',nargs='+');p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--config',type=Path,default=ROOT/'configs/public_study.json');p.add_argument('--study',default='public_study')
    a=p.parse_args();cfg=json.loads(a.config.read_text());ph=sha256(a.config)
    cv2.setNumThreads(1);torch.set_num_threads(8)
    if hasattr(cv2,'setLogLevel'): cv2.setLogLevel(0)
    cases=load_bbbc039(a.data)
    manifest=[{k:v for k,v in c.items() if k not in ('image','mask')} for c in cases]
    save_json(ROOT/'results'/a.study/'split_manifest.json',manifest)
    save_json(ROOT/'results'/a.study/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),protocol_sha256=ph,config=cfg,archives={p.name:sha256(p) for p in a.data.glob('*.zip')},
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/breastseg/core.py',ROOT/'src/breastseg/data.py']}))
    print('Caching 200 image features',flush=True)
    for c in cases:
        c['x4']=features(c['image'],True,cfg['image_size'])
        if any('sobel' in v for v in cfg['variants']): c['xsobel']=features(c['image'],True,cfg['image_size'],'sobel')
        c['y']=torch.from_numpy(cv2.resize(c['mask'],(cfg['image_size'],cfg['image_size']),interpolation=cv2.INTER_NEAREST).astype(np.float32)[None])
    for seed in (a.seeds if a.seeds is not None else cfg['seeds']):
        for variant in (a.variants or cfg['variants']): run_one(cfg,variant,seed,cases,ph,a.study)
    save_json(ROOT/'results'/a.study/'status.json',dict(state='complete',protocol_sha256=ph))

if __name__=='__main__': main()
