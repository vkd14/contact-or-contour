#!/usr/bin/env python3
"""Rank complete two-seed arms using only official validation records."""
import argparse
import json

import numpy as np

from astra_utils import (ARMS, METRICS, ROOT, code_digest, config_path, study_name)
from breastseg.core import sha256


def collect(arm):
    path = config_path(arm)
    cfg = json.loads(path.read_text())
    if cfg.get('validation_only') is not True or cfg['seeds'] != [0, 1]:
        raise ValueError(f'Invalid search protocol: {path}')
    name = next(iter(cfg['models']))
    records = []
    for seed in cfg['seeds']:
        folder = ROOT / 'results' / study_name(arm) / f'{name}_s{seed}'
        record_path = folder / 'astra_validation.json'
        if not record_path.exists():
            return None
        record = json.loads(record_path.read_text())
        checkpoint = ROOT / 'checkpoints' / study_name(arm) / f'{name}_s{seed}' / 'best.pt'
        if (record['split'] != 'val' or record['n_images'] != 50 or record['seed'] != seed
                or record['arm'] != arm or record['provenance']['protocol_sha256'] != sha256(path)
                or record['provenance']['checkpoint_sha256'] != sha256(checkpoint)
                or record['provenance']['code_sha256'] != code_digest()):
            raise ValueError(f'Invalid/stale validation record: {record_path}')
        records.append(record)
    modes = {}
    for mode in ('plain', 'tta8'):
        grids = [r['settings'][mode]['grid'] for r in records]
        if [g['config'] for g in grids[0]] != [g['config'] for g in grids[1]]:
            raise ValueError('Seeds must use identical decoder grids')
        pooled = []
        for a, b in zip(*grids):
            pooled.append(dict(config=a['config'], mean={m: (a['mean'][m] + b['mean'][m]) / 2
                                                        for m in METRICS},
                               diagnostics={k: a['diagnostics'][k] + b['diagnostics'][k]
                                            for k in a['diagnostics']},
                               seed_objf1=[a['mean']['caicedo_af1'], b['mean']['caicedo_af1']]))
        modes[mode] = max(pooled, key=rank_key)
    return dict(arm=arm, modes=modes)


def rank_key(entry):
    # Exact ties retain declared grid/arm order after these secondary endpoints.
    return tuple(entry['mean'][m] for m in ('caicedo_af1', 'aji', 'pq'))


def ranking(require_complete=False):
    rows, missing = [], []
    for arm in ARMS:
        record = collect(arm)
        if record is None:
            missing.append(arm)
        else:
            rows.append(record)
    if require_complete and missing:
        raise RuntimeError('Incomplete search arms: ' + ', '.join(missing))
    # TTA is predeclared as the primary final inference protocol for every arm.
    rows.sort(key=lambda r: rank_key(r['modes']['tta8']), reverse=True)
    return rows, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    rows, missing = ranking(args.require_complete)
    print('Rank  Arm             Mode    val_objF1   seed_SD    val_AJI   val_PQ')
    for rank, record in enumerate(rows, 1):
        for mode in ('plain', 'tta8'):
            entry = record['modes'][mode]
            m = entry['mean']
            sd = float(np.std(entry['seed_objf1'], ddof=1))
            print(f"{rank:4}  {record['arm']:16} {mode:5}   {m['caicedo_af1']:.6f}   {sd:.6f}  {m['aji']:.6f}  {m['pq']:.6f}")
            print('      decoder:', json.dumps(entry['config'], sort_keys=True))
            d = entry['diagnostics']
            print(f"      DQ={m['dq']:.4f} SQ={m['sq']:.4f}; "
                  f"merged={d['merged']/max(d['n_ref'], 1):.2%} "
                  f"split={d['split']/max(d['n_ref'], 1):.2%}; "
                  f"small miss={d['missed_small']/max(d['n_small'], 1):.2%}")
    if missing:
        print('Pending (never ranked from partial seeds):', ', '.join(missing))
    if rows:
        print('Validation winner:', rows[0]['arm'])
    print('Ranking: mean two-seed TTA objF1, then AJI, then PQ; no test files read.')


if __name__ == '__main__':
    main()
