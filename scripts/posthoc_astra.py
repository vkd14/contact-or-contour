#!/usr/bin/env python3
"""Strictly validation-only decoder search, or locked final TTA evaluation."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import torch

import posthoc_bbbc_decoding as inference
from astra_utils import (ROOT, METRICS, candidates, code_digest, config_path, decode,
                         diagnostics, study_name, write_new_or_equal)
from breastseg.core import Net, sha256
from breastseg.data import load_bbbc039
from breastseg.instance_metrics import published_metrics
from train_bbbc_contact import instance_annotations


def score_job(job):
    case_id, truth, probability, config, diagnostic = job
    pred = decode(probability, config)
    result = {'id': case_id, **published_metrics(truth, pred)}
    if diagnostic:
        result.update(diagnostics(truth, pred))
    return result


def score(pool, cases, probs, config, diagnostic=False):
    return pool.map(score_job, [(c['id'], c['instances'], probs[c['id']], config, diagnostic)
                               for c in cases], chunksize=2)


def mean(rows):
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--final-lock', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    cfg_path = config_path(args.arm)
    cfg = json.loads(cfg_path.read_text())
    if cfg.get('validation_only') is not True:
        raise ValueError('Search config must explicitly be validation_only')
    final = args.final_lock is not None
    lock = json.loads(args.final_lock.read_text()) if final else None
    if final and (lock['arm'] != args.arm or lock['code_sha256'] != code_digest()):
        raise ValueError('Final arm/code differs from locked protocol')
    split = 'test' if final else 'val'
    # No test loader, test annotations, or test summaries in the search path.
    cases = load_bbbc039(ROOT / 'data/bbbc039', (split,))
    annotations = instance_annotations(ROOT / 'data/bbbc039', {c['id'] for c in cases})
    for c in cases:
        c['instances'] = annotations[c['id']]
    inference.NATIVE_PAD = tuple(cfg['pad_to'])
    cv2.setNumThreads(1)
    torch.set_num_threads(8)
    study = lock['study'] if final else study_name(args.arm)
    expected_protocol = sha256(ROOT / lock['config_path']) if final else sha256(cfg_path)
    model_name, variant = next(iter(cfg['models'].items()))
    # Spawn avoids forking an initialized CUDA runtime.
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for seed in (range(10) if final else cfg['seeds']):
            run = f'{model_name}_s{seed}'
            folder = ROOT / 'results' / study / run
            output = folder / ('astra_test_tta8.json' if final else 'astra_validation.json')
            checkpoint_path = ROOT / 'checkpoints' / study / run / 'best.pt'
            provenance = dict(protocol_sha256=expected_protocol, checkpoint_sha256=sha256(checkpoint_path),
                              code_sha256=code_digest())
            if output.exists():
                old = json.loads(output.read_text())
                if old['provenance'] != provenance:
                    raise ValueError(f'Stale artifact: {output}')
                print('SKIP', output, flush=True)
                continue
            if not final:
                summary = json.loads((folder / 'summary.json').read_text())
                if summary.get('validation_only') is not True:
                    raise ValueError('Refusing non-validation search run')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if checkpoint['protocol_sha256'] != expected_protocol:
                raise ValueError('Checkpoint protocol mismatch')
            if checkpoint.get('training_code_sha256') != code_digest():
                raise ValueError('Checkpoint training code differs from evaluation code')
            model = Net(variant, pretrained=False, classes=2).cuda()
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            record = dict(arm=args.arm, seed=seed, split=split, n_images=len(cases),
                          provenance=provenance, settings={})
            for tta in ((True,) if final else (False, True)):
                mode = 'tta8' if tta else 'plain'
                probs = inference.probabilities(model, cases, model.use_edge, tta)
                grid = [lock['decoders'][mode]] if final else candidates()
                entries = []
                for config in grid:
                    rows = score(pool, cases, probs, config, diagnostic=True)
                    diagnostic_keys = ('n_ref', 'n_pred', 'merged', 'split', 'n_touching',
                                       'merged_touching', 'merged_isolated', 'n_small',
                                       'missed_small', 'missed_large', 'count_absolute_error')
                    entries.append(dict(config=config, mean=mean(rows),
                                        diagnostics={k: sum(r[k] for r in rows) for k in diagnostic_keys}))
                chosen = max(entries, key=lambda e: (e['mean']['caicedo_af1'], e['mean']['aji'], e['mean']['pq']))
                rows = score(pool, cases, probs, chosen['config'], diagnostic=True)
                record['settings'][mode] = dict(grid=entries, diagnostic_config=chosen['config'],
                                                diagnostic_per_image=rows)
                print(json.dumps(dict(arm=args.arm, seed=seed, split=split, mode=mode,
                                      best=chosen)), flush=True)
            write_new_or_equal(output, record)
            del model
            torch.cuda.empty_cache()
    if final:
        aggregate = {'arm': args.arm, 'seeds': list(range(10)),
                     'primary_inference': 'tta8', 'decoders': lock['decoders'], 'settings': {}}
        for mode in ('plain', 'tta8'):
            means = []
            for seed in range(10):
                folder = ROOT / 'results' / study / f'{model_name}_s{seed}'
                if mode == 'plain':
                    means.append(json.loads((folder / 'summary.json').read_text())['mean'])
                else:
                    result = json.loads((folder / 'astra_test_tta8.json').read_text())
                    means.append(result['settings']['tta8']['grid'][0]['mean'])
            aggregate['settings'][mode] = {
                metric: {'mean': float(np.mean([r[metric] for r in means])),
                         'seed_sd': float(np.std([r[metric] for r in means], ddof=1))}
                for metric in METRICS}
        write_new_or_equal(ROOT / 'results' / study / 'astra_final_aggregate.json', aggregate)
        print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
