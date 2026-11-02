#!/usr/bin/env python3
"""Freeze one validation winner before any Astra test evaluation."""
import argparse
import json

from astra_utils import ROOT, code_digest, config_path, write_new_or_equal
from breastseg.core import sha256
from summarize_astra_search import ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm')
    args = parser.parse_args()
    search_path = config_path(args.arm)
    rows, _ = ranking(require_complete=True)
    if rows[0]['arm'] != args.arm:
        raise ValueError(f"Requested arm is not the validation winner: {rows[0]['arm']}")
    winning = rows[0]
    cfg = json.loads(search_path.read_text())
    cfg.update(validation_only=False, seeds=list(range(10)),
               astra_study='bbbc_astra_final_locked_10seed',
               purpose='Single frozen Astra final study; no test-driven reselection',
               locked_decoder=winning['modes']['plain']['config'])
    final_path = ROOT / 'configs' / 'bbbc_astra_final_locked.json'
    lock = dict(arm=args.arm, study='bbbc_astra_final_locked_10seed',
                config_path=str(final_path.relative_to(ROOT)),
                search_config_sha256=sha256(search_path), code_sha256=code_digest(),
                selection='Mean two-seed validation TTA objF1, then AJI, then PQ',
                primary_inference='tta8',
                decoders={mode: winning['modes'][mode]['config'] for mode in ('plain', 'tta8')},
                validation_ranking=rows)
    # A fixed filename across all arms prevents accidentally testing another winner.
    lock_path = ROOT / 'configs' / 'astra_final_lock.json'
    write_new_or_equal(lock_path, lock)
    write_new_or_equal(final_path, cfg)
    print('Locked', args.arm, 'in', lock_path)


if __name__ == '__main__':
    main()
