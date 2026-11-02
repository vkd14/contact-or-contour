#!/usr/bin/env python3
"""Fetch the official BBBC039 and BBBC038 (2018 Data Science Bowl) archives into data/ and verify pinned SHA-256 hashes.

NucleusSegData (expanded release) is fetched separately with scripts/download_nucleusseg.py --release extended.
"""
import argparse
import os
from pathlib import Path
import tempfile
import urllib.request
from breastseg.core import sha256

ROOT = Path(__file__).resolve().parents[1]
URLS = {
    'bbbc039': {n: 'https://data.broadinstitute.org/bbbc/BBBC039/' + n for n in ('images.zip', 'masks.zip', 'metadata.zip')},
    'bbbc038': {n: 'https://data.broadinstitute.org/bbbc/BBBC038/' + n for n in ('stage1_test.zip', 'stage1_solution.csv', 'metadata.xlsx')},
}
HASHES = {
    'bbbc039': {'masks.zip': 'f9e6043d8ca56344a4886f96a700d804d6ee982f31e2b2cd3194af2a053c2710',
                'images.zip': '6f30a5d4fe38c928ded972704f085975f8dc0d65d9aa366df00e5a9d449fddd7',
                'metadata.zip': 'a2c1f900bed9ba92a99553efd4c2ae98598433691c7401d818653ab61110deb2'},
    'bbbc038': {'stage1_test.zip': 'c097954151884341448a5c77aab21f581815ad9cad203ff2ff38d6947c9f0733',
                'stage1_solution.csv': 'b697d9ae093d72079f45adea7e9823204d3f67dd5ab657f68ce86ab988fc32f9'},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', choices=list(URLS), default=list(URLS))
    p.add_argument('--dest', type=Path, default=ROOT / 'data')
    a = p.parse_args()
    for ds in a.datasets:
        expected = HASHES[ds]
        dest = a.dest / ds
        dest.mkdir(parents=True, exist_ok=True)
        for name, url in URLS[ds].items():
            final = dest / name
            if final.exists():
                if name in expected and sha256(final) != expected[name]:
                    raise ValueError(f'Hash mismatch; preserved {final}')
                print('Verified', final)
                continue
            with tempfile.NamedTemporaryFile(dir=dest, suffix='.download', delete=False) as f:
                temp = Path(f.name)
            urllib.request.urlretrieve(url, temp)
            if name in expected and sha256(temp) != expected[name]:
                raise ValueError(f'Download checksum differs: {temp}')
            os.replace(temp, final)
            print('Downloaded', final)


if __name__ == '__main__':
    main()
