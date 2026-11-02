#!/usr/bin/env python3
"""Download and verify an official research-only NucleusSegData release."""
import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path


RELEASES = {
    "original": {
        "url": "https://www.cs.bilkent.edu.tr/~gunduz/downloads/NucleusSegData/NucleusSegData.zip",
        "sha256": "6c9dc687f6fdbb56a46cda575a36186f0186a9dc83edfccfc32ee924fedb721a",
        "archive": "NucleusSegData.zip",
        "output": Path("data/nucleusseg"),
    },
    "extended": {
        "url": "https://www.cs.bilkent.edu.tr/~gunduz/downloads/NucleusSegData/NucleusSegDataExtended",
        "sha256": "31c382666dd8b6d4f5989c468998e932bb38bd5288d8ae37317baa94fbdff111",
        "archive": "NucleusSegDataExtended.zip",
        "output": Path("data/nucleusseg_extended"),
    },
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=RELEASES, default="original")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    release = RELEASES[args.release]
    output = args.output if args.output is not None else release["output"]
    output.mkdir(parents=True, exist_ok=True)
    archive = output / release["archive"]
    if not archive.exists():
        urllib.request.urlretrieve(release["url"], archive)
    if digest(archive) != release["sha256"]:
        raise RuntimeError(f"NucleusSegData {args.release} checksum mismatch")
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(output)
    print(f"Verified and extracted {args.release} release: {archive}")


if __name__ == "__main__":
    main()
