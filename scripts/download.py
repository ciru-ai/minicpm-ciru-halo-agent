#!/usr/bin/env python3
"""Download one immutable snapshot of the MiniCPM Ultra release."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "jcbtc/MiniCPM-4.7-Ultra-CIRU-HALO-AGENT"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    revision = HfApi().model_info(REPO, revision=args.revision).sha
    snapshot_download(REPO, revision=revision, local_dir=args.destination)
    release = json.loads((args.destination / "release.json").read_text())
    for name, item in release["archives"].items():
        path = args.destination / name
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise SystemExit("Checksum mismatch: " + name)
    (args.destination / "download-receipt.json").write_text(
        json.dumps({"repo": REPO, "revision": revision}, indent=2) + "\n"
    )
    print("Downloaded and verified runtime archives at revision " + revision)
    print("The installer will verify every model-weight file before installation.")


if __name__ == "__main__":
    main()
