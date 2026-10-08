#!/usr/bin/env python3
"""Give the verified upstream CI reader a distinct local conda build identity.

Only info/index.json changes. Compiled and Python payload bytes stay unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

UPSTREAM_SHA = "d3a63b0a43bef30070117a0aa1efb735409ffc4a32550939547c2efa570d4dce"
LOCAL_BUILD = "h7223423_pr7_84891bbb_1"


def repackage(source: Path, destination: Path) -> Path:
    """Copy an upstream artifact, changing only its conda build metadata."""
    if hashlib.sha256(source.read_bytes()).hexdigest() != UPSTREAM_SHA:
        raise ValueError("The input must be the verified PR 7 CI artifact.")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"cwa_reader_rs-0.4.0-{LOCAL_BUILD}.tar.bz2"
    with tarfile.open(source) as original, tarfile.open(output, "w:bz2", format=tarfile.PAX_FORMAT) as packaged:
        for member in original:
            stream = original.extractfile(member) if member.isfile() else None
            if member.name == "info/index.json":
                metadata = json.load(stream)
                metadata.update(build=LOCAL_BUILD, build_number=1)
                payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
                member.size = len(payload)
                stream = io.BytesIO(payload)
            packaged.addfile(member, stream)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "channel/emscripten-wasm32"
    )
    args = parser.parse_args()
    output = repackage(args.artifact, args.output)
    print(f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}")
