"""Read-only native/browser checks for mounted-file seek/read and MAT expansion.

Large probe files are created only in a caller-selected scratch directory.
No full-file read, hash or in-memory byte array is needed for the seek probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def create_sparse_probe(path: Path, size_bytes: int = 512 * 1024 * 1024) -> None:
    """Create a large disk-backed file using two tiny writes and a sparse gap."""
    if size_bytes < 128:
        raise ValueError("The sparse probe requires at least 128 bytes.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(b"MOBGAP-WORKERFS-BEGIN".ljust(32, b"."))
        file.seek(size_bytes - 32)
        file.write(b"MOBGAP-WORKERFS-END".ljust(32, b"."))


def seek_read_probe(path: str | Path) -> dict[str, Any]:
    """Seek to the beginning, middle and end, reading only 80 bytes total."""
    path = Path(path)
    size = path.stat().st_size
    positions = ((0, 32), (size // 2, 16), (size - 32, 32))
    blocks = []
    with path.open("rb", buffering=0) as file:
        for position, count in positions:
            file.seek(position)
            data = file.read(count)
            if len(data) != count:
                raise ValueError(f"Short read at offset {position}: expected {count}, got {len(data)}.")
            blocks.append({"offset": position, "bytes": count, "hex": data.hex()})
    return {
        "fileBytes": size,
        "bytesRequested": sum(block["bytes"] for block in blocks),
        "largestReadBytes": 32,
        "blocks": blocks,
        "probeSha256": hashlib.sha256(b"".join(bytes.fromhex(block["hex"]) for block in blocks)).hexdigest(),
    }


class CountingReader:
    """Count actual byte reads and seek positions consumed by SciPy's readers."""

    def __init__(self, path: str | Path):
        self.file = Path(path).open("rb", buffering=0)
        self.bytes_read = 0
        self.read_calls = 0
        self.max_read = 0
        self.seeks = 0

    def read(self, size: int = -1) -> bytes:
        data = self.file.read(size)
        self.bytes_read += len(data)
        self.read_calls += 1
        self.max_read = max(self.max_read, len(data))
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        self.seeks += 1
        return self.file.seek(offset, whence)

    def tell(self) -> int:
        return self.file.tell()

    def close(self) -> None:
        self.file.close()

    def stats(self) -> dict[str, int]:
        return {
            "bytesRead": self.bytes_read,
            "readCalls": self.read_calls,
            "largestReadBytes": self.max_read,
            "seekCalls": self.seeks,
        }


def mat_access_probe(path: str | Path) -> dict[str, Any]:
    """Measure header-only reads, full MATLAB loading and retained mobgap frames."""
    import numpy as np
    from mobgap.data import load_mobilised_matlab_format
    from scipy.io import loadmat, whosmat

    path = Path(path)
    header_reader = CountingReader(path)
    try:
        variables = whosmat(header_reader)
        header_stats = header_reader.stats()
    finally:
        header_reader.close()
    full_reader = CountingReader(path)
    try:
        raw = loadmat(full_reader, squeeze_me=True, struct_as_record=False, mat_dtype=True)
        full_stats = full_reader.stats()
    finally:
        full_reader.close()
    seen = set()

    def array_bytes(value: Any) -> int:
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            return value.nbytes + (sum(array_bytes(item) for item in value.flat) if value.dtype.hasobject else 0)
        if hasattr(value, "_fieldnames"):
            return sum(array_bytes(getattr(value, name)) for name in value._fieldnames)
        if isinstance(value, dict):
            return sum(array_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(array_bytes(item) for item in value)
        return 0

    decoded_bytes = array_bytes(raw)
    del raw
    loaded = load_mobilised_matlab_format(path)
    trials = []
    for name, recording in loaded.items():
        frame = recording.imu_data["LowerBack"]
        trials.append(
            {
                "name": list(name),
                "samples": len(frame),
                "frameBytes": int(frame.memory_usage(index=True, deep=True).sum()),
            }
        )
    return {
        "fileBytes": path.stat().st_size,
        "variables": variables,
        "whosmat": header_stats,
        "loadmat": full_stats,
        "decodedMatArrayBytes": decoded_bytes,
        "trials": trials,
        "retainedFrameBytes": sum(trial["frameBytes"] for trial in trials),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--create-sparse-mib", type=int)
    parser.add_argument("--mat", action="store_true")
    args = parser.parse_args()
    if args.create_sparse_mib:
        create_sparse_probe(args.path, args.create_sparse_mib * 1024 * 1024)
    print(json.dumps(mat_access_probe(args.path) if args.mat else seek_read_probe(args.path), indent=2))
