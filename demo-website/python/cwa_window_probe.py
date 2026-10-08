"""Bounded real-CWA native/browser parity evidence; never exports sensor rows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Hash every timestamp and every numeric value in a portable byte format."""
    index = np.asarray(frame.index.as_unit("us").asi8, dtype="<i8")
    values = np.asarray(frame.to_numpy(), dtype="<f4", order="C").copy()
    values[np.isnan(values)] = np.nan
    return {
        "samples": len(frame),
        "columns": list(frame.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "startTime": frame.index[0].isoformat() if len(frame) else None,
        "endTime": frame.index[-1].isoformat() if len(frame) else None,
        "indexSha256": hashlib.sha256(index.tobytes()).hexdigest(),
        "valuesSha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "frameBytes": int(frame.memory_usage(index=True, deep=True).sum()),
        "nanCounts": {column: int(frame[column].isna().sum()) for column in frame},
        "minimum": {column: float(frame[column].min()) for column in frame},
        "maximum": {column: float(frame[column].max()) for column in frame},
    }


def probe_file(
    path: str,
    windows: list[dict[str, float]],
    *,
    verify_context: bool = False,
) -> dict[str, Any]:
    """Read only requested raw/resampled windows, with offset zero for parity.

    UTC here is a numerical test convention, not a claim about the real clock
    synchronization timezone. Application analysis requires that timezone.
    """
    import cwa_reader_rs as reader

    started = time.perf_counter()
    metadata = reader.read_metadata(path)
    report: dict[str, Any] = {
        "fileBytes": Path(path).stat().st_size,
        "metadata": metadata,
        "metadataSeconds": time.perf_counter() - started,
        "versions": {name: importlib.metadata.version(name) for name in ("cwa_reader_rs", "numpy", "pandas")},
        "windows": [],
    }
    fs = float(metadata["sample_rate_hz"])
    origin = pd.Timestamp(metadata["start_from_data_raw"], tz="UTC")
    for window in windows:
        result: dict[str, Any] = dict(window)
        start = window["startSeconds"]
        end = start + window["durationSeconds"]
        for mode, hz in (("raw", None), ("resampled", fs)):
            started = time.perf_counter()
            frame = reader.read_cwa_file(
                path,
                cut=reader.seconds(start, end),
                include_magnetometer=False,
                include_temperature=False,
                include_light=False,
                include_battery=False,
                resample_hz=hz,
                fixed_utc_offset_timezone=UTC,
            )
            read_seconds = time.perf_counter() - started
            result[mode] = {**frame_summary(frame), "readSeconds": read_seconds}
            if verify_context:
                context = reader.read_cwa_file(
                    path,
                    cut=reader.seconds(max(0, start - 1), end + 1),
                    include_magnetometer=False,
                    include_temperature=False,
                    include_light=False,
                    include_battery=False,
                    resample_hz=hz,
                    fixed_utc_offset_timezone=UTC,
                )
                sliced = context.loc[
                    (context.index >= origin + pd.Timedelta(seconds=start))
                    & (context.index < origin + pd.Timedelta(seconds=end))
                ]
                pd.testing.assert_frame_equal(frame, sliced, check_exact=True)
                result[mode]["largerWindowSliceMatchesExactly"] = True
                del context, sliced
            del frame
        report["windows"].append(result)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--start", type=float, action="append", default=[])
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-context", action="store_true")
    args = parser.parse_args()
    result = probe_file(
        args.path,
        [{"startSeconds": start, "durationSeconds": args.duration} for start in args.start or [0]],
        verify_context=args.verify_context,
    )
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)
