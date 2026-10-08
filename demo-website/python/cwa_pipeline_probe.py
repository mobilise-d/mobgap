"""Run real CWA calendar days sequentially, retaining JSON results only.

Pass known participant measurements and the actual clock synchronization
timezone. Probe metadata values are never inferred from a filename.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mobgap_demo_api as api


def run_days(
    path: str,
    timezone: str,
    preset: str,
    participant_height_m: float,
    sensor_height_m: float,
    cohort: str,
    indexes: list[int] | None = None,
) -> dict[str, Any]:
    """Attempt every selected day; a day failure does not discard other results."""
    options = {
        "preset": preset,
        "cohort": cohort,
        "participantHeightM": participant_height_m,
        "sensorHeightM": sensor_height_m,
        "measurementCondition": "free_living",
        "timezone": timezone,
    }
    inspected = api.inspect_files([path], options)
    if inspected["errors"]:
        raise ValueError(inspected["errors"])
    recording = inspected["recordings"][0]
    days = api.cwa_day_windows(recording["id"], timezone)
    output: dict[str, Any] = {"recording": recording, "calendar": days, "days": []}
    api.start_cwa_day_batch(recording["id"], options, indexes)
    try:
        while True:
            started = time.perf_counter()
            step = api.next_cwa_day()
            elapsed = time.perf_counter() - started
            if "packet" in step:
                entry = {**step["packet"], "options": options, "elapsedSeconds": elapsed}
                if "result" in entry:
                    result = entry["result"]
                    entry.update(
                        resultJsonBytes=len(json.dumps(result, allow_nan=False).encode()),
                        tableSizes={
                            key: {"rows": len(table["rows"]), "jsonBytes": len(json.dumps(table).encode())}
                            for key, table in result["tables"].items()
                        },
                    )
                output["days"].append(entry)
            if step["done"]:
                break
    finally:
        api.cancel_cwa_day_batch()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--timezone", required=True)
    parser.add_argument("--preset", choices=("healthy", "impaired"), required=True)
    parser.add_argument("--participant-height-m", required=True, type=float)
    parser.add_argument("--sensor-height-m", required=True, type=float)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--day", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_days(
        args.path,
        args.timezone,
        args.preset,
        args.participant_height_m,
        args.sensor_height_m,
        args.cohort,
        args.day,
    )
    # Native process RSS includes decoder, pipeline, interpreter and allocator.
    import resource

    report["nativePeakRssKiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "peakRssKiB": report["nativePeakRssKiB"],
                "days": [
                    {
                        "label": day["day"]["label"],
                        "elapsedSeconds": day["elapsedSeconds"],
                        "summary": day.get("result", {}).get("summary"),
                        "error": day.get("error"),
                    }
                    for day in report["days"]
                ],
            },
            indent=2,
        )
    )
