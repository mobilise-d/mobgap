"""Regenerate full preset JSON evidence from the original public MATLAB files."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from mobgap_demo_api import analyze_recording, inspect_files

ROOT = Path(__file__).resolve().parents[2]


def export(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for preset, cohort, sensor_height, height in (
        ("healthy", "HA", 0.964, 1.59),
        ("impaired", "MS", 0.975, 1.68),
    ):
        path = ROOT / "example_data/data/lab" / cohort / "001/data.mat"
        options = {
            "preset": preset,
            "cohort": cohort,
            "sensorHeightM": sensor_height,
            "participantHeightM": height,
            "measurementCondition": "laboratory",
        }
        inspection = inspect_files([str(path)], options)
        if inspection["errors"]:
            raise ValueError(inspection["errors"])
        recording = inspection["recordings"][0]
        result = analyze_recording(recording["id"], options)
        (output / f"{preset}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        (output / f"{preset}-options.json").write_text(
            json.dumps(
                {
                    "path": str(path),
                    "recordingLabel": recording["label"],
                    "options": options,
                },
                indent=2,
            )
            + "\n"
        )
        print(preset, json.dumps(result["summary"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "mobgap-demo-native")
    export(parser.parse_args().output)
