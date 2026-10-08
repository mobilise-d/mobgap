"""Extract small lossless sample MAT files from this repository's public examples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "demo-website/public/samples"


def prepare_samples(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for identifier, cohort, preset, label, duration, sensor_height, height in (
        ("healthy", "HA", "healthy", "Healthy walking", 137.59, 0.964, 1.59),
        ("impaired", "MS", "impaired", "Impaired walking", 227.28, 0.975, 1.68),
    ):
        source = ROOT / "example_data/data/lab" / cohort / "001"
        data = loadmat(source / "data.mat", simplify_cells=True)["data"]
        metadata = loadmat(source / "infoForAlgo.mat", simplify_cells=True)["infoForAlgo"]
        trial = data["TimeMeasure1"]["Test11"]["Trial1"]
        selected = {key: trial[key] for key in ("StartDateTime", "TimeZone") if key in trial}
        selected["SU"] = {"LowerBack": trial["SU"]["LowerBack"]}
        path = output_dir / (identifier + ".mat")
        savemat(
            path,
            {"data": {"TimeMeasure1": {"Test11": {"Trial1": selected}}}, "infoForAlgo": metadata},
            do_compression=True,
        )
        samples.append(
            {
                "id": identifier,
                "label": label,
                "description": f"{duration:.2f}s laboratory walking trial from the public mobgap examples",
                "files": [f"/samples/{identifier}.mat"],
                "preset": preset,
                "cohort": cohort,
                "sensorHeightM": sensor_height,
                "participantHeightM": height,
                "recordingLabel": "TimeMeasure1 / Test11 / Trial1",
                "source": str((source / "data.mat").relative_to(ROOT)),
                "sourceSha256": hashlib.sha256((source / "data.mat").read_bytes()).hexdigest(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    result = {"samples": samples}
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(json.dumps(prepare_samples(arguments.output), indent=2))
