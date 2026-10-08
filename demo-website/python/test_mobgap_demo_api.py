"""User-facing file inspection and full pipeline contracts for the browser adapter."""

import importlib.util
import json
from pathlib import Path

import pytest
from scipy.io import savemat

MODULE_PATH = Path(__file__).with_name("mobgap_demo_api.py")
ROOT = Path(__file__).resolve().parents[2]
HA = ROOT / "example_data/data/lab/HA/001"


def api():
    spec = importlib.util.spec_from_file_location("mobgap_demo_api", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_matlab_recordings_keep_units_and_actual_metadata():
    result = api().inspect_files([str(HA / "data.mat"), str(HA / "infoForAlgo.mat")])
    assert result["errors"] == []
    assert len(result["recordings"]) == 3
    first = result["recordings"][0]
    assert first["testName"] == ["TimeMeasure1", "Test11", "Trial1"]
    assert first["samples"] == 13759
    assert first["samplingRateHz"] == 100
    assert first["durationSeconds"] == 137.59
    assert first["channels"] == ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    assert first["metadata"]["sensorHeightM"] == pytest.approx(0.964)
    assert first["metadata"]["heightM"] == pytest.approx(1.59)
    assert first["metadata"].get("cohort") is None
    json.dumps(result, allow_nan=False)


def test_full_healthy_pipeline_matches_known_matlab_trial():
    module = api()
    recording = module.inspect_files([str(HA / "data.mat")])["recordings"][0]
    assert recording["metadata"] == {}
    result = module.analyze_recording(
        recording["id"],
        {
            "preset": "healthy",
            "cohort": "HA",
            "sensorHeightM": 0.964,
            "participantHeightM": 1.59,
            "measurementCondition": "laboratory",
        },
    )
    assert result["summary"]["gaitSequences"] == 6
    assert result["summary"]["initialContacts"] == 60
    assert result["summary"]["walkingBouts"] == 6
    assert "aggregated_parameters" in result["tables"]
    per_second = result["tables"]["per_second_parameters"]
    length_column = per_second["columns"].index("stride_length_m")
    assert per_second["rows"][0][length_column] == pytest.approx(1.233167381971792, rel=1e-9)
    assert all(
        cell is None or isinstance(cell, (str, int, float, bool))
        for table in result["tables"].values()
        for row in table["rows"]
        for cell in row
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "content, message",
    [
        (b"not a MATLAB file", None),
        (b"MATLAB 7.3 MAT-file, Platform: GLNXA64".ljust(128, b" "), "v7.3/HDF5"),
        (b"\x89HDF\r\n\x1a\n", "v7.3/HDF5"),
    ],
)
def test_invalid_matlab_upload_reports_file_error(tmp_path, content, message):
    path = tmp_path / "uploaded.mat"
    path.write_bytes(content)
    result = api().inspect_files([str(path)])
    assert result["recordings"] == []
    assert len(result["errors"]) == 1
    assert result["errors"][0]["fileName"] == "uploaded.mat"
    assert result["errors"][0]["message"]
    if message:
        assert message in result["errors"][0]["message"]


def test_unknown_mat_structure_reports_supported_format(tmp_path):
    path = tmp_path / "other.mat"
    savemat(path, {"unrelated": [1, 2, 3]})
    result = api().inspect_files([str(path)])
    assert result["recordings"] == []
    assert "no Mobilise-D 'data'" in result["errors"][0]["message"]


def test_unknown_upload_requires_height_and_explicit_cohort():
    module = api()
    recording = module.inspect_files([str(HA / "data.mat")])["recordings"][0]
    with pytest.raises(ValueError, match="cohort"):
        module.analyze_recording(recording["id"], {"preset": "healthy"})
    with pytest.raises(ValueError, match="Sensor height is required"):
        module.analyze_recording(recording["id"], {"preset": "healthy", "cohort": "HA"})


def test_generated_samples_preserve_original_matlab_imu(tmp_path):
    from mobgap.data import load_mobilised_matlab_format
    from pandas.testing import assert_frame_equal

    from prepare_samples import prepare_samples

    manifest = prepare_samples(tmp_path)
    for sample in manifest["samples"]:
        original = load_mobilised_matlab_format(ROOT / sample["source"])[("TimeMeasure1", "Test11", "Trial1")]
        derived = load_mobilised_matlab_format(tmp_path / (sample["id"] + ".mat"))
        assert list(derived) == [("TimeMeasure1", "Test11", "Trial1")]
        selected = next(iter(derived.values()))
        assert_frame_equal(original.imu_data["LowerBack"], selected.imu_data["LowerBack"])
        assert original.metadata == selected.metadata


def test_user_sensor_height_overrides_companion_metadata():
    module = api()
    recording = module.inspect_files([str(HA / "data.mat"), str(HA / "infoForAlgo.mat")])["recordings"][0]
    result = module.analyze_recording(
        recording["id"],
        {
            "preset": "healthy",
            "cohort": "HA",
            "sensorHeightM": 0.8,
            "participantHeightM": 1.59,
        },
    )
    table = result["tables"]["per_second_parameters"]
    value = table["rows"][0][table["columns"].index("stride_length_m")]
    assert value < 1.2
    assert value > 0


def test_companion_metadata_is_not_assigned_to_multiple_participants(tmp_path):
    import shutil

    shutil.copyfile(HA / "data.mat", tmp_path / "healthy.mat")
    shutil.copyfile(ROOT / "example_data/data/lab/MS/001/data.mat", tmp_path / "impaired.mat")
    shutil.copyfile(HA / "infoForAlgo.mat", tmp_path / "infoForAlgo.mat")
    result = api().inspect_files([str(path) for path in tmp_path.glob("*.mat")])
    assert len(result["recordings"]) == 6
    assert all(recording["metadata"] == {} for recording in result["recordings"])
    assert all(any("ambiguous" in warning for warning in recording["warnings"]) for recording in result["recordings"])


def test_unreadable_participant_file_prevents_external_metadata_pairing(tmp_path):
    import shutil

    shutil.copyfile(HA / "data.mat", tmp_path / "healthy.mat")
    (tmp_path / "impaired.mat").write_bytes(b"not a MATLAB file")
    shutil.copyfile(ROOT / "example_data/data/lab/MS/001/infoForAlgo.mat", tmp_path / "infoForAlgo.mat")
    module = api()
    result = module.inspect_files([str(path) for path in tmp_path.glob("*.mat")])
    assert len(result["recordings"]) == 3
    assert len(result["errors"]) == 1
    assert result["errors"][0]["fileName"] == "impaired.mat"
    assert all(recording["metadata"] == {} for recording in result["recordings"])
    assert all(
        any("Enter heights manually" in warning for warning in recording["warnings"])
        for recording in result["recordings"]
    )
    with pytest.raises(ValueError, match="Sensor height is required"):
        module.analyze_recording(result["recordings"][0]["id"], {"preset": "healthy", "cohort": "HA"})


def test_embedded_metadata_remains_paired_when_another_upload_is_unreadable(tmp_path):
    unreadable = tmp_path / "impaired.mat"
    unreadable.write_bytes(b"not a MATLAB file")
    result = api().inspect_files([str(ROOT / "demo-website/public/samples/healthy.mat"), str(unreadable)])
    assert len(result["errors"]) == 1
    assert len(result["recordings"]) == 1
    assert result["recordings"][0]["metadata"] == pytest.approx({"heightM": 1.59, "sensorHeightM": 0.964})


@pytest.mark.parametrize("preset", ["healthy", "impaired"])
def test_cwa_inspection_defers_samples_and_reports_absent_gyro(preset):
    module = api()
    result = module.inspect_files([str(ROOT / "example_data/data/ax6/example-610-steps.cwa")])
    assert result["errors"] == []
    recording = result["recordings"][0]
    assert recording["sourceFormat"] == "cwa"
    assert recording["samples"] is None
    assert recording["metadata"] == {}
    assert recording["cwa"]["hasGyroscope"] is False
    days = module.cwa_day_windows(recording["id"], "UTC")
    assert len(days["windows"]) == 1
    assert days["windows"][0]["durationSeconds"] == pytest.approx(recording["durationSeconds"])
    with pytest.raises(ValueError, match="gyroscope"):
        module.analyze_recording(
            recording["id"],
            {
                "preset": preset,
                "cohort": "HA",
                "participantHeightM": 1.7,
                "sensorHeightM": 0.9,
                "cwaDay": {"index": 0, "timezone": "UTC"},
            },
        )


def test_mixed_cwa_mat_upload_does_not_guess_external_participant_metadata(tmp_path):
    import shutil

    shutil.copyfile(HA / "data.mat", tmp_path / "healthy.mat")
    shutil.copyfile(ROOT / "example_data/data/ax6/example-610-steps.cwa", tmp_path / "recording.cwa")
    shutil.copyfile(ROOT / "example_data/data/lab/MS/001/infoForAlgo.mat", tmp_path / "infoForAlgo.mat")
    result = api().inspect_files([str(path) for path in tmp_path.iterdir()])
    assert result["errors"] == []
    assert len(result["recordings"]) == 4
    assert all(recording["metadata"] == {} for recording in result["recordings"])


def test_cwa_day_batch_returns_day_errors_without_raising_into_ipython():
    module = api()
    recording = module.inspect_files([str(ROOT / "example_data/data/ax6/example-610-steps.cwa")])["recordings"][0]
    batch = module.start_cwa_day_batch(
        recording["id"],
        {
            "preset": "healthy",
            "cohort": "HA",
            "participantHeightM": 1.7,
            "sensorHeightM": 0.9,
            "timezone": "UTC",
        },
        [0],
    )
    assert batch["totalDays"] == 1
    step = module.next_cwa_day()
    assert step["done"] is False
    assert step["packet"]["day"]["index"] == 0
    assert "gyroscope" in step["packet"]["error"]
    assert "result" not in step["packet"]
    json.dumps(step, allow_nan=False)
    assert module.next_cwa_day() == {"done": True}
    module.start_cwa_day_batch(
        recording["id"],
        {
            "preset": "healthy",
            "cohort": "HA",
            "participantHeightM": 1.7,
            "sensorHeightM": 0.9,
            "timezone": "UTC",
        },
        [0],
    )
    module.cancel_cwa_day_batch()
    assert module.next_cwa_day() == {"done": True}
