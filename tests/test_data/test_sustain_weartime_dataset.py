import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from mobgap.consts import GRAV_MS2, SF_SENSOR_COLS
from mobgap.data import SustainWearTimeDataset

cwa_reader_rs = pytest.importorskip("cwa_reader_rs")

HERE = Path(__file__).parent
CWA_FIXTURE = HERE / "data" / "sustain_weartime" / "example-610-steps.cwa"


def _read_fixture_index() -> pd.DatetimeIndex:
    raw_data = cwa_reader_rs.read_cwa_file(
        str(CWA_FIXTURE),
        include_magnetometer=False,
        include_temperature=True,
        include_light=False,
        include_battery=False,
    )
    return pd.DatetimeIndex(pd.to_datetime(raw_data["timestamp"], unit="us", utc=True), name="time")


def _create_sustain_layout(tmp_path: Path) -> Path:
    base_path = tmp_path / "Wear-time"
    human_folder = base_path / "weartime_part_a_all" / "001"
    simulated_folder = base_path / "weartime_part_b" / "020"
    human_folder.mkdir(parents=True)
    simulated_folder.mkdir(parents=True)

    shutil.copy(CWA_FIXTURE, human_folder / "example_lowback.cwa")
    shutil.copy(CWA_FIXTURE, simulated_folder / "example_lowback.cwa")

    data_index = _read_fixture_index()
    sample_period = data_index[11] - data_index[10]
    reference_row = {
        "id": "1",
        "sensor": "lowerback",
        "device_off": (data_index[10] + sample_period * 0.3).isoformat(),
        "device_on": (data_index[20] + sample_period * 0.2).isoformat(),
        "wear_status": "non_wear",
    }
    (base_path / "weartime_part_a_all" / "reference.json").write_text(json.dumps(reference_row) + "\n")
    return base_path


def test_index_creation(tmp_path):
    base_path = _create_sustain_layout(tmp_path)

    dataset = SustainWearTimeDataset(base_path)

    expected_index = pd.DataFrame(
        [
            {
                "recording_type": "human_movement",
                "participant_id": "001",
                "sensor_position": "lowerback",
                "recording_id": "human_movement_001_lowerback",
            },
            {
                "recording_type": "simulated_movements",
                "participant_id": "020",
                "sensor_position": "lowerback",
                "recording_id": "simulated_movements_020_lowerback",
            },
        ]
    ).astype("string")
    assert_frame_equal(dataset.index, expected_index)


def test_loads_cwa_data_in_sensor_frame(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path).get_subset(recording_id="human_movement_001_lowerback")

    raw_data = cwa_reader_rs.read_cwa_file(
        str(CWA_FIXTURE),
        include_magnetometer=False,
        include_temperature=True,
        include_light=False,
        include_battery=False,
    )
    data = datapoint.data_ss

    assert list(data.columns) == [*SF_SENSOR_COLS, "temperature"]
    assert data.index.tz is not None
    assert data.index.name == "time"
    assert "LowerBack" in datapoint.data

    assert data.iloc[0]["acc_x"] == pytest.approx(float(raw_data["acc_x"][0]) * GRAV_MS2)
    assert data.iloc[0]["acc_y"] == pytest.approx(float(raw_data["acc_y"][0]) * GRAV_MS2)
    assert data.iloc[0]["acc_z"] == pytest.approx(float(raw_data["acc_z"][0]) * GRAV_MS2)
    assert data.iloc[0]["gyr_x"] == pytest.approx(float(raw_data["gyro_x"][0]))
    assert data.iloc[0]["gyr_y"] == pytest.approx(float(raw_data["gyro_y"][0]))
    assert data.iloc[0]["gyr_z"] == pytest.approx(float(raw_data["gyro_z"][0]))


def test_human_movement_references_are_snapped_to_sample_boundaries(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path).get_subset(recording_id="human_movement_001_lowerback")

    data = datapoint.data_ss
    nonwear = datapoint.reference_nonwear_

    assert nonwear.index.name == "nonwear_id"
    assert nonwear.iloc[0]["start"] == 10
    assert nonwear.iloc[0]["end"] == 20
    assert nonwear.iloc[0]["duration"] == 10
    assert nonwear.iloc[0]["start_dt"] == data.index[10]
    assert nonwear.iloc[0]["end_dt"] == data.index[20]
    assert nonwear.iloc[0]["duration_s"] == pytest.approx(10 / datapoint.sampling_rate_hz)

    weartime = datapoint.reference_weartime_
    assert weartime[["start", "end", "duration"]].to_dict("records") == [
        {"start": 0, "end": 10, "duration": 10},
        {"start": 20, "end": len(data), "duration": len(data) - 20},
    ]


def test_simulated_movements_are_all_nonwear(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path).get_subset(recording_id="simulated_movements_020_lowerback")

    data = datapoint.data_ss
    nonwear = datapoint.reference_nonwear_
    weartime = datapoint.reference_weartime_

    assert nonwear[["start", "end", "duration"]].to_dict("records") == [
        {"start": 0, "end": len(data), "duration": len(data)}
    ]
    assert nonwear.iloc[0]["start_dt"] == data.index[0]
    assert nonwear.iloc[0]["end_dt"] == data.index[-1] + pd.to_timedelta(1 / datapoint.sampling_rate_hz, unit="s")

    assert list(weartime.columns) == ["start", "end", "duration", "start_dt", "end_dt", "duration_s"]
    assert weartime.empty
