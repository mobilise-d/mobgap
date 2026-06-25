import json
import shutil
import warnings
from pathlib import Path

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from mobgap.consts import GRAV_MS2, SF_SENSOR_COLS
from mobgap.data import SustainWearTimeDataset

cwa_reader_rs = pytest.importorskip("cwa_reader_rs")

HERE = Path(__file__).parent
CWA_FIXTURE = HERE / "data" / "sustain_weartime" / "example-610-steps.cwa"
TIMING_WARNING_MATCH = "effective sampling rate waries considerable"
HUMAN_RECORDING_ID = "human_movement_001_example_lowback"
SIMULATED_RECORDING_ID = "simulated_movements_020_example_lowback"


def _read_fixture_data(**kwargs):
    kwargs = {
        "include_magnetometer": False,
        "include_temperature": True,
        "include_light": False,
        "include_battery": False,
        **kwargs,
    }
    return cwa_reader_rs.read_cwa_file(
        str(CWA_FIXTURE),
        **kwargs,
        resample_hz=cwa_reader_rs.read_header(str(CWA_FIXTURE))["sample_rate_hz"],
        resample_method="cubic",
    )


def _read_fixture_index() -> pd.DatetimeIndex:
    raw_data = _read_fixture_data()
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
                "recording_id": HUMAN_RECORDING_ID,
            },
            {
                "recording_type": "simulated_movements",
                "participant_id": "020",
                "recording_id": SIMULATED_RECORDING_ID,
            },
        ]
    ).astype("string")
    assert_frame_equal(dataset.index, expected_index)


def test_loads_cwa_data_in_sensor_frame(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path).get_subset(recording_id=HUMAN_RECORDING_ID)

    raw_data = _read_fixture_data()
    with pytest.warns(UserWarning, match=TIMING_WARNING_MATCH):
        data = datapoint.data_ss

    assert list(data.columns) == [*SF_SENSOR_COLS, "temperature"]
    assert data.index.tz is not None
    assert data.index.name == "time"
    assert (data.index[1:] - data.index[:-1]).unique().tolist() == [
        pd.to_timedelta(1 / datapoint.sampling_rate_hz, unit="s")
    ]
    with pytest.warns(UserWarning, match=TIMING_WARNING_MATCH):
        assert "LowerBack" in datapoint.data

    assert data.iloc[0]["acc_x"] == pytest.approx(float(raw_data["acc_x"][0]) * GRAV_MS2)
    assert data.iloc[0]["acc_y"] == pytest.approx(float(raw_data["acc_y"][0]) * GRAV_MS2)
    assert data.iloc[0]["acc_z"] == pytest.approx(float(raw_data["acc_z"][0]) * GRAV_MS2)
    assert data.iloc[0]["gyr_x"] == pytest.approx(float(raw_data["gyro_x"][0]))
    assert data.iloc[0]["gyr_y"] == pytest.approx(float(raw_data["gyro_y"][0]))
    assert data.iloc[0]["gyr_z"] == pytest.approx(float(raw_data["gyro_z"][0]))


def test_can_disable_additional_cwa_channels(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path, additional_channels=()).get_subset(recording_id=HUMAN_RECORDING_ID)

    with pytest.warns(UserWarning, match=TIMING_WARNING_MATCH):
        assert list(datapoint.data_ss.columns) == list(SF_SENSOR_COLS)


def test_can_load_all_available_additional_cwa_channels(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path, additional_channels=("temperature", "light", "battery")).get_subset(
        recording_id=HUMAN_RECORDING_ID
    )

    raw_data = _read_fixture_data(
        include_magnetometer=False,
        include_temperature=True,
        include_light=True,
        include_battery=True,
    )
    expected_additional_columns = [column for column in ["temperature", "light", "battery"] if column in raw_data]

    with pytest.warns(UserWarning, match=TIMING_WARNING_MATCH):
        data = datapoint.data_ss
    assert list(data.columns) == [*SF_SENSOR_COLS, *expected_additional_columns]
    for column in expected_additional_columns:
        assert data.iloc[0][column] == pytest.approx(float(raw_data[column][0]))


def test_dataset_unsupported_additional_channel_raises(tmp_path):
    base_path = _create_sustain_layout(tmp_path)

    datapoint = SustainWearTimeDataset(base_path, additional_channels=("magnetometer",)).get_subset(
        recording_id=HUMAN_RECORDING_ID
    )

    assert datapoint.available_additional_channels_ == ("temperature", "light", "battery")
    with pytest.raises(ValueError, match="Unknown additional CWA channels"):
        datapoint.data_ss


def test_unknown_additional_channel_raises(tmp_path):
    base_path = _create_sustain_layout(tmp_path)

    with pytest.raises(ValueError, match="Unknown additional CWA channels"):
        SustainWearTimeDataset(base_path, additional_channels=("temperature", "not_a_channel")).get_subset(
            recording_id=HUMAN_RECORDING_ID
        ).data_ss


def test_cwa_header_is_available_on_single_recording(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path).get_subset(recording_id=HUMAN_RECORDING_ID)

    header = datapoint.cwa_header_
    timing_report = datapoint.cwa_timing_report_

    assert header["hardware_type"] == "AX3"
    assert header["sample_rate_hz"] == 100.0
    assert timing_report["samplingrate_hz_from_header"] == 100.0
    assert abs(timing_report["samplingrate_hz_from_data"] - timing_report["samplingrate_hz_from_header"]) > 0.2
    assert datapoint.recording_metadata["cwa_header"] == header


def test_warns_when_effective_sampling_rate_deviates(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path).get_subset(recording_id=HUMAN_RECORDING_ID)

    with pytest.warns(UserWarning, match=TIMING_WARNING_MATCH):
        datapoint.data_ss


def test_sampling_rate_deviation_warning_threshold_can_be_adjusted(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path, warn_thres_for_sampling_rate_deviations_hz=10.0).get_subset(
        recording_id=HUMAN_RECORDING_ID
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        datapoint.data_ss

    assert not [warning for warning in caught_warnings if TIMING_WARNING_MATCH in str(warning.message)]


def test_human_movement_references_are_snapped_to_sample_boundaries(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path, warn_thres_for_sampling_rate_deviations_hz=None).get_subset(
        recording_id=HUMAN_RECORDING_ID
    )

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


def test_missing_reference_timestamps_raise(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    reference_row = {
        "id": "1",
        "sensor": "lowerback",
        "device_off": None,
        "device_on": None,
        "wear_status": "non_wear",
    }
    (base_path / "weartime_part_a_all" / "reference.json").write_text(json.dumps(reference_row) + "\n")
    datapoint = SustainWearTimeDataset(base_path, warn_thres_for_sampling_rate_deviations_hz=None).get_subset(
        recording_id=HUMAN_RECORDING_ID
    )

    with pytest.raises(ValueError, match="missing `device_off` or `device_on` timestamps"):
        datapoint.reference_nonwear_


def test_missing_reference_timestamps_for_unrelated_recording_are_ignored(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    valid_reference_row = json.loads((base_path / "weartime_part_a_all" / "reference.json").read_text())
    unrelated_reference_row = {
        "id": "999",
        "sensor": "lowerback",
        "device_off": None,
        "device_on": None,
        "wear_status": "non_wear",
    }
    (base_path / "weartime_part_a_all" / "reference.json").write_text(
        "\n".join(json.dumps(row) for row in [valid_reference_row, unrelated_reference_row]) + "\n"
    )
    datapoint = SustainWearTimeDataset(base_path, warn_thres_for_sampling_rate_deviations_hz=None).get_subset(
        recording_id=HUMAN_RECORDING_ID
    )

    assert len(datapoint.reference_nonwear_) == 1


def test_simulated_movements_are_all_nonwear(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path, warn_thres_for_sampling_rate_deviations_hz=None).get_subset(
        recording_id=SIMULATED_RECORDING_ID
    )

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
