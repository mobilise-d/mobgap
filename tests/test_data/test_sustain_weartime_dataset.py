import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from mobgap.consts import SF_SENSOR_COLS
from mobgap.data import SustainWearTimeDataset, get_example_cwa_data_path
from mobgap.data import _sustain_weartime_dataset as sustain_dataset
from mobgap.data import ax6 as ax6_module
from mobgap.utils.misc import get_env_var

cwa_reader_rs = pytest.importorskip("cwa_reader_rs")

CWA_FIXTURE = get_example_cwa_data_path()
HUMAN_RECORDING_ID = "human_movement_001_example_lowback"
SIMULATED_RECORDING_ID = "simulated_movements_020_example_lowback"
SUSTAIN_DATA_PATH = get_env_var("MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH", None)
requires_sustain_data = pytest.mark.skipif(
    not SUSTAIN_DATA_PATH,
    reason="SUSTAIN wear-time dataset path (`MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH`) not set. Skipping test.",
)


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
                "file_path": str(base_path / "weartime_part_a_all" / "001" / "example_lowback.cwa"),
                "recording_type": "human_movement",
                "participant_id": "001",
                "recording_id": HUMAN_RECORDING_ID,
            },
            {
                "file_path": str(base_path / "weartime_part_b" / "020" / "example_lowback.cwa"),
                "recording_type": "simulated_movements",
                "participant_id": "020",
                "recording_id": SIMULATED_RECORDING_ID,
            },
        ]
    ).astype({"recording_type": "string", "participant_id": "string", "recording_id": "string"})
    assert_frame_equal(dataset.index, expected_index)


def test_index_creation_detects_lb_abbreviation(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    human_file = base_path / "weartime_part_a_all" / "001" / "example_lowback.cwa"
    human_file.rename(human_file.with_name("example_lb.cwa"))

    dataset = SustainWearTimeDataset(base_path)

    expected_human_row = {
        "file_path": str(human_file.with_name("example_lb.cwa")),
        "recording_type": "human_movement",
        "participant_id": "001",
        "recording_id": "human_movement_001_example_lb",
    }
    assert dataset.index.iloc[0].to_dict() == expected_human_row


def test_non_lowerback_reference_rows_are_filtered_before_timestamp_parsing(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    valid_reference_row = json.loads((base_path / "weartime_part_a_all" / "reference.json").read_text())
    ignored_reference_row = {
        "id": "1",
        "sensor": "wrist",
        "device_off": "not-a-date",
        "device_on": "also-not-a-date",
        "wear_status": "non_wear",
    }
    (base_path / "weartime_part_a_all" / "reference.json").write_text(
        "\n".join(json.dumps(row) for row in [valid_reference_row, ignored_reference_row]) + "\n"
    )
    datapoint = SustainWearTimeDataset(base_path, warn_thres_for_sampling_rate_deviations_hz=None).get_subset(
        recording_id=HUMAN_RECORDING_ID
    )

    assert len(datapoint.reference_nonwear_) == 1


def test_split_by_day_index_creation(tmp_path, monkeypatch):
    base_path = _create_sustain_layout(tmp_path)

    def fake_recording_info(file_path, _identity):
        participant_id = file_path.parent.name
        if participant_id == "001":
            timing = {
                "start_from_data": "2020-01-01T23:59:58+00:00",
                "end_from_data": "2020-01-03T00:00:01+00:00",
                "samplingrate_hz_from_header": 100.0,
                "samplingrate_hz_from_data": 100.0,
            }
        else:
            timing = {
                "start_from_data": "2020-02-01T00:00:00+00:00",
                "end_from_data": "2020-02-01T12:00:00+00:00",
                "samplingrate_hz_from_header": 100.0,
                "samplingrate_hz_from_data": 100.0,
            }
        return {"sample_rate_hz": 100.0}, timing

    monkeypatch.setattr(ax6_module, "_recording_info", fake_recording_info)

    dataset = SustainWearTimeDataset(base_path, split_by_day=True)

    expected_index = pd.DataFrame(
        [
            {
                "recording_type": "human_movement",
                "participant_id": "001",
                "recording_id": HUMAN_RECORDING_ID,
                "recording_day": "2020-01-01",
            },
            {
                "recording_type": "human_movement",
                "participant_id": "001",
                "recording_id": HUMAN_RECORDING_ID,
                "recording_day": "2020-01-02",
            },
            {
                "recording_type": "human_movement",
                "participant_id": "001",
                "recording_id": HUMAN_RECORDING_ID,
                "recording_day": "2020-01-03",
            },
            {
                "recording_type": "simulated_movements",
                "participant_id": "020",
                "recording_id": SIMULATED_RECORDING_ID,
                "recording_day": "2020-02-01",
            },
        ]
    ).astype("string")
    assert_frame_equal(dataset.index.drop(columns="file_path"), expected_index)
    assert dataset.index["file_path"].tolist() == [
        str(base_path / "weartime_part_a_all" / "001" / "example_lowback.cwa"),
        str(base_path / "weartime_part_a_all" / "001" / "example_lowback.cwa"),
        str(base_path / "weartime_part_a_all" / "001" / "example_lowback.cwa"),
        str(base_path / "weartime_part_b" / "020" / "example_lowback.cwa"),
    ]


def test_split_by_day_loads_selected_day_with_seconds_cut(tmp_path, monkeypatch):
    base_path = _create_sustain_layout(tmp_path)
    timing_report = {
        "start_from_data": "2020-01-01T23:59:58+00:00",
        "end_from_data": "2020-01-03T00:00:01+00:00",
        "samplingrate_hz_from_header": 100.0,
        "samplingrate_hz_from_data": 100.0,
    }
    cuts = []

    def fake_recording_info(_file_path, _identity):
        return {"sample_rate_hz": 100.0}, timing_report

    def fake_load_cwa_data(_path, _identity, start_s, end_s, _channels, _rate, start_time, end_time):
        cuts.append((start_s, end_s))
        data = pd.DataFrame(
            [[0.0] * (len(SF_SENSOR_COLS) + 1), [1.0] * (len(SF_SENSOR_COLS) + 1)],
            columns=[*SF_SENSOR_COLS, "temperature"],
            index=pd.DatetimeIndex(["2020-01-02T23:59:59Z", "2020-01-03T00:00:00Z"], name="time"),
        )
        return data.loc[(data.index >= start_time) & (data.index < end_time)]

    monkeypatch.setattr(ax6_module, "_recording_info", fake_recording_info)
    monkeypatch.setattr(ax6_module, "_load_cwa_data", fake_load_cwa_data)

    datapoint = SustainWearTimeDataset(
        base_path, split_by_day=True, warn_thres_for_sampling_rate_deviations_hz=None
    ).get_subset(recording_id=HUMAN_RECORDING_ID, recording_day="2020-01-02")

    data = datapoint.data_ss

    assert data.index.to_list() == [pd.Timestamp("2020-01-02T23:59:59Z")]
    assert cuts == [(2.0, 86402.0)]


def test_n_samples_matches_loaded_recording_length(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path, warn_thres_for_sampling_rate_deviations_hz=None).get_subset(
        recording_id=HUMAN_RECORDING_ID
    )

    assert datapoint.n_samples == len(datapoint.data_ss)


def test_split_by_day_n_samples_matches_loaded_recording_length(tmp_path, monkeypatch):
    base_path = _create_sustain_layout(tmp_path)
    timing_report = {
        "start_from_data": "2020-01-01T23:59:58+00:00",
        "end_from_data": "2020-01-02T00:00:01+00:00",
        "duration_s_from_data": 3.0,
        "samplingrate_hz_from_header": 1.0,
        "samplingrate_hz_from_data": 1.0,
    }

    def fake_recording_info(_file_path, _identity):
        return {"sample_rate_hz": 1.0}, timing_report

    def fake_load_cwa_data(_path, _identity, _start_s, _end_s, _channels, _rate, start_time, end_time):
        data = pd.DataFrame(
            [[0.0] * (len(SF_SENSOR_COLS) + 1), [1.0] * (len(SF_SENSOR_COLS) + 1)],
            columns=[*SF_SENSOR_COLS, "temperature"],
            index=pd.DatetimeIndex(["2020-01-02T00:00:00Z", "2020-01-02T00:00:01Z"], name="time"),
        )
        return data.loc[(data.index >= start_time) & (data.index < end_time)]

    monkeypatch.setattr(ax6_module, "_recording_info", fake_recording_info)
    monkeypatch.setattr(ax6_module, "_load_cwa_data", fake_load_cwa_data)

    datapoint = SustainWearTimeDataset(
        base_path, split_by_day=True, warn_thres_for_sampling_rate_deviations_hz=None
    ).get_subset(recording_id=HUMAN_RECORDING_ID, recording_day="2020-01-02")

    assert datapoint.n_samples == 2
    assert datapoint.n_samples == len(datapoint.data_ss)


def test_split_by_day_n_samples_handles_floating_point_boundary_artifacts():
    assert sustain_dataset._sample_count_from_time_bounds_s(0.0, 86_399.999999999, 100.0) == 8_640_000


def test_split_by_day_n_samples_handles_unaligned_partial_day_boundaries():
    assert sustain_dataset._sample_count_from_time_bounds_s(0.6, 130.2, 1.0) == 130


@requires_sustain_data
def test_real_dataset_regression_index(snapshot):
    dataset = SustainWearTimeDataset(SUSTAIN_DATA_PATH)
    split_dataset = SustainWearTimeDataset(SUSTAIN_DATA_PATH, split_by_day=True)

    snapshot.assert_match(dataset.index.drop(columns="file_path"), "recording")
    snapshot.assert_match(split_dataset.index.drop(columns="file_path"), "split_by_day")


@requires_sustain_data
def test_real_dataset_split_by_day_matches_full_recording_for_single_participant():
    dataset = SustainWearTimeDataset(
        SUSTAIN_DATA_PATH,
        additional_sensors_enabled=(),
        warn_thres_for_sampling_rate_deviations_hz=None,
    )
    split_dataset = SustainWearTimeDataset(
        SUSTAIN_DATA_PATH,
        additional_sensors_enabled=(),
        warn_thres_for_sampling_rate_deviations_hz=None,
        split_by_day=True,
    )
    recording_cols = ["recording_type", "participant_id", "recording_id"]
    split_index = split_dataset.index
    n_days_per_recording = split_index.groupby(recording_cols, sort=False).size()
    multi_day_recordings = n_days_per_recording[n_days_per_recording > 1]
    if multi_day_recordings.empty:
        pytest.skip("No multi-day recording found in the SUSTAIN wear-time dataset.")

    participant_id = multi_day_recordings.index[0][1]
    participant_index = dataset.index[dataset.index["participant_id"] == participant_id]

    for recording_row in participant_index.to_dict("records"):
        full_data = dataset.get_subset(**recording_row).data_ss
        split_recording_index = split_index[
            (split_index["recording_type"] == recording_row["recording_type"])
            & (split_index["participant_id"] == recording_row["participant_id"])
            & (split_index["recording_id"] == recording_row["recording_id"])
        ]
        split_data = pd.concat(
            split_dataset.get_subset(index=recording_day.to_frame().T).data_ss
            for _, recording_day in split_recording_index.iterrows()
        )

        assert_frame_equal(split_data, full_data)


def test_recording_metadata_uses_selected_file_and_sustain_labels(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path).get_subset(recording_id=HUMAN_RECORDING_ID)

    metadata = datapoint.recording_metadata
    assert metadata["recording_id"] == HUMAN_RECORDING_ID
    assert metadata["recording_type"] == "human_movement"
    assert metadata["file_name"] == "example_lowback.cwa"
    assert metadata["measurement_condition"] == "laboratory"
    assert metadata["cwa_header"]["hardware_type"] == "AX3"


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


def test_missing_reference_device_off_timestamps_raise(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    data_index = _read_fixture_index()
    reference_row = {
        "id": "1",
        "sensor": "lowerback",
        "device_off": None,
        "device_on": data_index[20].isoformat(),
        "wear_status": "non_wear",
    }
    (base_path / "weartime_part_a_all" / "reference.json").write_text(json.dumps(reference_row) + "\n")
    datapoint = SustainWearTimeDataset(base_path, warn_thres_for_sampling_rate_deviations_hz=None).get_subset(
        recording_id=HUMAN_RECORDING_ID
    )

    with pytest.raises(ValueError, match="missing `device_off` timestamps"):
        datapoint.reference_nonwear_


def test_missing_reference_device_on_extends_nonwear_to_end_of_data(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    data_index = _read_fixture_index()
    reference_row = {
        "id": "1",
        "sensor": "lowerback",
        "device_off": data_index[10].isoformat(),
        "device_on": None,
        "wear_status": "non_wear",
    }
    (base_path / "weartime_part_a_all" / "reference.json").write_text(json.dumps(reference_row) + "\n")
    datapoint = SustainWearTimeDataset(base_path, warn_thres_for_sampling_rate_deviations_hz=None).get_subset(
        recording_id=HUMAN_RECORDING_ID
    )

    data = datapoint.data_ss
    nonwear = datapoint.reference_nonwear_

    assert nonwear[["start", "end", "duration"]].to_dict("records") == [
        {"start": 10, "end": len(data), "duration": len(data) - 10}
    ]
    assert nonwear.iloc[0]["start_dt"] == data.index[10]
    assert nonwear.iloc[0]["end_dt"] == data.index[-1] + pd.to_timedelta(1 / datapoint.sampling_rate_hz, unit="s")

    weartime = datapoint.reference_weartime_
    assert weartime[["start", "end", "duration"]].to_dict("records") == [{"start": 0, "end": 10, "duration": 10}]


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
