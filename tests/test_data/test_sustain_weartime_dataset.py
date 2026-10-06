import json
import shutil
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from mobgap.consts import SF_SENSOR_COLS
from mobgap.data import SustainWearTimeDataset, get_example_cwa_data_path, split_by_utc_day
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
    header = cwa_reader_rs.read_metadata(str(CWA_FIXTURE))
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
        fixed_utc_offset_timezone=ax6_module._clock_timezone(header, "Europe/London"),
        resample_hz=header["sample_rate_hz"],
        resample_method="cubic",
    )


def _read_fixture_index() -> pd.DatetimeIndex:
    return _read_fixture_data().index.rename("time")


def _fake_recording_info(start, end, timing=None):
    header = {
        "sample_rate_hz": 100.0,
        "last_change_time_raw": "2020-01-01T12:00:00",
        "start_from_data_raw": pd.Timestamp(start).tz_localize(None).isoformat(),
        "end_from_data_raw": pd.Timestamp(end).tz_localize(None).isoformat(),
    }
    return header, timing or {}


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

    dataset = SustainWearTimeDataset(base_path, splitter=None)

    expected_index = pd.DataFrame(
        [
            {
                "file_path": "weartime_part_a_all/001/example_lowback.cwa",
                "recording_type": "human_movement",
                "participant_id": "001",
                "recording_id": HUMAN_RECORDING_ID,
            },
            {
                "file_path": "weartime_part_b/020/example_lowback.cwa",
                "recording_type": "simulated_movements",
                "participant_id": "020",
                "recording_id": SIMULATED_RECORDING_ID,
            },
        ]
    ).astype({"recording_type": "string", "participant_id": "string", "recording_id": "string"})
    assert_frame_equal(
        dataset.index.drop(columns=["recording", "start_time", "end_time", "recording_day"]), expected_index
    )


def test_sustain_local_output_uses_uk_timezone_in_index_and_metadata(tmp_path):
    dataset = SustainWearTimeDataset(_create_sustain_layout(tmp_path), splitter=None, output_timezone="local")
    datapoint = dataset.get_subset(recording_id=HUMAN_RECORDING_ID)

    assert datapoint.index.start_time.iloc[0] == pd.Timestamp("2012-03-27T11:14:57.500+01:00")
    assert datapoint.data_ss.index[0] == datapoint.index.start_time.iloc[0]
    assert datapoint.cwa_header_["start_from_data"] == datapoint.index.start_time.iloc[0]
    assert datapoint.cwa_timing_report_["start_from_data"] == datapoint.index.start_time.iloc[0]


def test_sustain_naive_reference_times_use_uk_local_time(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    reference_path = base_path / "weartime_part_a_all" / "reference.json"
    reference_row = json.loads(reference_path.read_text())
    reference_row["device_off"] = (
        pd.Timestamp(reference_row["device_off"]).tz_convert("Europe/London").tz_localize(None).isoformat()
    )
    reference_row["device_on"] = (
        pd.Timestamp(reference_row["device_on"]).tz_convert("Europe/London").tz_localize(None).isoformat()
    )
    reference_path.write_text(json.dumps(reference_row) + "\n")

    datapoint = SustainWearTimeDataset(base_path, splitter=None).get_subset(recording_id=HUMAN_RECORDING_ID)

    assert datapoint.reference_nonwear_.iloc[0]["start"] == 10
    assert datapoint.reference_nonwear_.iloc[0]["end"] == 20


def test_sensor_name_is_configurable_and_survives_clone(tmp_path):
    base_path = _create_sustain_layout(tmp_path)

    dataset = SustainWearTimeDataset(base_path, splitter=None, sensor_name="Waist").clone()
    datapoint = dataset.get_subset(recording_id=HUMAN_RECORDING_ID)

    assert datapoint.sensor_name == "Waist"
    assert list(datapoint.data) == ["Waist"]


def test_index_creation_detects_lb_abbreviation(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    human_file = base_path / "weartime_part_a_all" / "001" / "example_lowback.cwa"
    human_file.rename(human_file.with_name("example_lb.cwa"))

    dataset = SustainWearTimeDataset(base_path, splitter=None)

    expected_human_row = {
        "file_path": "weartime_part_a_all/001/example_lb.cwa",
        "recording_type": "human_movement",
        "participant_id": "001",
        "recording_id": "human_movement_001_example_lb",
    }
    assert dataset.index.iloc[0][list(expected_human_row)].to_dict() == expected_human_row


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
    datapoint = SustainWearTimeDataset(
        base_path, splitter=None, warn_thres_for_sampling_rate_deviations_hz=None
    ).get_subset(recording_id=HUMAN_RECORDING_ID)

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
        return _fake_recording_info(timing["start_from_data"], timing["end_from_data"], timing)

    monkeypatch.setattr(ax6_module, "_recording_info", fake_recording_info)

    dataset = SustainWearTimeDataset(base_path, splitter=split_by_utc_day)

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
    assert_frame_equal(dataset.index[expected_index.columns], expected_index)
    assert dataset.index["file_path"].tolist() == [
        "weartime_part_a_all/001/example_lowback.cwa",
        "weartime_part_a_all/001/example_lowback.cwa",
        "weartime_part_a_all/001/example_lowback.cwa",
        "weartime_part_b/020/example_lowback.cwa",
    ]


def test_index_and_data_survive_moving_the_dataset_root(tmp_path):
    original_root = _create_sustain_layout(tmp_path / "original")
    original_index = SustainWearTimeDataset(original_root, splitter=None).index
    relocated_root = tmp_path / "relocated" / "Wear-time"
    relocated_root.parent.mkdir()
    original_root.rename(relocated_root)

    relocated = SustainWearTimeDataset(relocated_root, splitter=None)

    assert_frame_equal(relocated.index, original_index)
    datapoint = relocated.get_subset(file_path="weartime_part_a_all/001/example_lowback.cwa")
    assert len(datapoint.data_ss) == 72472


def test_configured_daily_splitter_omits_short_days(tmp_path, monkeypatch):
    base_path = _create_sustain_layout(tmp_path)

    def fake_recording_info(file_path, _identity):
        if file_path.parent.name == "001":
            start, end = "2020-01-01T23:59:58Z", "2020-01-03T00:00:01Z"
        else:
            start, end = "2020-02-01T00:00:00Z", "2020-02-01T12:00:00Z"
        return _fake_recording_info(start, end)

    monkeypatch.setattr(ax6_module, "_recording_info", fake_recording_info)
    dataset = SustainWearTimeDataset(
        base_path,
        splitter=partial(split_by_utc_day, min_duration=pd.Timedelta(hours=1)),
    )

    assert dataset.clone().index.recording_day.tolist() == ["2020-01-02", "2020-02-01"]


def test_default_splitter_keeps_only_days_with_eight_hours(tmp_path, monkeypatch):
    base_path = _create_sustain_layout(tmp_path)

    def fake_recording_info(file_path, _identity):
        end = "2020-01-01T07:59:59.990Z" if file_path.parent.name == "001" else "2020-01-01T07:59:59Z"
        return _fake_recording_info("2020-01-01T00:00:00Z", end)

    monkeypatch.setattr(ax6_module, "_recording_info", fake_recording_info)
    dataset = SustainWearTimeDataset(base_path)

    assert dataset.clone().index.participant_id.tolist() == ["001"]
    assert dataset.index.end_time.iloc[0] - dataset.index.start_time.iloc[0] == pd.Timedelta(hours=8)


def test_default_splitter_is_independent_between_datasets(tmp_path):
    first = SustainWearTimeDataset(tmp_path)
    second = SustainWearTimeDataset(tmp_path)

    first.splitter.keywords["min_duration"] = pd.Timedelta(hours=1)

    assert second.splitter.keywords["min_duration"] == pd.Timedelta(hours=8)


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
        return _fake_recording_info(timing_report["start_from_data"], timing_report["end_from_data"], timing_report)

    def fake_load_cwa_data(_path, _identity, start_s, end_s, _channels, _rate, start_time, end_time, *_args):
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
        base_path, splitter=split_by_utc_day, warn_thres_for_sampling_rate_deviations_hz=None
    ).get_subset(recording_id=HUMAN_RECORDING_ID, recording_day="2020-01-02")

    data = datapoint.data_ss

    assert data.index.to_list() == [pd.Timestamp("2020-01-02T23:59:59Z")]
    assert cuts == [(2.0, 86402.0)]


@requires_sustain_data
def test_real_dataset_index_smoke():
    dataset = SustainWearTimeDataset(SUSTAIN_DATA_PATH)
    index = dataset.index

    assert not index.empty
    assert {"file_path", "recording", "start_time", "end_time", "recording_day"}.issubset(index.columns)
    assert not index["file_path"].map(lambda path: Path(path).is_absolute()).any()
    assert ((index["end_time"] - index["start_time"]) >= pd.Timedelta(hours=8)).all()


def test_recording_metadata_uses_selected_file_and_sustain_labels(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(base_path, splitter=None).get_subset(recording_id=HUMAN_RECORDING_ID)

    metadata = datapoint.recording_metadata
    assert metadata["recording_id"] == HUMAN_RECORDING_ID
    assert metadata["recording_type"] == "human_movement"
    assert metadata["file_name"] == "example_lowback.cwa"
    assert metadata["measurement_condition"] == "laboratory"
    assert metadata["cwa_header"]["hardware_type"] == "AX3"


def test_human_movement_references_are_snapped_to_sample_boundaries(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(
        base_path, splitter=None, warn_thres_for_sampling_rate_deviations_hz=None
    ).get_subset(recording_id=HUMAN_RECORDING_ID)

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
    datapoint = SustainWearTimeDataset(
        base_path, splitter=None, warn_thres_for_sampling_rate_deviations_hz=None
    ).get_subset(recording_id=HUMAN_RECORDING_ID)

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
    datapoint = SustainWearTimeDataset(
        base_path, splitter=None, warn_thres_for_sampling_rate_deviations_hz=None
    ).get_subset(recording_id=HUMAN_RECORDING_ID)

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
    datapoint = SustainWearTimeDataset(
        base_path, splitter=None, warn_thres_for_sampling_rate_deviations_hz=None
    ).get_subset(recording_id=HUMAN_RECORDING_ID)

    assert len(datapoint.reference_nonwear_) == 1


def test_simulated_movements_are_all_nonwear(tmp_path):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(
        base_path, splitter=None, warn_thres_for_sampling_rate_deviations_hz=None
    ).get_subset(recording_id=SIMULATED_RECORDING_ID)

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


@pytest.mark.parametrize("recording_id", [HUMAN_RECORDING_ID, SIMULATED_RECORDING_ID])
def test_grouping_preserves_selected_recording_metadata_and_references(tmp_path, recording_id):
    base_path = _create_sustain_layout(tmp_path)
    datapoint = SustainWearTimeDataset(
        base_path, splitter=None, warn_thres_for_sampling_rate_deviations_hz=None
    ).get_subset(recording_id=recording_id)
    grouped = datapoint.groupby("recording_id")

    assert grouped.recording_metadata == datapoint.recording_metadata
    assert_frame_equal(grouped.reference_nonwear_, datapoint.reference_nonwear_)
    assert_frame_equal(grouped.reference_weartime_, datapoint.reference_weartime_)
