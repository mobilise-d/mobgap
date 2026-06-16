import json
from pathlib import Path

import pandas as pd
import pytest
from tpcp import Pipeline

from mobgap.data import LabExampleDataset
from mobgap.signal_based import MobilisedSDMO
from mobgap.utils.conversions import to_body_frame

from mobgap.signal_based import RegularitySymmetry, SampleEntropy, Jerk

SNAPSHOT_DIR = Path(__file__).parent / "snapshot"


def test_mobilised_sdmo_structure():
    sdmo = MobilisedSDMO()
    assert issubclass(type(sdmo), Pipeline)
    assert hasattr(sdmo, "calculate"), "MobilisedSDMO missing calculate method"
    assert callable(sdmo.calculate), "MobilisedSDMO.calculate is not callable"
    assert "calculators" in sdmo._composite_params
    assert hasattr(sdmo, "_composite_params")


@pytest.fixture
def example_walking_bout():
    data = LabExampleDataset(reference_system="INDIP").get_subset(test="Test11")[0]
    wb_id = data.reference_parameters_relative_to_wb_.turn_parameters.index.get_level_values(0)[0]
    turn_list = data.reference_parameters_relative_to_wb_.turn_parameters.loc[wb_id]
    stride_list = data.reference_parameters_relative_to_wb_.stride_parameters.loc[wb_id].rename(
        columns={"duration_s": "stride_duration_s", "length_m": "stride_length_m"}
    )
    stride_list["cadence_spm"] = 60 * stride_list["speed_mps"] / stride_list["stride_length_m"]
    sampling_rate_hz = data.sampling_rate_hz
    data_start = data.reference_parameters_.stride_parameters.loc[wb_id][["start", "end"]].min().min()
    data_end = data.reference_parameters_.stride_parameters.loc[wb_id][["start", "end"]].max().max()
    data = to_body_frame(data.data_ss.iloc[data_start:data_end])
    return data, stride_list, turn_list, sampling_rate_hz


def test_mobilised_sdmo_regression(example_walking_bout, request):
    data, stride_list, turn_list, sampling_rate_hz = example_walking_bout

    sdmo = MobilisedSDMO()
    result = sdmo.calculate(
        data,
        stride_list=stride_list,
        turn_list=turn_list,
        sampling_rate_hz=sampling_rate_hz,
        replicate_matlab=True,
    )

    result_dict = result.signal_based_parameters.iloc[0].to_dict()

    snapshot_file = SNAPSHOT_DIR / f"{request.node.name}.json"

    if not snapshot_file.exists():
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        with open(snapshot_file, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        pytest.skip(f"Snapshot created: {snapshot_file}")

    with open(snapshot_file) as f:
        expected = json.load(f)

    result_series = pd.Series(result_dict)
    expected_series = pd.Series(expected)

    pd.testing.assert_series_equal(
        result_series,
        expected_series,
        rtol=1e-5,
        atol=1e-5,
        check_dtype=False,
    )

def test_mobilised_sdmo_runs_with_defaults(example_walking_bout):
    data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
    sdmo = MobilisedSDMO()
    result = sdmo.calculate(
        data=data,
        sampling_rate_hz=sampling_rate_hz,
        stride_list=stride_list,
        turn_list=turn_list,
        replicate_matlab=True,
    )
    df = result.signal_based_parameters
    assert df.shape[0] == 1
    assert df.shape[1] > 0

    expected_columns = [
        "sample_entropy_acc_is",
        "harmonic_ratio_acc_is",
        "harmonic_ratio_acc_pa",
        "sd_acc_is", "range_acc_is",
        "jerk_acc_is", "jerk_acc_ml", "jerk_acc_pa",
        "rms_acc_is", "rms_total_acc",
        "step_regularity_is",
        "amplitude_is",
        "cv_stride_length_m",
        "turn_mean_ang_vel",
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"

def test_mobilised_sdmo_with_custom_calculators(example_walking_bout):
    data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
    custom = (
        ("sample_entropy", SampleEntropy(acc_columns=["acc_ml"])),
        ("regularity_symmetry", RegularitySymmetry()),
        ("jerk", Jerk(gyr_columns=["gyr_is"])),
    )
    sdmo = MobilisedSDMO(calculators=custom)
    result = sdmo.calculate(
        data=data,
        sampling_rate_hz=sampling_rate_hz,
        stride_list=stride_list,
        turn_list=turn_list,
        replicate_matlab=False
    )
    df = result.signal_based_parameters
    assert "sample_entropy_acc_ml" in df.columns
    assert "step_regularity_is" in df.columns
    assert "jerk_gyr_is" in df.columns
    assert "jerk_gyr_ml" not in df.columns
    assert "jerk_acc_is" not in df.columns

