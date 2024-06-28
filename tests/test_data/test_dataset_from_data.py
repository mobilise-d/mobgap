import pandas as pd
import pytest

from mobgap.data import GaitDatasetFromData


class TestDatasetFromData:
    def test_single_participant_single_sensor(self):
        data = {"p1": {"sensor1": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})}}
        sampling_rate_hz = 100

        dataset = GaitDatasetFromData(data, sampling_rate_hz, index_cols="participant_id")

        assert len(dataset) == 1
        assert dataset.get_subset(participant_id="p1").data["sensor1"] is data["p1"]["sensor1"]
        assert dataset.get_subset(participant_id="p1").sampling_rate_hz == sampling_rate_hz

        with pytest.raises(AttributeError):
            _ = dataset.get_subset(participant_id="p1").participant_metadata

    def test_single_participant_single_sensor_no_col_names(self):
        data = {"p1": {"sensor1": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})}}
        sampling_rate_hz = 100

        dataset = GaitDatasetFromData(data, sampling_rate_hz)

        assert len(dataset) == 1
        assert dataset.get_subset(level_0="p1").data["sensor1"] is data["p1"]["sensor1"]
        assert dataset.get_subset(level_0="p1").sampling_rate_hz == sampling_rate_hz

    def test_multi_participant_single_sensor(self):
        data = {
            "p1": {"sensor1": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})},
            "p2": {"sensor1": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})},
        }

        metadata = {
            "p1": {"age": 20},
            "p2": {"age": 30},
        }
        sampling_rate_hz = 100

        dataset = GaitDatasetFromData(data, sampling_rate_hz, metadata, index_cols="participant_id")

        assert len(dataset) == 2
        assert dataset.get_subset(participant_id="p1").data["sensor1"] is data["p1"]["sensor1"]
        assert dataset.get_subset(participant_id="p2").data["sensor1"] is data["p2"]["sensor1"]
        assert dataset.get_subset(participant_id="p1").sampling_rate_hz == sampling_rate_hz
        assert dataset.get_subset(participant_id="p2").sampling_rate_hz == sampling_rate_hz
        assert dataset.get_subset(participant_id="p1").participant_metadata is metadata["p1"]
        assert dataset.get_subset(participant_id="p2").participant_metadata is metadata["p2"]

    def test_multi_participant_different_sampling_rates(self):
        data = {
            "p1": {"sensor1": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})},
            "p2": {"sensor1": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})},
        }

        sampling_rate_hz = {"p1": 100, "p2": 200}

        dataset = GaitDatasetFromData(data, sampling_rate_hz, index_cols="participant_id")

        assert len(dataset) == 2
        assert dataset.get_subset(participant_id="p1").data["sensor1"] is data["p1"]["sensor1"]
        assert dataset.get_subset(participant_id="p2").data["sensor1"] is data["p2"]["sensor1"]
        assert dataset.get_subset(participant_id="p1").sampling_rate_hz == sampling_rate_hz["p1"]
        assert dataset.get_subset(participant_id="p2").sampling_rate_hz == sampling_rate_hz["p2"]

    def test_tuple_key(self):
        data = {
            ("p1", "s1"): {"pos1": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})},
            ("p2", "s1"): {"pos1": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})},
        }

        sampling_rate_hz = 100
        dataset = GaitDatasetFromData(
            data, sampling_rate_hz, index_cols=["participant_id", "sensor_id"], single_sensor_name="pos1"
        )

        assert dataset.index.shape == (2, 2)
        assert list(dataset.index.columns) == ["participant_id", "sensor_id"]
        assert dataset.get_subset(participant_id="p1", sensor_id="s1").data is data[("p1", "s1")]
        assert dataset.get_subset(participant_id="p2", sensor_id="s1").data is data[("p2", "s1")]
        assert dataset.get_subset(participant_id="p1", sensor_id="s1").data_ss is data[("p1", "s1")]["pos1"]
        assert dataset.get_subset(participant_id="p2", sensor_id="s1").data_ss is data[("p2", "s1")]["pos1"]
        assert dataset.get_subset(participant_id="p1", sensor_id="s1").sampling_rate_hz == sampling_rate_hz

    def test_tuple_key_no_index_cols(self):
        data = {
            ("p1", "s1"): {"pos1": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})},
            ("p2", "s1"): {"pos1": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})},
        }

        sampling_rate_hz = 100
        dataset = GaitDatasetFromData(data, sampling_rate_hz, single_sensor_name="pos1")

        assert dataset.index.shape == (2, 2)
        assert list(dataset.index.columns) == ["level_0", "level_1"]
        assert dataset.get_subset(level_0="p1", level_1="s1").data is data[("p1", "s1")]
        assert dataset.get_subset(level_0="p2", level_1="s1").data is data[("p2", "s1")]
        assert dataset.get_subset(level_0="p1", level_1="s1").data_ss is data[("p1", "s1")]["pos1"]
        assert dataset.get_subset(level_0="p2", level_1="s1").data_ss is data[("p2", "s1")]["pos1"]
        assert dataset.get_subset(level_0="p1", level_1="s1").sampling_rate_hz == sampling_rate_hz
