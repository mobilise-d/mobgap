from mobgap import PROJECT_ROOT
from mobgap.data import LabExampleDataset, get_all_lab_example_data_paths


def test_get_lab_example_data_path():
    paths = get_all_lab_example_data_paths()

    assert len(paths) == 3

    assert all(p.is_relative_to(PROJECT_ROOT / "example_data/" / "data" / "lab") for p in paths.values())

    assert all((p / "data.mat").is_file() for p in paths.values())
    assert all((p / "infoForAlgo.mat").is_file() for p in paths.values())

    example_path = paths[("HA", "002")]
    assert example_path == PROJECT_ROOT / "example_data/" / "data" / "lab" / "HA" / "002"


class TestLabExampleDataset:
    def test_index(self):
        dataset = LabExampleDataset()
        assert len(dataset) == 9

        assert dataset.index.columns.tolist() == ["cohort", "participant_id", "time_measure", "test", "trial"]

    def test_data_loading(self):
        dataset = LabExampleDataset()

        test_11_subset = dataset.get_subset(test="Test11")

        assert len(test_11_subset[0].data_ss) == 13759
        assert (
            test_11_subset[0].selected_data_file
            == PROJECT_ROOT / "example_data/" / "data" / "lab" / "HA" / "001" / "data.mat"
        )
        assert len(test_11_subset[1].data_ss) == 15984
        assert (
            test_11_subset[1].selected_data_file
            == PROJECT_ROOT / "example_data/" / "data" / "lab" / "HA" / "002" / "data.mat"
        )
