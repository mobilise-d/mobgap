from gaitlink.data import LabExampleDataset, get_all_lab_example_data_paths
from tests import PROJECT_ROOT


def test_get_lab_example_data_path():
    paths = get_all_lab_example_data_paths()

    assert len(paths) == 3

    assert all(p.is_relative_to(PROJECT_ROOT / "example_data/" / "data" / "lab") for p in paths.values())

    assert all((p / "data.mat").is_file() for p in paths.values())
    assert all((p / "infoForAlgo.mat").is_file() for p in paths.values())

    example_path = paths[("HA", "002")]
    assert PROJECT_ROOT / "example_data/" / "data" / "lab" / "HA" / "002" == example_path


class TestLabExampleDataset:
    def test_index(self):
        dataset = LabExampleDataset()
        assert len(dataset) == 9

        assert dataset.index.columns.tolist() == ["cohort", "participant_id", "time_measure", "test", "trial"]
