from gaitlink.data import get_lab_example_data_path
from tests import PROJECT_ROOT


def test_get_lab_example_data_path():
    path = get_lab_example_data_path("HA", "002")

    assert PROJECT_ROOT / "example_data/" / "data" / "lab" / "HA" / "002" == path
