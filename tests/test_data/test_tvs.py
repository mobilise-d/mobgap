"""Tests for the TVS dataset.

Note, that they can only be run, if you have the TVS dataset available locally.
If not, these tests will be skipped!

To run the tests, make sure you export `MOBGAP_TVS_DATASET_PATH` to the path of the TVS dataset.

The tests are focused on the additional features of the TVS dataset, so we will not repeat tests about data loading.
"""

import pandas as pd
import pytest

from mobgap.data import TVSFreeLivingDataset, TVSLabDataset
from mobgap.utils.misc import get_env_var

TVS_DATA_PATH = get_env_var("MOBGAP_TVS_DATASET_PATH", None)


pytestmark = pytest.mark.skipif(
    not TVS_DATA_PATH,
    reason="TVS dataset path (`MOBGAP_TVS_DATASET_PATH`) not set. Skipping tests.",
)


@pytest.mark.parametrize("dataset", [TVSLabDataset, TVSFreeLivingDataset])
def test_regression_index(snapshot, dataset):
    dataset = dataset(TVS_DATA_PATH)
    snapshot.assert_match(dataset.index)


def test_regression_information(snapshot):
    dataset = TVSLabDataset(TVS_DATA_PATH)
    snapshot.assert_match(dataset.participant_information)


def test_regression_dataquality(snapshot):
    dataset = TVSLabDataset(TVS_DATA_PATH)
    quality = dataset.data_quality
    quality.columns = pd.Index("_".join(c) for c in quality.columns.to_flat_index())
    snapshot.assert_match(quality)


# Note we test with the second slice that the unique_center_id really only contain the participants that are in the
# index after subset.
@pytest.mark.parametrize("selection", [slice(None), slice(0, -20)])
def test_unique_center_id(selection):
    dataset = TVSLabDataset(TVS_DATA_PATH)[selection]
    n_participants = len(dataset.index["participant_id"].unique())
    unique_ids = dataset.unique_center_id
    assert len(unique_ids) == n_participants
    assert set(unique_ids) == set(str(i) for i in range(1, 6))


def test_data_can_be_loaded():
    dataset = TVSLabDataset(
        TVS_DATA_PATH,
    )
    assert dataset[0].data_ss.shape == (6286, 6)


@pytest.mark.parametrize("reference", ["INDIP", "Stereophoto"])
def test_reference_can_be_loaded(reference):
    dataset = TVSLabDataset(TVS_DATA_PATH, reference_system=reference)
    ref_paras = dataset[10].reference_parameters_
    assert len(ref_paras.wb_list) == 1
    assert isinstance(ref_paras.ic_list, pd.DataFrame)
    assert ref_paras.turn_parameters is None
    assert isinstance(ref_paras.stride_parameters, pd.DataFrame)
