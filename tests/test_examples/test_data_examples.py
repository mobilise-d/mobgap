from pandas._testing import assert_frame_equal


def test_loading_example_data(snapshot):
    from examples.data._01_loading_example_data import (
        data_with_reference,
        ref_paras,
        single_trial_with_reference,
        test_list,
    )

    # We round the index, otherwise it will not be properly stored in json due to float representation
    dataset_data = single_trial_with_reference.data_ss.head()
    dataset_data.index = dataset_data.index.round("ms")
    snapshot.assert_match(dataset_data, "dataset")
    assert len(single_trial_with_reference.raw_reference_parameters_["wb"]) == 1
    functional_data = data_with_reference[test_list[2]].imu_data["LowerBack"].head()
    functional_data.index = functional_data.index.round("ms")
    snapshot.assert_match(functional_data, "functional_interface")
    assert len(data_with_reference[test_list[2]].raw_reference_parameters["wb"]) == 3
    for k, p in ref_paras._asdict().items():
        snapshot.assert_match(p, f"ref_paras_{k}")


def test_reference_data_usage():
    from examples.data._02_working_with_ref_data import gs_iterator, ref_data

    # The final ics should be equivalent to the non-relative ref data, as we add the GS offset during aggregation
    assert_frame_equal(gs_iterator.results_.ic_list, ref_data.ic_list)


def test_custom_dataset(snapshot):
    from examples.data._05_custom_datasets import csv_data

    snapshot.assert_match(csv_data.index, "csv_data_index")
    # A random piece of data as a snapshot
    snapshot.assert_match(csv_data[3].data_ss.head(), "csv_data_data")


def test_dmo_data(snapshot):
    from examples.data._03_dmo_data import dataset, single_participant

    assert len(dataset) == 14
    assert len(single_participant) == 7
    assert single_participant.data.shape == single_participant.data.shape == (2378, 11)

    snapshot.assert_match(single_participant.data.head(), "single_participant_data")
    snapshot.assert_match(single_participant.data_mask.head(), "single_participant_data_mask")

    snapshot.assert_match(dataset.index, "dataset_index")
