def test_loading_example_data(snapshot):
    from examples.data._01_loading_example_data import (
        data_with_reference,
        ref_paras,
        single_trial_with_reference,
        test_list,
    )

    # We round the index, otherwise it will not be properly stored in json due to float representation
    dataset_data = single_trial_with_reference.data["LowerBack"].head()
    dataset_data.index = dataset_data.index.round("ms")
    snapshot.assert_match(dataset_data, "dataset")
    assert len(single_trial_with_reference.raw_reference_parameters_["wb"]) == 1
    functional_data = data_with_reference[test_list[2]].imu_data["LowerBack"].head()
    functional_data.index = functional_data.index.round("ms")
    snapshot.assert_match(functional_data, "functional_interface")
    assert len(data_with_reference[test_list[2]].raw_reference_parameters["wb"]) == 3
    for k, p in ref_paras._asdict().items():
        snapshot.assert_match(p, f"ref_paras_{k}")
