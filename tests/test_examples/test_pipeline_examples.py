import pandas as pd


def test_gs_iterator(snapshot):
    from examples.pipeline._01_gs_iterator import custom_iterator, iterator, long_trial_gs

    assert len(custom_iterator.raw_results_) == len(long_trial_gs)

    wb_ids = tuple(v.input[0].id for v in custom_iterator.raw_results_)
    assert wb_ids == tuple(long_trial_gs.index)

    snapshot.assert_match(custom_iterator.results_.n_samples.to_frame(), "n_samples")
    filtered_data = pd.concat(custom_iterator.results_.filtered_data)
    filtered_data.index = filtered_data.index.round("ms")
    snapshot.assert_match(filtered_data, "filtered_data")

    snapshot.assert_match(iterator.results_.ic_list, "initial_contacts")
    snapshot.assert_match(iterator.results_.cad_per_sec, "cadence")
