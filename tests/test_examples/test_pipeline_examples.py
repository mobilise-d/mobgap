import pandas as pd


def test_gs_iterator(snapshot):
    from examples.pipeline._01_gs_iterator import iterator, long_trial_gs

    assert len(iterator.raw_results_) == len(long_trial_gs)

    s_ids, data = zip(*iterator.inputs_)
    assert s_ids == tuple(long_trial_gs.index)

    snapshot.assert_match(iterator.n_samples_.to_frame(), "n_samples")
    filtered_data = pd.concat(iterator.filtered_data_)
    filtered_data.index = filtered_data.index.round("ms")
    snapshot.assert_match(filtered_data, "filtered_data")
