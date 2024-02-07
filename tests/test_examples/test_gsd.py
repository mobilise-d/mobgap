import numpy as np
import pandas as pd


def test_gsd_iluz(snapshot):
    from examples.gsd._01_gsd_iluz import (
        categorized_intervals,
        long_trial_output,
        long_trial_output_modified,
        matches,
        metrics_all,
        prec_rec_f1_dict,
        short_trial_output,
    )

    snapshot.assert_match(long_trial_output.gs_list_, "long_trial_output")
    snapshot.assert_match(long_trial_output_modified.gs_list_, "long_trial_output_modified")
    snapshot.assert_match(short_trial_output.gs_list_, "short_trial_output")
    snapshot.assert_match(categorized_intervals, "categorized_intervals")
    snapshot.assert_match(np.array(list(prec_rec_f1_dict.values())), "prec_rec_f1_values")
    snapshot.assert_match(pd.DataFrame(metrics_all, index=[0]), "metrics_all")
    snapshot.assert_match(matches, "matches")
