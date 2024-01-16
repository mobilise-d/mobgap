# This is needed to avoid plots to open
import matplotlib

matplotlib.use("Agg")


def test_icd_ionescu(snapshot):
    from examples.icd._01_icd_ionescu import short_trial_output

    snapshot.assert_match(short_trial_output.gsd_list_, "short_trial_output")
