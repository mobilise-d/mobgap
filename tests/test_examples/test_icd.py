# This is needed to avoid plots to open
import matplotlib

matplotlib.use("Agg")


def test_icd_ionescu(snapshot):
    from examples.icd._01_icd_ionescu import detected_ics

    snapshot.assert_match(detected_ics, "short_trial_output")


def test_icd_shin_improved(snapshot):
    from examples.icd._02_shin_algo import detected_ics

    snapshot.assert_match(detected_ics, "short_trial_output")
