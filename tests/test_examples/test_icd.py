# This is needed to avoid plots to open
import matplotlib
import pandas as pd

matplotlib.use("Agg")


def test_icd_ionescu(snapshot):
    from examples.icd._01_icd_ionescu import detected_ics

    snapshot.assert_match(detected_ics, "short_trial_output")


def test_icd_shin_improved(snapshot):
    from examples.icd._02_shin_algo import detected_ics

    snapshot.assert_match(detected_ics, "short_trial_output")


def test_icd_hklee_improved(snapshot):
    from examples.icd._03_hklee_algo import detected_ics

    snapshot.assert_match(detected_ics, "short_trial_output")


def test_icd_evaluation(snapshot):
    from examples.icd._04_icd_evaluation import detected_ics, matches, metrics, reference_ics

    snapshot.assert_match(detected_ics, "detected_ics")
    snapshot.assert_match(reference_ics, "reference_ics")
    snapshot.assert_match(matches, "matches")
    snapshot.assert_match(pd.DataFrame(metrics, index=[0]), "metrics")
