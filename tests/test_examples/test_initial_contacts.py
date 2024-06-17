# This is needed to avoid plots to open
import matplotlib
import pandas as pd

matplotlib.use("Agg")


def test_icd_ionescu(snapshot):
    from examples.initial_contacts._01_icd_ionescu import detected_ics

    snapshot.assert_match(detected_ics, "short_trial_output")


def test_icd_shin_improved(snapshot):
    from examples.initial_contacts._02_shin_algo import detected_ics

    snapshot.assert_match(detected_ics, "short_trial_output")


def test_icd_hklee_improved(snapshot):
    from examples.initial_contacts._03_hklee_algo import detected_ics

    snapshot.assert_match(detected_ics, "short_trial_output")


def test_icd_evaluation(snapshot):
    from examples.initial_contacts._04_icd_evaluation import matches_per_wb, metrics_all

    snapshot.assert_match(matches_per_wb, "matches")
    snapshot.assert_match(pd.DataFrame(metrics_all, index=[0]), "metrics")
