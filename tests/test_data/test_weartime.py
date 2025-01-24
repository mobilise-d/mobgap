from pathlib import Path

from mobgap.data._mobilsed_weartime_loader import load_weartime_from_daily_mcroberts_report


def test_basic_weartime():
    test = load_weartime_from_daily_mcroberts_report(
        Path(
            "/home/arne/Documents/repos/work/mobilised/agg_dmo_data/T3/T3_MinutetByMinute-21May24/20240521_Mobilise-D_CVS_T3_MinutetoMinuteWearTime_classification_23116_minute.csv"
        )
    )
    test
