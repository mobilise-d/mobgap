from pathlib import Path

import pytest
from statsmodels.compat.pandas import assert_series_equal

from mobgap.data._mobilsed_weartime_loader import load_weartime_from_daily_mcroberts_report

HERE = Path(__file__).parent


@pytest.mark.parametrize("waking_hours", [(0, 24), (1, 23), (7, 22)])
def test_basic_weartime(snapshot, waking_hours):
    output = load_weartime_from_daily_mcroberts_report(
        HERE / "data/weartime/example_weartime_min_by_min_report.csv", waking_hours
    )
    snapshot.assert_match(output.reset_index().astype({"visit_date": "string"}), name=waking_hours)

    max_hours = waking_hours[1] - waking_hours[0] + 1
    assert (output.total_worn_during_waking_h <= max_hours).all()

    if waking_hours == (0, 24):
        assert_series_equal(output.total_worn_h, output.total_worn_during_waking_h, check_names=False)
