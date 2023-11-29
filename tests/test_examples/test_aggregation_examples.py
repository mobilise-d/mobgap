def test_mobilised_aggregator(snapshot):
    from examples.aggregation._01_mobilised_aggregator import agg, weekly_agg

    snapshot.assert_match(agg.aggregated_data_, "aggregated_data")
    snapshot.assert_match(weekly_agg, "weekly_aggregated_data")
