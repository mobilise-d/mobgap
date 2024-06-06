def test_zjilstra(snapshot):
    from examples.stride_length._01_sl_zijlstra import sl_zijlstra, sl_zijlstra_reoriented

    snapshot.assert_match(sl_zijlstra.stride_length_per_sec_, "sl_zijlstra")
    snapshot.assert_match(sl_zijlstra_reoriented.stride_length_per_sec_, "sl_zijlstra_reoriented")
