def test_assembling_wbs(snapshot):
    from examples.wba._01_assembling_wbs import stride_selection, wb_assembly

    snapshot.assert_match(wb_assembly.annotated_stride_list_, "wba")
    snapshot.assert_match(stride_selection.filtered_stride_list_, "ss")
