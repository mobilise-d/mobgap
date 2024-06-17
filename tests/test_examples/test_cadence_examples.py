def test_cad_from_ic(snapshot):
    from examples.cadence._01_cad_from_ic import cad_from_ic, cad_from_ic_detector

    snapshot.assert_match(cad_from_ic.cadence_per_sec_)
    snapshot.assert_match(cad_from_ic_detector.cadence_per_sec_)
