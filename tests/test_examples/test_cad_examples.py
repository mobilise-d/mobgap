def test_cad_from_ic(snapshot):
    from examples.cad._01_cad_from_ic import cad_from_ic

    snapshot.assert_match(cad_from_ic.cad_per_sec_)
