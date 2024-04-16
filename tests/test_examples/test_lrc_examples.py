def test_mccamley(snapshot):
    from examples.lrc._01_mccamley import detected_ics

    snapshot.assert_match(detected_ics)
