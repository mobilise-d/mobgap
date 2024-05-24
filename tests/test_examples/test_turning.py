def test_elgohary(snapshot):
    from examples.turning._01_td_elgohary import turning_detector, turning_detector_global, turns

    snapshot.assert_match(turning_detector.turn_list_, "full_recording")
    snapshot.assert_match(turns, "turns_per_gs")
    snapshot.assert_match(turning_detector_global.turn_list_, "full_recording_global_frame")
