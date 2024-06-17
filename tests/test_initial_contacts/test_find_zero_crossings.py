import numpy as np
import pytest

from mobgap.initial_contacts._shin_algo_improved import find_zero_crossings


class TestFindZeroCrossings:
    def test_invalid_mode_parameter(self):
        with pytest.raises(ValueError):
            find_zero_crossings(np.ndarray(shape=(1000, 1)), mode="invalid")

    @pytest.mark.parametrize(
        "mode, expected",
        [("both", [0.5, 1.5, 2.5, 3.5]), ("positive_to_negative", [0.5, 2.5]), ("negative_to_positive", [1.5, 3.5])],
    )
    def test_simple_case(self, mode, expected):
        signal = np.array([1, -1, 1, -1, 1])

        output = find_zero_crossings(signal, mode=mode)

        np.testing.assert_allclose(output, np.array(expected))

    @pytest.mark.parametrize(
        "mode, expected",
        [
            ("both", [0.25, 1.75, 2.25, 3.75]),
            ("positive_to_negative", [0.25, 2.25]),
            ("negative_to_positive", [1.75, 3.75]),
        ],
    )
    def test_asymmetric_scale(self, mode, expected):
        signal = np.array([1, -3, 1, -3, 1])

        output = find_zero_crossings(signal, mode=mode)

        np.testing.assert_allclose(output, np.array(expected))

    @pytest.mark.parametrize(
        "mode, expected",
        # Note, that the first "0-crossing" at 0 and the last at 999, is not detected by definition.
        [
            ("both", np.arange(49, 999, 50)),
            ("positive_to_negative", np.arange(49, 999, 100)),
            ("negative_to_positive", np.arange(99, 999, 100)),
        ],
    )
    def test_sin_signal(self, mode, expected):
        # sin with 10 periods, one period is exactly 50 samples
        signal = np.sin(np.linspace(0, 20 * np.pi, 1000))

        output = find_zero_crossings(signal, mode=mode)

        assert np.all(np.diff(output) > 0)
        np.testing.assert_allclose(output.astype("int64"), expected)
