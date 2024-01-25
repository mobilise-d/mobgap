import numpy as np
import pytest

from gaitlink.ICD import find_zero_crossings


class Testfind_zero_crossing:
    __test__ = True

    def test_invalid_mode_parameter(self):
        with pytest.raises(ValueError):
            find_zero_crossings(np.ndarray(shape=(1000, 1)), mode="invalid")