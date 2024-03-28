import numpy as np
from gaitlink.gsd._gsd_pi import find_intersections


class TestIntersect:
    def test_non_overlapping_intervals(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        expected = []
        result = find_intersections(a, b)
        assert np.array_equal(result, expected), "Should return an empty array for non-overlapping intervals"

    def test_overlapping_intervals(self):
        a = np.array([[1, 5], [6, 10]])
        b = np.array([[4, 7], [8, 12]])
        expected = np.array([[4, 5], [6, 7], [8, 10]])
        result = find_intersections(a, b)
        assert np.array_equal(
            result[: len(expected)], expected
        ), "Should return correct intersections for overlapping intervals"
