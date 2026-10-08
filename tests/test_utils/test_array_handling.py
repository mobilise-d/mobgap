import numpy as np
import pytest

from mobgap.utils.array_handling import (
    sliding_window_view,
)


class TestSlidingWindowView:
    def test_no_overlap(self):
        view = sliding_window_view(np.arange(14), window_size_samples=3, overlap_samples=0)

        assert view.shape == (4, 3)
        assert np.all(view == np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))

    def test_overlap(self):
        view = sliding_window_view(np.arange(14), window_size_samples=3, overlap_samples=1)

        assert view.shape == (6, 3)
        assert np.all(view == np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 10], [10, 11, 12]]))

    def test_nd_array(self):
        data = np.arange(112).reshape(14, 2, 4)
        view = sliding_window_view(data, window_size_samples=3, overlap_samples=1)

        assert view.shape == (6, 3, 2, 4)
        assert np.all(view[:, :, 0] == sliding_window_view(data[:, 0], window_size_samples=3, overlap_samples=1))
        assert np.all(view[:, :, 1] == sliding_window_view(data[:, 1], window_size_samples=3, overlap_samples=1))

    def test_error_overlap_larger_than_window(self):
        with pytest.raises(ValueError):
            sliding_window_view(np.arange(14), window_size_samples=3, overlap_samples=4)

    @pytest.mark.parametrize("step", [2, -1, -2])
    def test_noncontiguous_views_keep_values_and_remain_read_only(self, step):
        data = np.arange(120).reshape(20, 2, 3)[::step]
        view = sliding_window_view(data, window_size_samples=3, overlap_samples=1)
        expected = np.stack([data[start : start + 3] for start in range(0, len(data) - 2, 2)])
        np.testing.assert_array_equal(view, expected)
        assert np.shares_memory(view, data)
        assert not view.flags.writeable

    def test_half_day_window_view_does_not_require_every_one_sample_offset(self):
        # The input has one allocated value. On 32-bit NumPy, the unused view
        # of every offset would describe over 10 GB and fail before subsampling.
        data = np.broadcast_to(np.array([7.0]), (4_319_378,))
        view = sliding_window_view(data, window_size_samples=300, overlap_samples=0)
        assert view.shape == (14_397, 300)
        assert view[0, 0] == view[-1, -1] == 7.0
        assert np.shares_memory(view, data)
        assert not view.flags.writeable
