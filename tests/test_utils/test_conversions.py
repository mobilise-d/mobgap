import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.spatial.transform import Rotation

from mobgap.consts import BF_SENSOR_COLS, BGF_SENSOR_COLS, GF_SENSOR_COLS, INITIAL_MOBILISED_ORIENTATION, SF_SENSOR_COLS
from mobgap.utils.conversions import to_body_frame, to_normal_frame, to_sensor_frame, transform_to_global_frame


class TestToAndFromBodyFrame:
    @pytest.mark.parametrize("in_cols, out_cols", [(SF_SENSOR_COLS, BF_SENSOR_COLS), (GF_SENSOR_COLS, BGF_SENSOR_COLS)])
    def test_correct_cols(self, in_cols, out_cols):
        df = pd.DataFrame(np.random.random((100, 6)), columns=in_cols)

        bf_df = to_body_frame(df)

        assert set(bf_df.columns) == set(out_cols)

        # Roundtrip
        df_round = to_normal_frame(bf_df)

        assert set(df_round.columns) == set(in_cols)

        assert_frame_equal(df, df_round)

    def test_sensor_frame_to_body_frame(self):
        # For the sensor frame, we literally just rename cols:
        df = pd.DataFrame(np.random.random((100, 6)), columns=SF_SENSOR_COLS)

        bf_df = to_body_frame(df)
        assert_frame_equal(df.T.reset_index(drop=True).T, bf_df.T.reset_index(drop=True).T)

    @pytest.mark.parametrize("func", (to_normal_frame, to_sensor_frame))
    def test_body_frame_to_sensor_frame(self, func):
        df = pd.DataFrame(np.random.random((100, 6)), columns=BF_SENSOR_COLS)

        bf_df = func(df)
        assert_frame_equal(df.T.reset_index(drop=True).T, bf_df.T.reset_index(drop=True).T)

    def test_global_frame_to_body_aligned_global_frame(self):
        df = pd.DataFrame(np.random.random((100, 6)), columns=GF_SENSOR_COLS)

        bf_df = to_body_frame(df)

        assert_frame_equal(
            df.filter(like="_gx").T.reset_index(drop=True).T, bf_df.filter(like="_gpa").T.reset_index(drop=True).T
        )
        assert_frame_equal(
            df.filter(like="_gz").T.reset_index(drop=True).T, bf_df.filter(like="_gis").T.reset_index(drop=True).T
        )
        assert_frame_equal(
            df.filter(like="_gy").T.reset_index(drop=True).T, -bf_df.filter(like="_gml").T.reset_index(drop=True).T
        )

    def test_body_aligned_global_frame_to_global_frame(self):
        df = pd.DataFrame(np.random.random((100, 6)), columns=BGF_SENSOR_COLS)

        bf_df = to_normal_frame(df)

        assert_frame_equal(
            bf_df.filter(like="_gx").T.reset_index(drop=True).T, df.filter(like="_gpa").T.reset_index(drop=True).T
        )
        assert_frame_equal(
            bf_df.filter(like="_gz").T.reset_index(drop=True).T, df.filter(like="_gis").T.reset_index(drop=True).T
        )
        assert_frame_equal(
            bf_df.filter(like="_gy").T.reset_index(drop=True).T, -df.filter(like="_gml").T.reset_index(drop=True).T
        )


class TestTransformToGlobalFrame:
    def test_transform_to_global_frame_sensor(self):
        # We just need to test the renaming here
        df = pd.DataFrame(np.random.random((100, 6)), columns=SF_SENSOR_COLS)

        dummy_rotations = Rotation.from_quat(np.repeat([[0, 0, 0, 1]], 100, axis=0))

        rotated = transform_to_global_frame(df, dummy_rotations)

        # As we used dummy rotations, there should be no rotation happening.
        assert_frame_equal(df.T.reset_index(drop=True).T, rotated.T.reset_index(drop=True).T)

        assert set(rotated.columns) == set(GF_SENSOR_COLS)

    def test_transform_to_global_frame_body(self):
        # We just need to test the renaming here
        df = pd.DataFrame(np.random.random((100, 6)), columns=BF_SENSOR_COLS)

        # To have the body frames align we need to use the initial orientation
        dummy_rotations = Rotation.from_quat(INITIAL_MOBILISED_ORIENTATION.as_quat()[None, :].repeat(100, axis=0))

        rotated = transform_to_global_frame(df, dummy_rotations)

        # As we used dummy rotations which is equal to the correct intitial, there should be no rotation happening.
        assert_frame_equal(df.T.reset_index(drop=True).T, rotated.T.reset_index(drop=True).T)

        assert set(rotated.columns) == set(BGF_SENSOR_COLS)
