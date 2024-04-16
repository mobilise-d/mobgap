"""Algorithms to detect the laterality (left/right) of initial contacts within raw IMU data."""

from mobgap.lrc._lrc_mccamley import LrcMcCamley
from mobgap.lrc._lrc_ullrich import LrcUllrich

__all__ = ["LrcMcCamley", "LrcUllrich"]
