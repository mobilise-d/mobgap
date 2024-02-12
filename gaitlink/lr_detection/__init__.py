"""Algorithms for left/right foot detection for a single IMU sensor placed on the lower back."""

from gaitlink.lr_detection._lr_detect_ML import UllrichLRDetection
from gaitlink.lr_detection._lr_detect_McCamley import McCamleyLRDetection

__all__ = ["UlrichLRDetection", "McCamleyLRDetection"]

# TODO: we will need to come back to this once all the classes are implemented.
# This list is defined to specify which symbols (classes, functions or variables) should be exported when someone imports this module.