"""Wear-time detection algorithms for identifying when a wearable device is being worn.

Algorithm Selection
-------------------
Three wear-time detection algorithms are available:

**WtdMegaritisCNN** (Recommended)
    Deep learning approach using convolutional neural networks (pure CNN or CNN-LSTM variant)
    trained on raw IMU data. Best overall accuracy with reasonable computational cost. Requires
    TensorFlow/Keras. Processes ~2 minutes per day of data (runs on CPU or GPU).

**WtdMegaritisXGBoost**
    Gradient boosting machine learning approach using 230 or 79 features.
    Good accuracy but computationally expensive due to extensive
    feature extraction. Requires XGBoost. Processes ~5-17 minutes per day of data
    depending on feature set.

**WtdMegaritisSignal** (Default in pipelines)
    Signal processing approach using gyroscope spectral centroids and accelerometer
    variability with 2-out-of-3 voting logic. Detects characteristic low-frequency
    rotational patterns (<15-17 Hz) during natural body movement. Lightweight with
    no ML dependencies. Fastest execution (~1 minute per day of data) but lower
    accuracy than DL methods.

**Recommendation**: WtdMegaritisCNN provides the best accuracy-speed trade-off for most
use cases. Use WtdMegaritisSignal when minimising dependencies is essential or when
interpretable feature-based detection is required.

.. note:: Processing times are approximate benchmarks from a local development machine.
"""

from mobgap.weartime._wtd_megaritis_signal import WtdMegaritisSignal
try:
    from mobgap.weartime._wtd_megaritis_cnn import WtdMegaritisCNN
except ModuleNotFoundError:
    WtdMegaritisCNN = None

try:
    from mobgap.weartime._wtd_megaritis_xgboost import WtdMegaritisXGBoost
except ModuleNotFoundError:
    WtdMegaritisXGBoost = None

__all__ = ["WtdMegaritisCNN", "WtdMegaritisSignal", "WtdMegaritisXGBoost"]
