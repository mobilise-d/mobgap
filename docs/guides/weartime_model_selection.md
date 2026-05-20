# Selecting a Wear-Time Detection Algorithm

## Available Algorithms

mobgap provides three validated wear-time detection algorithms with different accuracy-speed trade-offs:

### WtdMegaritisCNN (Recommended)

Deep learning approach using CNNs (pure CNN or CNN-LSTM variant) trained on raw IMU data. Best overall accuracy with reasonable computational cost.

- **Accuracy**: Excellent (daily error: ~ 1.3 minutes)
- **Speed**: ~2 minutes per day of data
- **Dependencies**: TensorFlow/Keras
- **Hardware**: Runs on CPU or GPU

### WtdMegaritisXGBoost

Gradient boosting ML approach using 79-230 engineered features. Good accuracy but computationally expensive.

- **Accuracy**: High (daily error: ~ 2.5 minutes)
- **Speed**: ~5-17 minutes per day (feature-set dependent)
- **Dependencies**: XGBoost
- **Hardware**: CPU-based

### WtdMegaritisSignal (Pipeline Default)

Signal processing approach using gyroscope spectral centroids and accelerometer variability with 2-out-of-3 voting. Detects low-frequency rotational patterns (<15-17 Hz) during natural movement.

- **Accuracy**: Good (daily error: ~ 15 minutes)
- **Speed**: ~1 minute per day (fastest)
- **Dependencies**: None (numpy/pandas only)
- **Hardware**: Minimal requirements

## Recommendation

**Use WtdMegaritisCNN** for best accuracy-speed trade-off in most applications.

**Use WtdMegaritisSignal** when:

- Minimizing dependencies is essential
- Interpretable feature-based detection is required
- Maintaining consistency with mobgap's signal processing philosophy

> **Note:** WtdMegaritisSignal is the pipeline default to align with mobgap's primarily signal processing-based approach, though WtdMegaritisCNN achieves superior performance.

> **Note:** Processing times are approximate benchmarks from local hardware. Actual performance varies by system configuration.
