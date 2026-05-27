# Selecting a Wear-Time Detection Algorithm

## Available Algorithms

mobgap provides three validated wear-time detection algorithms with different accuracy-speed trade-offs:

### WtdMegaritisCNN (Recommended)

Deep learning approach using CNNs (pure CNN or CNN-LSTM variant) trained on raw IMU data. Best overall accuracy and fastest **pipeline** inference.

- **Accuracy**: Excellent (daily error: ~1.3 minutes, **~12× more accurate** than signal processing**; CNN-LSTM variant marginally outperforms pure CNN**)
- **Speed**: ~20 seconds per day (full pipeline runtime; 3× faster than using signal processing weartime)
- **Dependencies**: TensorFlow **≥2.18**
- **Hardware**: Runs on CPU or GPU
- **Limitations**: Problematic on native Windows builds or Python ≥3.14 (TensorFlow constraints)

### WtdMegaritisXGBoost

Gradient boosting ML approach using 79-230 engineered features. **Intermediate accuracy between CNN and signal processing**, but computationally expensive due to feature extraction overhead.

- **Accuracy**: High (daily error: ~2.5 minutes, **~6× more accurate** than signal processing)
- **Speed**: ~5-17 minutes per day (feature-set dependent, slowest option)
- **Dependencies**: XGBoost
- **Hardware**: CPU-based

### WtdMegaritisSignal **(Pipeline Default)**

Signal processing approach using gyroscope spectral centroids and accelerometer variability with 2-out-of-3 voting. Detects low-frequency rotational patterns (<15-17 Hz) during natural movement.

- **Accuracy**: Good (daily error: ~15 minutes)
- **Speed**: ~1 minute per day
- **Dependencies**: None **(scipy/numpy/pandas only)**
- **Hardware**: Minimal requirements
- **Advantages**: Cross-platform compatibility, interpretable features, no ML dependencies

## Recommendation

**Use WtdMegaritisCNN** for best accuracy and speed in most applications where TensorFlow is available.

**Use WtdMegaritisSignal** when:

- Deployment environment may have TensorFlow compatibility issues (native Windows, Python ≥3.14)
- Minimizing dependencies is essential
- Interpretable feature-based detection is required
- Maintaining consistency with mobgap's signal processing philosophy

**Use WtdMegaritisXGBoost** when:

- **TensorFlow is unavailable but XGBoost is supported in your environment**
- **Better-than-signal-processing accuracy is needed without deep learning dependencies**

> **Note:** WtdMegaritisSignal is the pipeline default to ensure cross-platform compatibility. Users are encouraged to switch to WtdMegaritisCNN where TensorFlow is supported for superior performance.

> **Note:** Processing times are approximate benchmarks from development hardware (Apple M-series, 16GB RAM). Actual performance varies by system configuration and hardware acceleration availability.

> **Note:** All algorithms represent novel methodologies validated on ground-truth labeled datasets, using both accelerometer and gyroscope data to detect wear-like movement patterns. Full validation results will be published in an upcoming paper.