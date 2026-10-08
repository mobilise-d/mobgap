"""Shared native/browser probes for the real PyWavelets1.9 package."""

import inspect
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pywt


def collect():
    t = np.arange(96)
    signal = (((t * 7) % 23) - 11) / 11.0
    matrix = np.stack([signal, signal[::-1] * 0.75 + 0.1])
    cases = {}
    for wavelet in ("gaus2", "morl", "cmor1.5-1.0"):
        for precision in (10, 12):
            for method in ("conv", "fft"):
                name = f"{wavelet}-precision{precision}-{method}"
                coefficients, frequencies = pywt.cwt(
                    matrix, [1, 2.5, 7, 15], wavelet, sampling_period=0.01, method=method, axis=1, precision=precision
                )
                cases[name] = {
                    "real": coefficients.real.tolist(),
                    "imag": coefficients.imag.tolist(),
                    "frequencies": frequencies.tolist(),
                }
    reconstructed = pywt.idwt(*pywt.dwt(signal, "db2"), "db2")
    np.testing.assert_allclose(reconstructed, signal, atol=1e-12, rtol=1e-12)
    default, _ = pywt.cwt(signal, [2.5, 7], "gaus2")
    explicit12, _ = pywt.cwt(signal, [2.5, 7], "gaus2", precision=12)
    np.testing.assert_array_equal(default, explicit12)
    return {
        "version": importlib.metadata.version("PyWavelets"),
        "module_version": pywt.__version__,
        "numpy": np.__version__,
        "cwt_signature": str(inspect.signature(pywt.cwt)),
        "extension": pywt._extensions._pywt.__file__,
        "cases": cases,
        "dwt_roundtrip_max_abs_error": float(np.max(np.abs(reconstructed - signal))),
    }


def compare(reference):
    actual = collect()
    assert actual["version"] == "1.9.0", actual["version"]
    assert "precision=12" in actual["cwt_signature"], actual["cwt_signature"]
    metrics = {}
    for name, expected in reference["cases"].items():
        error = {}
        for field in ("real", "imag", "frequencies"):
            a = np.asarray(actual["cases"][name][field])
            b = np.asarray(expected[field])
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12, err_msg=f"{name} {field}")
            error[field] = float(np.max(np.abs(a - b)))
        metrics[name] = error
    difference = float(
        np.max(
            np.abs(
                np.asarray(actual["cases"]["gaus2-precision10-conv"]["real"])
                - np.asarray(actual["cases"]["gaus2-precision12-conv"]["real"])
            )
        )
    )
    assert difference > 1e-6, "precision10 and12 must select different wavelet discretization"
    return {k: v for k, v in actual.items() if k != "cases"} | {
        "checks": len(metrics),
        "native_comparison_max_abs_errors": metrics,
        "precision10_vs12_max_difference": difference,
        "passed": True,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--reference":
        Path(sys.argv[2]).write_text(json.dumps(collect(), indent=2))
    else:
        print(json.dumps(compare(json.loads(Path(sys.argv[1]).read_text())), indent=2))
