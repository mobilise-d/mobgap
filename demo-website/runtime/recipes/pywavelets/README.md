# PyWavelets1.9.0 for the genuine WASM runtime

This recipe builds the unmodified official PyPI1.9.0 source with Meson/Cython and Emscripten4.0.9, targeting CPython3.13.1 and NumPy2.4.6. It produces four genuine WASM extension modules. There are no precision or Numba adapters.

Run `./build-package.sh` with rattler-build0.67.0 on PATH. The script locates `toolchain/env/bin/rattler-build` relative to the package tree, or uses `rattler-build` from PATH. `RATTLER_BUILD`, `VARIANT_CONFIG` and `PYWAVELETS_OUTPUT_DIR` override these locations. `variant.yaml` pins the target. The script downloads upstream dependencies and uses target NumPy headers through pkg-config. `rebuild.sh` forwards to this portable script.

Source SHA256 is `148d12203377772bea452a59211d98649c8ee4a05eff019a9021853a36babdc8`. Artifact `pywavelets-1.9.0-np24py313h588d514_0.tar.bz2` has SHA256 `80335c15cceebcbe5528fbc839194e36f36493dcc3ec8889aa218456bd7a34d4`. `validate_artifact.py` verifies target metadata and all four WASM headers.

Generate the comparison data with native PyWavelets 1.9.0 and NumPy 2.4.6:

```sh
python parity_probe.py --reference native-reference.json
```

For browser verification, load `parity_probe.py` and the generated `native-reference.json` into the fresh kernel, then execute:

```python
import json
from pathlib import Path
from parity_probe import compare

print(json.dumps(compare(json.loads(Path("native-reference.json").read_text())), indent=2))
```

The probe checks12 full-array CWT cases against native PyWavelets1.9.0/NumPy2.4.6 at absolute/relative tolerance1e-12. Cases cover gaus2/morl/complex Morlet, precision10/12, convolution/FFT and a two-channel axis1 input. It also verifies DWT reconstruction and default precision12. The dedicated native CPython3.13.12 reference passed.

Upstream1.9.0's wheel and source retain `pywt.__version__ == '1.8.0'` while distribution metadata says1.9.0. The new `cwt` signature supports `precision=12`. This build preserves upstream's version-string bug. The probe checks distribution metadata and reports the module version separately.
