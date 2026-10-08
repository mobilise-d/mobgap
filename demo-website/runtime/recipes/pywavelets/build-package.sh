#!/usr/bin/env bash
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
package_root="${MOBGAP_WASM_PACKAGES_ROOT:-$(cd -- "$recipe_dir/../.." && pwd)}"
if [[ -n "${RATTLER_BUILD:-}" ]]; then
  rattler_cmd="$RATTLER_BUILD"
elif [[ -x "$package_root/toolchain/env/bin/rattler-build" ]]; then
  rattler_cmd="$package_root/toolchain/env/bin/rattler-build"
else
  rattler_cmd=rattler-build
fi
"$rattler_cmd" build \
  --recipe "$recipe_dir" \
  --target-platform emscripten-wasm32 \
  --variant-config "${VARIANT_CONFIG:-$recipe_dir/variant.yaml}" \
  --channel https://repo.prefix.dev/emscripten-forge-4x \
  --channel conda-forge \
  --package-format tar-bz2 \
  --output-dir "${PYWAVELETS_OUTPUT_DIR:-$package_root/build/output-pywavelets}" \
  --keep-build
