#!/usr/bin/env bash
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "$0")" && pwd)
runtime_root=$(cd -- "$recipe_dir/../.." && pwd)
"${RATTLER_BUILD:-rattler-build}" build \
  --recipe "$recipe_dir" \
  --target-platform emscripten-wasm32 \
  --variant-config "${VARIANT_CONFIG:-$recipe_dir/variant.yaml}" \
  --channel https://repo.prefix.dev/emscripten-forge-4x \
  --channel conda-forge \
  --package-format tar-bz2 \
  --output-dir "${XXHASH_OUTPUT_DIR:-$runtime_root/build/output-xxhash}" \
  --keep-build --log-style plain
