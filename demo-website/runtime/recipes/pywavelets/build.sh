#!/usr/bin/env bash
set -euxo pipefail
# Meson uses target NumPy's C headers and package config, never a native ABI.
export PKG_CONFIG_PATH="$PREFIX/lib/python$PY_VER/site-packages/numpy/_core/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
sed "s|@PYTHON@|${PYTHON}|g" "$RECIPE_DIR/emscripten.meson.cross" > "$SRC_DIR/emscripten.meson.cross"
"${PYTHON}" -m pip install . ${PIP_ARGS} --no-build-isolation \
  -Csetup-args="--cross-file=$SRC_DIR/emscripten.meson.cross" \
  -Cbuild-dir="_build" \
  -Ccompile-args="--verbose"
