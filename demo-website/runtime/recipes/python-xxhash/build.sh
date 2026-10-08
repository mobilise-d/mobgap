#!/usr/bin/env bash
set -euxo pipefail
${PYTHON} -m pip install . ${PIP_ARGS} --no-build-isolation
