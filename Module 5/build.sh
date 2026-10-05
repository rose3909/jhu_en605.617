#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p evidence
command -v nvcc >/dev/null || { echo 'nvcc missing. Select a Colab GPU runtime.' >&2; exit 1; }
nvcc --version | tee evidence/compiler.txt
make -B 2>&1 | tee evidence/build.log
