#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
nvcc -O2 -std=c++14 -o streams_events streams_events.cu
printf 'Built %s/streams_events\n' "$PWD"
