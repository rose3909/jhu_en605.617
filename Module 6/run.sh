#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x ./streams_events ]]; then
  bash ./build.sh
fi
./streams_events
