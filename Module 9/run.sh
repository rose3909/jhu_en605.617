#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x bin/entity_match ]; then
  ./build.sh
fi
printf '\n=== Scenario 1: moderate noise ===\n'
./bin/entity_match 256 32 4 0.08 0.65 42
printf '\n=== Scenario 2: more records ===\n'
./bin/entity_match 512 64 4 0.10 0.45 42
printf '\n=== Scenario 3: difficult duplicates ===\n'
./bin/entity_match 1024 64 4 0.14 0.40 42
printf '\n=== Scenario 4: stricter threshold, same data ===\n'
./bin/entity_match 1024 64 4 0.14 0.45 42
printf '\n=== Scenario 5: higher-dimensional matching ===\n'
./bin/entity_match 2048 128 4 0.09 0.40 42
