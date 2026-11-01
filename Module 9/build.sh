#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p bin
nvcc -O3 -std=c++17 -lineinfo -Xcompiler=-Wall \
  -o bin/entity_match src/main.cu -lcublas
printf 'Built %s\n' 'bin/entity_match'
