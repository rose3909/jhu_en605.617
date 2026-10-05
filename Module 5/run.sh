#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p evidence
if [[ ! -x assignment.exe ]]; then bash build.sh; fi
if (( $# > 0 )); then
    ./assignment.exe "$@" | tee "evidence/run_${1}_${2:-256}.txt"
    exit 0
fi
nvidia-smi | tee evidence/gpu.txt
for spec in '65536 64' '65536 256' '1048576 128' '1048576 256' '1048576 512' '100003 256' '17 64'; do
    read -r elements threads <<< "$spec"
    echo "Running ./assignment.exe $elements $threads 100"
    ./assignment.exe "$elements" "$threads" 100 "evidence/results_${elements}_${threads}.csv" \
        | tee "evidence/run_${elements}_${threads}.txt"
done
