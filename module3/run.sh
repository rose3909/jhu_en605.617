#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

total_threads="${1:-512}"
block_size="${2:-256}"

./assignment.exe "$total_threads" "$block_size"
python3 chart.py

# Preserve the evidence from each externally supplied configuration instead
# of allowing the next run to overwrite it.
cp results.csv "results_${total_threads}_${block_size}.csv"
cp performance_comparison.png \
   "performance_comparison_${total_threads}_${block_size}.png"

echo "Saved results_${total_threads}_${block_size}.csv"
echo "Saved performance_comparison_${total_threads}_${block_size}.png"
