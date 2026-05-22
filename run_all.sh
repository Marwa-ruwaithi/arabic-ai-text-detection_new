#!/usr/bin/env bash
# Linux runner for the Arabic AI-Generated-Text-Detection pipeline.
#
# Usage:
#     ./run_all.sh                     # run every batch phase
#     ./run_all.sh mapreduce           # run only one phase
#     ./run_all.sh acquire prep model  # run a subset
#
# Phases:
#     acquire    Phase 1  - Spark data acquisition from Hugging Face
#     prep       Phase 2  - Spark UDF preprocessing
#     mapreduce  Phase 2b - PySpark RDD MapReduce corpus aggregations
#     features   Phase 3a - Spark UDF + MLlib TF-IDF feature engineering
#     model      Phase 3b - Spark MLlib classification training
#     eval       Phase 4  - evaluation plots
#     stream     Phase 5  - Spark Structured Streaming (long-running)
#     bench      Phase 6  - scalability benchmark

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

PHASES=("$@")
if [ ${#PHASES[@]} -eq 0 ]; then
    PHASES=(acquire prep mapreduce features model eval)
fi

for p in "${PHASES[@]}"; do
    echo
    echo "============================================================"
    echo "  Running phase: $p"
    echo "============================================================"
    python3 main.py "$p"
done

echo
echo "All requested phases finished."
