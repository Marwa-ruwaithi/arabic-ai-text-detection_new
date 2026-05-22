"""
Main pipeline entry point.

Runs the full Big Data pipeline phase by phase. Equivalent to run_all.sh
but callable from inside an IDE / notebook.

Usage:
    python main.py                # run every phase
    python main.py mapreduce      # run only the MapReduce phase
    python main.py prep features  # run a custom subset
"""

import sys
import time

from src import (
    data_acquisition,
    data_preparation,
    mapreduce_jobs,
    feature_engineering,
    modeling,
    evaluation,
    streaming_pipeline,
    scalability_benchmark,
)


PHASES = {
    "acquire":   ("Phase 1: Data Acquisition (Spark + Hugging Face)", data_acquisition.run),
    "prep":      ("Phase 2: Distributed Preprocessing (Spark UDFs)", data_preparation.run),
    "mapreduce": ("Phase 2b: MapReduce Corpus Aggregations (PySpark RDD)", mapreduce_jobs.main),
    "features":  ("Phase 3a: Feature Engineering (Spark + MLlib TF-IDF)", feature_engineering.run),
    "model":     ("Phase 3b: Distributed Modelling (Spark MLlib)", modeling.run),
    "eval":      ("Phase 4: Evaluation & Visualisation", evaluation.run),
    "stream":    ("Phase 5: Spark Structured Streaming", streaming_pipeline.run),
    "bench":     ("Phase 6: Scalability Benchmark (Spark)", scalability_benchmark.run),
}


def main():
    args = sys.argv[1:]
    if not args:
        # Note: streaming and bench are interactive / long-running so we
        # do not auto-run them as part of the default sweep.
        args = ["acquire", "prep", "mapreduce", "features", "model", "eval"]

    for phase in args:
        if phase not in PHASES:
            print(f"Unknown phase: {phase}")
            print(f"Available phases: {', '.join(PHASES)}")
            sys.exit(1)

    for phase in args:
        title, fn = PHASES[phase]
        print()
        print("=" * 72)
        print(f"  {title}")
        print("=" * 72)
        t0 = time.time()
        fn()
        print(f"\n[{phase}] finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
