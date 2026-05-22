"""
Scalability benchmark with Spark MLlib.

This is the Phase IV 'Scalability Test' deliverable: measure batch wall-clock
time for the preprocessing and training stages as a function of allocated
resources, here expressed as Spark shuffle partitions and parallelism.

We sweep across {1, 2, 4, 8} partition levels. At each level a fresh Spark
session is built with that many shuffle partitions, the feature stage and
the Random Forest training stage are timed, and a CSV + two PNG plots are
written for the report.
"""

import os
import time
import shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

from src.utils import (
    PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, ensure_dirs,
    normalize_arabic,
)
from src.feature_engineering import (
    count_words_with_repeated_letters, burstiness,
    avg_words_per_paragraph, count_top100_embedding_words,
)


STYLO_COLS = [
    "repeated_letter_words", "avg_words_per_paragraph",
    "top100_embedding_count", "burstiness", "roberta_probability",
]


def _spark_session(partitions):
    """Build a SparkSession with a specific shuffle-partition setting.
    Stopping and restarting the session per level is the cleanest way to
    measure the effect of parallelism in local mode."""
    builder = (SparkSession.builder
               .appName(f"ArabicAIGT-Bench-{partitions}")
               .master(f"local[{partitions}]")
               .config("spark.sql.shuffle.partitions", str(partitions))
               .config("spark.default.parallelism", str(partitions))
               .config("spark.driver.memory", "4g"))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def bench_preprocessing(partitions_list=(1, 2, 4, 8)):
    """Time the Spark UDF feature stage at each parallelism level."""
    results = []
    in_path = os.path.join(PROCESSED_DIR, "balanced_abstracts.parquet")

    for p in partitions_list:
        spark = _spark_session(p)
        df = spark.read.parquet(in_path)

        norm_udf = F.udf(normalize_arabic, T.StringType())
        rep_udf = F.udf(count_words_with_repeated_letters, T.IntegerType())
        avp_udf = F.udf(avg_words_per_paragraph, T.DoubleType())
        t100_udf = F.udf(count_top100_embedding_words, T.IntegerType())
        burst_udf = F.udf(burstiness, T.DoubleType())

        t0 = time.time()
        out = (df
            .withColumn("text_norm2", norm_udf(F.col("text")))
            .withColumn("rep", rep_udf(F.col("text")))
            .withColumn("avp", avp_udf(F.col("text")))
            .withColumn("t100", t100_udf(F.col("text_normalized")))
            .withColumn("burst", burst_udf(F.col("text"))))
        out.count()  # force the actions
        dt = time.time() - t0

        print(f"  parallelism={p}  preproc_time={dt:.2f}s")
        results.append({"parallelism": p, "preproc_time_sec": dt})
        spark.stop()
    return pd.DataFrame(results)


def bench_training(partitions_list=(1, 2, 4, 8)):
    """Time the Spark MLlib RandomForest training at each parallelism level."""
    results = []
    in_path = os.path.join(PROCESSED_DIR, "features.parquet")

    for p in partitions_list:
        spark = _spark_session(p)
        df = spark.read.parquet(in_path)

        assembler = VectorAssembler(inputCols=STYLO_COLS,
                                    outputCol="features",
                                    handleInvalid="keep")
        rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                                    numTrees=100, maxDepth=10, seed=42)
        pipeline = Pipeline(stages=[assembler, rf])

        t0 = time.time()
        model = pipeline.fit(df)
        dt = time.time() - t0

        print(f"  parallelism={p}  train_time={dt:.2f}s")
        results.append({"parallelism": p, "train_time_sec": dt})
        spark.stop()
    return pd.DataFrame(results)


def plot(df, ycol, title, out_path):
    plt.figure(figsize=(7, 4))
    plt.plot(df["parallelism"], df[ycol], marker="o", color="#4C72B0", lw=2)
    plt.xlabel("Parallelism level (Spark partitions)")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run():
    ensure_dirs()
    print("=== Preprocessing scalability benchmark (Spark) ===")
    prep_df = bench_preprocessing()
    prep_df.to_csv(os.path.join(MODELS_DIR, "scalability_preprocessing.csv"),
                   index=False)
    plot(prep_df, "preproc_time_sec",
         "Preprocessing Scalability (Spark)",
         os.path.join(FIGURES_DIR, "scalability_preprocessing_time.png"))

    print("\n=== Random Forest training scalability benchmark (Spark MLlib) ===")
    train_df = bench_training()
    train_df.to_csv(os.path.join(MODELS_DIR, "scalability_training.csv"),
                    index=False)
    plot(train_df, "train_time_sec",
         "Random Forest Training Scalability (Spark MLlib)",
         os.path.join(FIGURES_DIR, "scalability_training_time.png"))
    print("Done")


if __name__ == "__main__":
    run()
