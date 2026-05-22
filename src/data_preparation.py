

import os
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.utils import (
    get_spark, PROCESSED_DIR, ensure_dirs,
    normalize_arabic, remove_stopwords, isri_stem,
)


normalize_udf = F.udf(normalize_arabic, T.StringType())
remove_stop_udf = F.udf(remove_stopwords, T.StringType())
isri_stem_udf = F.udf(isri_stem, T.StringType())


def stratified_undersample(df, seed=42):
    """
    Balance the classes by under-sampling the bigger class.
    
    Our data has roughly:
        Class 0 (human)     :  2,992 abstracts
        Class 1 (generated) : 33,533 abstracts   <- 11x bigger!
    
    We sample down to make them equal.
    
    Spark's sampleBy() does this in a distributed way:
    it samples each class with a different fraction.
    """
    # Count how many rows we have per class
    counts = {row["label"]: row["count"]
              for row in df.groupBy("label").count().collect()}

    # Find the smaller class size
    minority = min(counts.values())

    # Compute the fraction to keep from each class
    fractions = {lbl: minority / cnt for lbl, cnt in counts.items()}

    # sampleBy = Spark's stratified sampler
    return df.sampleBy("label", fractions=fractions, seed=seed)


def run():
    """Main entry point."""
    ensure_dirs()

    # ---- Start Spark ----
    spark = get_spark("ArabicAIGT-Preparation")

    # ---- Load input data ----
    in_path = os.path.join(PROCESSED_DIR, "binary_abstracts.parquet")
    df = spark.read.parquet(in_path)
    n = df.count()
    print(f"Loaded {n} rows from {in_path}")

    print("Class distribution BEFORE preprocessing:")
    df.groupBy("label").count().show()

    # ---- Apply the 3 cleaning UDFs in a chain ----
    # Each .withColumn adds a new column based on the previous one.
    # Spark builds a "lazy" plan and only executes when we call .write.
    print("Applying Spark UDFs:")
    print("   text  ->  normalize_arabic  ->  remove_stopwords  ->  isri_stem")

    df = (df
        .withColumn("text_normalized", normalize_udf(F.col("text")))
        .withColumn("text_no_stop", remove_stop_udf(F.col("text_normalized")))
        .withColumn("text_stemmed", isri_stem_udf(F.col("text_no_stop")))
    )

    # ---- Save the FULL processed corpus ----
    # This is what mapreduce_jobs.py will read.
    out_full = os.path.join(PROCESSED_DIR, "processed_abstracts.parquet")
    df.write.mode("overwrite").parquet(out_full)
    print(f"Saved full processed data -> {out_full}")

    # ---- Balance the classes by under-sampling ----
    print("\nApplying stratified under-sampling...")
    balanced = stratified_undersample(df, seed=42)

    print("Class distribution AFTER under-sampling:")
    balanced.groupBy("label").count().show()

    out_balanced = os.path.join(PROCESSED_DIR, "balanced_abstracts.parquet")
    balanced.write.mode("overwrite").parquet(out_balanced)
    print(f"Saved balanced data -> {out_balanced}")

    spark.stop()
    print("Preparation phase complete.")


if __name__ == "__main__":
    run()
