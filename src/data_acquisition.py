import os
from datasets import load_dataset
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from src.utils import get_spark, RAW_DIR, PROCESSED_DIR, ensure_dirs

DATASET_NAME = "KFUPM-JRCAI/arabic-generated-abstracts"
SPLITS = ["by_polishing", "from_title", "from_title_and_content"]
MODEL_COLS = [
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract",
]


def download_dataset():
    ensure_dirs()
    frames = []
    for split in SPLITS:
        ds = load_dataset(DATASET_NAME, split=split)
        df = ds.to_pandas()
        df["source_split"] = split
        frames.append(df)
    import pandas as pd
    full = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(RAW_DIR, "raw_abstracts.parquet")
    full.to_parquet(out_path, index=False)
    return out_path


def reshape_to_binary(spark, raw_path):
    pdf = spark.read.parquet(raw_path)

    originals = pdf.select(
        F.col("original_abstract").alias("text"),
        F.lit("human").alias("source"),
        F.lit(0).alias("label"),
        F.col("source_split"),
    ).filter(F.col("text").isNotNull())

    generated_frames = []
    for col in MODEL_COLS:
        if col in pdf.columns:
            model_name = col.replace("_generated_abstract", "")
            generated_frames.append(
                pdf.select(
                    F.col(col).alias("text"),
                    F.lit(model_name).alias("source"),
                    F.lit(1).alias("label"),
                    F.col("source_split"),
                ).filter(F.col("text").isNotNull())
            )

    binary = originals
    for g in generated_frames:
        binary = binary.unionByName(g)

    binary = binary.dropDuplicates(["text"])
    return binary


def quality_checks(df):
    total = df.count()
    nulls = df.filter(F.col("text").isNull() | (F.length(F.col("text")) == 0)).count()
    label_counts = df.groupBy("label").count().collect()
    distribution = {row["label"]: row["count"] for row in label_counts}
    return {"total": total, "empty_or_null": nulls, "label_distribution": distribution}


def run():
    raw_path = download_dataset()
    spark = get_spark("DataAcquisition")
    binary = reshape_to_binary(spark, raw_path)
    binary.write.mode("overwrite").parquet(os.path.join(PROCESSED_DIR, "binary_abstracts.parquet"))
    stats = quality_checks(binary)
    print("Acquisition complete")
    print(f"Total rows: {stats['total']}")
    print(f"Empty or null: {stats['empty_or_null']}")
    print(f"Label distribution: {stats['label_distribution']}")
    spark.stop()


if __name__ == "__main__":
    run()
