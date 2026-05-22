

import os
import sys
import json
import time
import shutil

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.utils import (
    get_spark, PROCESSED_DIR, MODELS_DIR, ensure_dirs, normalize_arabic
)
from src.feature_engineering import (
    count_words_with_repeated_letters,
    avg_words_per_paragraph,
    count_top100_embedding_words,
    burstiness,
)



# Folder paths for the streaming demo


STREAM_INPUT_DIR = os.path.join(PROCESSED_DIR, "stream_input")
STREAM_OUTPUT_DIR = os.path.join(PROCESSED_DIR, "stream_output")
STREAM_CHECKPOINT_DIR = os.path.join(PROCESSED_DIR, "stream_checkpoint")



# Spark needs to know this in advance for the streaming reader

INPUT_SCHEMA = T.StructType([
    T.StructField("text", T.StringType(), True),
    T.StructField("label", T.IntegerType(), True),
    T.StructField("source", T.StringType(), True),
])


# Wrap our feature functions as Spark UDFs (same as in batch)

normalize_udf = F.udf(normalize_arabic, T.StringType())
repeated_udf = F.udf(count_words_with_repeated_letters, T.IntegerType())
avg_para_udf = F.udf(avg_words_per_paragraph, T.DoubleType())
top100_udf = F.udf(count_top100_embedding_words, T.IntegerType())
burst_udf = F.udf(burstiness, T.DoubleType())


# PRODUCER: drops JSON files into the input folder
# (simulates a Kafka producer for this demo)

def produce_stream(n_files=10, rows_per_file=50):
    """Split the balanced dataset into 10 small JSON-Lines files
    and write them to STREAM_INPUT_DIR with a small delay between
    each one. Spark Structured Streaming will pick them up."""
    import pandas as pd

    if os.path.exists(STREAM_INPUT_DIR):
        shutil.rmtree(STREAM_INPUT_DIR)
    os.makedirs(STREAM_INPUT_DIR, exist_ok=True)

    src = os.path.join(PROCESSED_DIR, "balanced_abstracts.parquet")

    # Load (handles both single-file and partitioned parquet)
    if os.path.isdir(src):
        files = sorted([os.path.join(src, f) for f in os.listdir(src)
                        if f.endswith(".parquet") and not f.startswith(("_", "."))])
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(src)

    # Shuffle and take a sample
    df = df.sample(n=min(n_files * rows_per_file, len(df)),
                   random_state=42).reset_index(drop=True)

    # Write 10 JSON-Lines files, one every 0.5 seconds
    for i in range(n_files):
        chunk = df.iloc[i * rows_per_file:(i + 1) * rows_per_file]
        if len(chunk) == 0:
            break
        out = os.path.join(STREAM_INPUT_DIR, f"batch_{i:03d}.json")
        records = chunk[["text", "label", "source"]].to_dict(orient="records")
        with open(out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {out}  ({len(chunk)} rows)")
        time.sleep(0.5)


# CONSUMER: runs the Spark Structured Streaming query

def run_streaming_query():
    spark = get_spark("ArabicAIGT-Streaming")

    # ---- Load the trained Random Forest pipeline ----
    # This is the model saved by modeling.py
    model_path = os.path.join(MODELS_DIR, "random_forest_spark")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}\n"
            "Run modeling.py first."
        )
    print(f"Loading trained pipeline -> {model_path}")
    model = PipelineModel.load(model_path)

    # ---- Clean output and checkpoint directories ----
    for d in [STREAM_OUTPUT_DIR, STREAM_CHECKPOINT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    os.makedirs(STREAM_INPUT_DIR, exist_ok=True)

 
    raw = (spark.readStream
                .schema(INPUT_SCHEMA)        # tell Spark the column types
                .option("multiLine", "false") # JSON Lines (1 record per line)
                .json(STREAM_INPUT_DIR))      # the folder to watch

    # ============================================================
    # FEATURE STAGE: compute the same 5 features used in training
    # Each .withColumn() applies a Spark UDF in parallel.
    # ============================================================
    featured = (raw
        .withColumn("text_normalized", normalize_udf(F.col("text")))
        .withColumn("repeated_letter_words", repeated_udf(F.col("text")))
        .withColumn("avg_words_per_paragraph", avg_para_udf(F.col("text")))
        .withColumn("top100_embedding_count", top100_udf(F.col("text_normalized")))
        .withColumn("burstiness", burst_udf(F.col("text")))
        .withColumn("roberta_probability", F.lit(0.0))  # disabled for speed
    )

    # SCORING STAGE: use the trained pipeline to predict
  
    scored = model.transform(featured)

    # Project to a clean output schema
    out = scored.select(
        "label",
        F.col("prediction").cast("int").alias("prediction"),
        vector_to_array(F.col("probability")).getItem(1).alias("score"),
        F.current_timestamp().alias("ingest_ts"),
    )

    # ============================================================
    # SINK: write predictions to Parquet
    # ============================================================
    # - format("parquet")   : output as Parquet files
    # - checkpointLocation  : Spark uses this to track what's been processed
    # - trigger("2 seconds"): process new data every 2 seconds
    # - outputMode("append"): only write NEW rows, not updates
    query = (out.writeStream
                .format("parquet")
                .option("path", STREAM_OUTPUT_DIR)
                .option("checkpointLocation", STREAM_CHECKPOINT_DIR)
                .outputMode("append")
                .trigger(processingTime="2 seconds")
                .start())

    print("Spark Structured Streaming query started.")
    print(f"  watching   : {STREAM_INPUT_DIR}")
    print(f"  writing to : {STREAM_OUTPUT_DIR}")
    print(f"  checkpoint : {STREAM_CHECKPOINT_DIR}")
    print("\nDrop JSON-Lines files into the watch directory to feed the query.")
    print("Running for 60 seconds...\n")


    deadline = time.time() + 60
    while time.time() < deadline:
        time.sleep(2)
        if query.recentProgress:
            for p in query.recentProgress[-3:]:
                ts = p.get("timestamp", "")
                num = p.get("numInputRows", 0)
                dur = p.get("durationMs", {}).get("triggerExecution", 0)
                print(f"  [{ts}] rows={num}  trigger_ms={dur}")
    query.stop()
    print("Streaming query stopped.")


def run():
    ensure_dirs()
    if len(sys.argv) > 1 and sys.argv[1] == "produce":
        produce_stream()
    else:
        run_streaming_query()


if __name__ == "__main__":
    run()
