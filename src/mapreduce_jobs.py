

import os
import sys
import time


# ----------------------------------------------------------------
# Configuration: where the input file is, and where to save output
# ----------------------------------------------------------------
# By default we read from the local file system.
# To use HDFS, set:  export MR_INPUT_PATH=hdfs:///path/to/file.parquet
# This way the same code works on both local and Hadoop clusters.

DEFAULT_INPUT = os.environ.get(
    "MR_INPUT_PATH",
    "data/processed/processed_abstracts.parquet",
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "MR_OUTPUT_DIR",
    "data/processed/mapreduce_output",
)

TOP_N_WORDS = 20
TOP_N_BIGRAMS = 20




def map_words(document):
    """
    MAP phase for Job 1 (WordCount).
    
    Input : one Arabic text (a string)
    Output: list of (word, 1) pairs - one pair for each word.
    
    Example:
        Input : "هذه الدراسة تهدف"
        Output: [("هذه", 1), ("الدراسة", 1), ("تهدف", 1)]
    """
    if document is None:
        return []
    return [(word, 1) for word in str(document).split() if word]


def map_bigrams(document):
    """
    MAP phase for Job 2 (BigramCount).
    
    A bigram = two words next to each other.
    
    Input : one Arabic text
    Output: list of ((word1, word2), 1) pairs.
    
    Example:
        Input : "هذه الدراسة تهدف"
        Output: [(("هذه", "الدراسة"), 1), (("الدراسة", "تهدف"), 1)]
    """
    if document is None:
        return []
    tokens = str(document).split()
    return [
        ((tokens[i], tokens[i + 1]), 1)
        for i in range(len(tokens) - 1)
    ]


def map_hapax(word_count_pair):
    """
    MAP phase for Job 3 - Stage B (Hapax Legomena Ratio).
    
    Hapax = a word that appears EXACTLY ONCE in the whole corpus.
    
    Input : (word, count)   <- output from Job 1 (WordCount)
    Output: list of (category, value) pairs.
            We re-key every word into either 'hapax' or 'total'
            so the next reduceByKey can sum them.
    
    Example:
        Input : ("الدراسة", 6)   -> not a hapax (count > 1)
        Output: [("total", 6)]
        
        Input : ("نادر", 1)      -> hapax!
        Output: [("total", 1), ("hapax", 1)]
    """
    word, count = word_count_pair
    pairs = [("total", count)]
    if count == 1:
        pairs.append(("hapax", 1))
    return pairs



# REDUCE FUNCTION



def reduce_sum(a, b):
    """
    REDUCE phase: take two counts of the same key and add them.
    
    Spark calls this many times to combine all values
    that share the same key (after shuffle).
    
    Example:
        For the key "هذه" with values [1, 1, 1, 1, 1, 1],
        Spark calls reduce_sum repeatedly:
            reduce_sum(1, 1) = 2
            reduce_sum(2, 1) = 3
            ...
            Final result: ("هذه", 6)
    """
    return a + b



# HELPER: detect if path is HDFS or local file system


def is_hdfs_path(path):
    """Return True if the path lives on HDFS (or other distributed FS)."""
    return path.startswith(("hdfs://", "s3://", "s3a://"))


def wipe_output_dir(path):
    """
    Spark refuses to write to a folder that already exists.
    For local paths, we delete it first. For HDFS, we use Hadoop's
    file system API later.
    """
    import shutil
    if not is_hdfs_path(path) and os.path.exists(path):
        shutil.rmtree(path)



# JOB 1: WORD COUNT


def run_word_count(docs_rdd, output_path):
    """
    The classic 'Hello World' of MapReduce.
    Count how many times each word appears in the corpus.
    """
    print("\n--- Job 1: WordCount  (Map -> Shuffle -> Reduce) ---")
    t0 = time.time()

    # ---- MAP phase ----
    # flatMap applies map_words to every document and flattens the result.
    # This is the MAP step of MapReduce.
    mapped = docs_rdd.flatMap(map_words)
    n_mapped = mapped.count()
    print(f"  [Map]      emitted (word, 1) pairs : {n_mapped:>12,}")

    # ---- SHUFFLE + REDUCE phases ----
    # reduceByKey does TWO things at once:
    #   1) Shuffle: groups all pairs with the same word together
    #   2) Reduce: applies reduce_sum to combine them
    word_counts = mapped.reduceByKey(reduce_sum)

    # We will use this RDD again in Job 3, so cache it in memory.
    word_counts.cache()

    n_keys = word_counts.count()
    print(f"  [Shuffle]  distinct words after group_by_key: {n_keys:>12,}")
    print(f"  [Reduce]   (word, total_count) pairs          : {n_keys:>12,}")

    # ---- Get statistics for the report ----
    total_tokens = word_counts.map(lambda kv: kv[1]).sum()
    vocab_size = n_keys
    ttr = vocab_size / total_tokens if total_tokens else 0.0
    print(f"  total tokens           : {total_tokens:>12,}")
    print(f"  vocabulary size        : {vocab_size:>12,}")
    print(f"  Type-Token Ratio (TTR) : {ttr:.4f}")

    # ---- Print top words ----
    top = word_counts.takeOrdered(TOP_N_WORDS, key=lambda kv: -kv[1])
    print(f"  top {TOP_N_WORDS} words:")
    for word, count in top:
        print(f"      {word:<25} {count}")

    # ---- Save output (this is where HDFS is used if path starts with hdfs://) ----
    wipe_output_dir(output_path)
    (word_counts
        .map(lambda kv: f"{kv[0]}\t{kv[1]}")
        .coalesce(1)
        .saveAsTextFile(output_path))
    print(f"  saved -> {output_path}")
    print(f"  job 1 wall-clock: {time.time() - t0:.2f}s")

    return word_counts, total_tokens, vocab_size


# JOB 2: BIGRAM COUNT


def run_bigram_count(docs_rdd, output_path):
    """
    Count how many times each two-word combination appears.
    Useful for finding common phrases in Arabic AI-generated text.
    """
    print("\n--- Job 2: BigramCount  (Map -> Shuffle -> Reduce) ---")
    t0 = time.time()

    # ---- MAP: emit ((w1, w2), 1) for every adjacent word pair ----
    mapped = docs_rdd.flatMap(map_bigrams)
    n_mapped = mapped.count()
    print(f"  [Map]      emitted (bigram, 1) pairs: {n_mapped:>12,}")

    # ---- SHUFFLE + REDUCE: same idea as Job 1 ----
    bigram_counts = mapped.reduceByKey(reduce_sum)
    n_keys = bigram_counts.count()
    print(f"  [Shuffle]  distinct bigrams after group_by_key: {n_keys:>12,}")
    print(f"  [Reduce]   (bigram, total_count) pairs        : {n_keys:>12,}")

    # ---- Print top bigrams ----
    top = bigram_counts.takeOrdered(TOP_N_BIGRAMS, key=lambda kv: -kv[1])
    print(f"  top {TOP_N_BIGRAMS} bigrams:")
    for (w1, w2), count in top:
        print(f"      {w1} {w2:<25} {count}")

    # ---- Save to disk (or HDFS) ----
    wipe_output_dir(output_path)
    (bigram_counts
        .map(lambda kv: f"{kv[0][0]} {kv[0][1]}\t{kv[1]}")
        .coalesce(1)
        .saveAsTextFile(output_path))
    print(f"  saved -> {output_path}")
    print(f"  job 2 wall-clock: {time.time() - t0:.2f}s")

    return bigram_counts



# JOB 3: HAPAX LEGOMENA RATIO (TWO-STAGE MAPREDUCE)


def run_hapax_ratio(word_counts_rdd, output_path):
    """
    Two-stage MapReduce, exactly as described in the project brief.

    STAGE A: WordCount  (already done in Job 1)
       Map    : (document) -> [(word, 1), ...]
       Reduce : sum -> (word, total_count)

    STAGE B: count hapax words and total tokens
       Map    : (word, count) -> [('total', count), ('hapax', 1) if count==1]
       Reduce : sum per category

    Final result:
       hapax_ratio = hapax_count / total_tokens
    """
    print("\n--- Job 3: Hapax Legomena Ratio  (two-stage MapReduce) ---")
    print("  Stage A: re-using WordCount output from Job 1")
    t0 = time.time()

    # ---- Stage B MAP: re-key each word into 'hapax' or 'total' ----
    rekeyed = word_counts_rdd.flatMap(map_hapax)
    n_rekeyed = rekeyed.count()
    print(f"  [Stage B - Map]     re-keyed pairs: {n_rekeyed:>10,}")

    # ---- Stage B SHUFFLE + REDUCE: sum per category ----
    aggregated = rekeyed.reduceByKey(reduce_sum)
    print(f"  [Stage B - Shuffle] distinct keys (expect 2): {aggregated.count()}")

    # Collect the small result (just 2 numbers) to the driver
    result = aggregated.collectAsMap()
    print(f"  [Stage B - Reduce]  aggregated: {result}")

    # ---- Compute final ratio ----
    hapax_n = result.get("hapax", 0)
    total_n = result.get("total", 0)
    ratio = (hapax_n / total_n) if total_n else 0.0

    print(f"\n  hapax words (count==1) : {hapax_n:>12,}")
    print(f"  total tokens           : {total_n:>12,}")
    print(f"  Hapax Legomena Ratio   : {ratio:.4f}")

    # ---- Save the final result ----
    wipe_output_dir(output_path)
    summary = word_counts_rdd.context.parallelize([
        f"hapax_count\t{hapax_n}",
        f"total_tokens\t{total_n}",
        f"hapax_ratio\t{ratio:.6f}",
    ], 1)
    summary.saveAsTextFile(output_path)
    print(f"  saved -> {output_path}")
    print(f"  job 3 wall-clock: {time.time() - t0:.2f}s")

    return hapax_n, total_n, ratio


# MAIN


def main():
    """Run all three MapReduce jobs, once per class (original / generated)."""
    # ---- This is where SPARK starts ----
    # SparkSession is the entry point for any Spark application.
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder
             .appName("ArabicAIGT-MapReduce")
             .config("spark.sql.shuffle.partitions", "4")
             .getOrCreate())

    # SparkContext (sc) is what we use to create RDDs.
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    print("=" * 72)
    print("PySpark MapReduce jobs - Arabic AI-Generated Text Detection")
    print(f"  Spark master   : {sc.master}")
    print(f"  Spark app id   : {sc.applicationId}")
    print(f"  Input path     : {DEFAULT_INPUT}")
    print(f"  Output dir     : {DEFAULT_OUTPUT_DIR}")
    print(f"  Using HDFS?    : {is_hdfs_path(DEFAULT_INPUT)}")
    print("=" * 72)

    # ---- Read the input parquet (from local FS or HDFS) ----
    df = spark.read.parquet(DEFAULT_INPUT)
    n_total = df.count()
    print(f"\nLoaded {n_total:,} processed abstracts")

    # Check we have the columns we need
    required = {"text_normalized", "label"}
    missing = required - set(df.columns)
    if missing:
        spark.stop()
        raise RuntimeError(f"Input is missing columns: {missing}")

    # ---- Run all 3 jobs per class (original vs generated) ----
    # This gives us class-conditional statistics for the report.
    for label_value, label_name in [(0, "original"), (1, "generated")]:
        print("\n" + "#" * 72)
        print(f"#  Class: {label_name.upper()}  (label={label_value})")
        print("#" * 72)

        # Build an RDD of just the text column for this class
        docs_rdd = (df.filter(df["label"] == label_value)
                      .select("text_normalized")
                      .rdd
                      .map(lambda row: row["text_normalized"])
                      .filter(lambda t: t is not None and len(t) > 0))

        n_docs = docs_rdd.count()
        print(f"\nDocuments in this class: {n_docs:,}")

        # Output paths for this class
        wc_out = os.path.join(DEFAULT_OUTPUT_DIR, f"wordcount_{label_name}")
        bg_out = os.path.join(DEFAULT_OUTPUT_DIR, f"bigrams_{label_name}")
        hp_out = os.path.join(DEFAULT_OUTPUT_DIR, f"hapax_{label_name}")

        # Run the three jobs in sequence.
        # Notice that Job 3 RE-USES the WordCount RDD from Job 1
        # (this is why we cached it).
        word_counts, total_tokens, vocab_size = run_word_count(docs_rdd, wc_out)
        run_bigram_count(docs_rdd, bg_out)
        run_hapax_ratio(word_counts, hp_out)

        # Free up memory before next iteration
        word_counts.unpersist()

    print("\n" + "=" * 72)
    print("All MapReduce jobs finished successfully.")
    print("=" * 72)

    spark.stop()


if __name__ == "__main__":
    main()
