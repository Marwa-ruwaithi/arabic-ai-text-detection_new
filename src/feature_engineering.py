"""
feature_engineering.py
======================
Phase 3a: Extract features from the cleaned Arabic abstracts.

================================================================
WHERE SPARK IS USED HERE:
  - Stylometric features computed using Spark UDFs (parallel).
  - TF-IDF built using Spark MLlib pipeline:
        Tokenizer  ->  HashingTF  ->  IDF
    This is the distributed alternative to scikit-learn's TfidfVectorizer.
  - The fitted TF-IDF model is saved so the streaming layer
    can re-use the SAME vocabulary at inference time.

WHAT FEATURES WE EXTRACT:
  Based on the index formula f(k*n + i) with i=16, n=21:
  
    f_16  = Number of words with repeated letters
    f_37  = Average words per paragraph
    f_58  = Top-100 Arabic word embedding count
    f_79  = Burstiness  (sentence-length variance)
    f_100 = RoBERTa output probability (substitute)
  
  Plus a TF-IDF representation with 5,000 features.
================================================================
"""

import os
import re
import math
import shutil
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_spark, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, ensure_dirs


# ================================================================
# Top-100 Arabic words list (function words + common academic stems)
# This is used by feature f_58.
# ================================================================

TOP_100_ARABIC_WORDS = {
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "ما", "لا", "لم", "لن", "قد", "كان", "كانت",
    "يكون", "تكون", "هو", "هي", "هم", "نحن", "أنا", "أنت", "كل", "بعض",
    "بين", "حيث", "حين", "عند", "أو", "ثم", "بل", "إذا", "إن", "أن",
    "كما", "لكن", "غير", "سوف", "أيضا", "كذلك", "حتى", "ولا", "ولم",
    "وما", "وقد", "ولكن", "فقد", "فإن", "فيها", "فيه", "بها", "به",
    "لها", "له", "منها", "منه", "عليها", "عليه", "إليها", "إليه",
    "هناك", "هنا", "الدراسة", "البحث", "النتائج", "المقال", "الورقة",
    "خلال", "أجل", "هدف", "تهدف", "يهدف", "العديد", "المختلفة",
    "أهمية", "أهم", "العربية", "الجزائر", "الجزائرية", "تحليل",
    "تركيز", "اضافة", "استكشاف", "تعزيز", "ضوء", "علاقة", "تحقيق",
    "وجود", "خاصة", "بشكل", "كبير", "مجال", "طريق", "ظل", "والتي",
    "وفي"
}


# ================================================================
# PURE PYTHON FEATURE FUNCTIONS
# Each function computes ONE feature for ONE text.
# Later we wrap them as Spark UDFs to run in parallel.
# ================================================================

def count_words_with_repeated_letters(text):
    """
    f_16: Count words where the SAME letter appears more than once.
    
    Example:
        "كتاب" -> 0 (each letter unique)
        "ممم"  -> 1 (the letter م repeats)
        "هاهاها" -> 1 (هـ and ا repeat)
    
    AI-generated text tends to have a different pattern here.
    """
    if not text:
        return 0
    count = 0
    for word in str(text).split():
        if len(set(word)) < len(word):
            count += 1
    return count


def avg_words_per_paragraph(text):
    """
    f_37: Average number of words per paragraph.
    
    A paragraph is detected by:
      - Blank lines (\n\n)
      - Or end-of-sentence dots
    
    Human writing tends to have longer, denser paragraphs.
    AI-generated text is more fragmented (around 21 words/para
    vs 76 for humans according to our analysis).
    """
    if not text:
        return 0.0
    paragraphs = [p for p in re.split(r"\n\s*\n|\.\s+", str(text)) if p.strip()]
    if not paragraphs:
        return 0.0
    return float(sum(len(p.split()) for p in paragraphs) / len(paragraphs))


def count_top100_embedding_words(text):
    """
    f_58: Count how many words in the text appear in our
    top-100 Arabic word list.
    
    These are the "filler" words that AI models tend to overuse.
    """
    if not text:
        return 0
    return sum(1 for w in str(text).split() if w in TOP_100_ARABIC_WORDS)


def burstiness(text):
    """
    f_79: Burstiness measures how varied sentence lengths are.
    
    Formula: (sigma - mu) / (sigma + mu)
      where sigma = std deviation of sentence lengths
            mu    = mean sentence length
    
    Range:
        Negative -> sentences are uniform length (typical of AI)
        Positive -> sentences vary a lot (typical of humans)
    """
    if not text:
        return 0.0
    sentences = [s.strip() for s in re.split(r"[\.!\?؟]+", str(text)) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mu = float(np.mean(lengths))
    sigma = float(np.std(lengths))
    if mu + sigma == 0:
        return 0.0
    return float((sigma - mu) / (sigma + mu))


def roberta_probability(text):
    """
    f_100: Mean per-token probability under XLM-RoBERTa.
    
    This is heavy (loads a 280MB neural model), so by default
    we skip it (return 0.0). To enable, set:
        export USE_ROBERTA=1
    """
    if not text or os.environ.get("USE_ROBERTA", "0") != "1":
        return 0.0
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import torch
        global _roberta_model, _roberta_tokenizer
        if "_roberta_model" not in globals():
            _roberta_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            _roberta_model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
            _roberta_model.eval()
        inputs = _roberta_tokenizer(str(text), return_tensors="pt",
                                    truncation=True, max_length=256)
        with torch.no_grad():
            outputs = _roberta_model(**inputs, labels=inputs["input_ids"])
        return float(math.exp(-outputs.loss.item()))
    except Exception:
        return 0.0


# ================================================================
# WRAP THE FEATURE FUNCTIONS AS SPARK UDFs
# This is what makes them run in parallel across Spark partitions.
# ================================================================

repeated_udf = F.udf(count_words_with_repeated_letters, T.IntegerType())
avg_para_udf = F.udf(avg_words_per_paragraph, T.DoubleType())
top100_udf = F.udf(count_top100_embedding_words, T.IntegerType())
burst_udf = F.udf(burstiness, T.DoubleType())
roberta_udf = F.udf(roberta_probability, T.DoubleType())


# ================================================================
# PLOTTING: violin plots of the 5 features by class
# ================================================================

def plot_stylometric_distributions(df_spark, out_path):
    """Make a violin plot showing each feature for human vs generated."""
    cols = ["repeated_letter_words", "avg_words_per_paragraph",
            "top100_embedding_count", "burstiness", "roberta_probability"]
    titles = ["f_16: Repeated-letter words", "f_37: Avg words / paragraph",
              "f_58: Top-100 embedding count", "f_79: Burstiness",
              "f_100: RoBERTa probability"]

    # Bring data to the driver (small, only 5 numeric columns + label)
    pdf = df_spark.select(["label"] + cols).toPandas()
    pdf["class"] = pdf["label"].map({0: "Original", 1: "Generated"})

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, (c, t) in enumerate(zip(cols, titles)):
        sns.violinplot(data=pdf, x="class", y=c, ax=axes[i],
                       hue="class",
                       palette={"Original": "#4C72B0", "Generated": "#C44E52"},
                       legend=False)
        axes[i].set_title(t)
        axes[i].set_xlabel("")
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ================================================================
# MAIN
# ================================================================

def run():
    ensure_dirs()
    spark = get_spark("ArabicAIGT-FeatureEngineering")

    # ---- Load the balanced data ----
    in_path = os.path.join(PROCESSED_DIR, "balanced_abstracts.parquet")
    df = spark.read.parquet(in_path)
    n = df.count()
    print(f"Loaded {n} rows from {in_path}")

    # ---- STEP 1: Add the 5 stylometric features as new columns ----
    # Each .withColumn() runs a Spark UDF on every row in parallel.
    print("\nComputing 5 stylometric features (Spark UDFs)...")
    df = (df
        .withColumn("repeated_letter_words", repeated_udf(F.col("text")))
        .withColumn("avg_words_per_paragraph", avg_para_udf(F.col("text")))
        .withColumn("top100_embedding_count", top100_udf(F.col("text_normalized")))
        .withColumn("burstiness", burst_udf(F.col("text")))
        .withColumn("roberta_probability", roberta_udf(F.col("text")))
    )

    # ---- Sanity check: print average feature values per class ----
    print("\nFeature means by class (should differ between human/generated):")
    df.groupBy("label").agg(
        F.avg("repeated_letter_words").alias("repeated"),
        F.avg("avg_words_per_paragraph").alias("avg_para"),
        F.avg("top100_embedding_count").alias("top100"),
        F.avg("burstiness").alias("burst"),
        F.avg("roberta_probability").alias("roberta"),
    ).show()

    # ---- STEP 2: Build TF-IDF using Spark MLlib ----
    # This is Spark's distributed alternative to sklearn's TfidfVectorizer.
    #
    # The pipeline has 3 stages:
    #   Tokenizer  : split text into words
    #   HashingTF  : convert words to integer hashes, count them
    #   IDF        : weight by Inverse Document Frequency
    #
    # Output: a 5000-dimensional sparse vector per document.
    print("\nBuilding Spark MLlib TF-IDF pipeline...")
    print("  Tokenizer  ->  HashingTF  ->  IDF")

    tokenizer = Tokenizer(inputCol="text_stemmed", outputCol="tokens")
    hashing_tf = HashingTF(inputCol="tokens", outputCol="tf", numFeatures=5000)
    idf = IDF(inputCol="tf", outputCol="tfidf")

    tfidf_pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
    tfidf_model = tfidf_pipeline.fit(df)

    # ---- Save the fitted TF-IDF pipeline ----
    # So the streaming layer can use EXACTLY the same vocabulary.
    tfidf_model_path = os.path.join(MODELS_DIR, "tfidf_pipeline_spark")
    if os.path.isdir(tfidf_model_path):
        shutil.rmtree(tfidf_model_path)
    tfidf_model.write().overwrite().save(tfidf_model_path)
    print(f"Saved TF-IDF pipeline -> {tfidf_model_path}")

    # ---- Save the features parquet ----
    keep_cols = [
        "label", "source", "text", "text_normalized", "text_stemmed",
        "repeated_letter_words", "avg_words_per_paragraph",
        "top100_embedding_count", "burstiness", "roberta_probability"
    ]
    out_path = os.path.join(PROCESSED_DIR, "features.parquet")
    df.select(keep_cols).write.mode("overwrite").parquet(out_path)
    print(f"Saved features -> {out_path}")

    # ---- Plot the feature distributions ----
    fig_path = os.path.join(FIGURES_DIR, "fig_05_stylometric_distributions.png")
    plot_stylometric_distributions(df, fig_path)
    print(f"Saved plot -> {fig_path}")

    spark.stop()
    print("Feature engineering done.")


if __name__ == "__main__":
    run()
