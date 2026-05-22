"""
utils.py
========
Helper functions used by all the other modules in the project.

================================================================
WHAT IS HERE:
  1) Paths    - where to read/write data, models, figures.
  2) Spark    - factory function to create a SparkSession.
  3) Hadoop   - helper to detect HDFS paths.
  4) Arabic   - normalization, stop-word removal, ISRI stemming.
================================================================
"""

import os
import re
import sys
from pyspark.sql import SparkSession


# Make sure Spark workers use the same Python interpreter as the driver.
# This matters when running inside a virtualenv or on a cluster.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


# ================================================================
# 1) PROJECT PATHS
# ================================================================
# All paths are relative to the project root folder.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")


def ensure_dirs():
    """Create the output folders if they don't exist yet."""
    for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


# ================================================================
# 2) SPARK SESSION FACTORY
# ================================================================
# This is where Spark gets configured and started.

def get_spark(app_name="ArabicAIGT", shuffle_partitions=4):
    """
    Create a SparkSession.
    
    A SparkSession is the main entry point for using Spark.
    It manages the Spark application, the SparkContext, and lets us
    read/write DataFrames and run SQL.
    
    The 'master' setting tells Spark WHERE to run:
        - "local[*]"  : run on this single machine using all CPU cores
        - "yarn"      : run on a Hadoop YARN cluster
        - "spark://..."  : run on a standalone Spark cluster
    
    For our project we use 'local[*]' in development and 'yarn' when
    we submit to Hadoop with spark-submit.
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        # How many partitions to use during shuffle operations.
        # Smaller value = faster on small data, larger = better for big data.
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ================================================================
# 3) HADOOP / HDFS HELPER
# ================================================================

def is_hdfs_path(path):
    """
    Check if a path lives on HDFS (Hadoop Distributed File System).
    
    HDFS paths start with 'hdfs://'.
    Local paths just look like '/home/marwah/data/...' or 'data/...'.
    
    We use this to write code that works on both local files AND HDFS.
    """
    return path.startswith(("hdfs://", "s3://", "s3a://"))


# ================================================================
# 4) ARABIC TEXT PROCESSING
# ================================================================
# These functions clean Arabic text before feeding it to Spark/ML.

# Arabic-specific regex patterns:
#   - diacritics : the little marks above/below Arabic letters
#   - non_arabic : anything that is not Arabic letters or whitespace
#   - elongation : repeated characters like "كتاااب" -> "كتاب"

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")
NON_ARABIC = re.compile(r"[^\u0600-\u06FF\s]")
EXTRA_SPACES = re.compile(r"\s+")
ELONGATION = re.compile(r"(.)\1{2,}")


def normalize_arabic(text):
    """
    Normalize Arabic text:
      - Remove diacritics (تشكيل)
      - Unify alef variants:   إ أ آ  ->  ا
      - Unify yeh variants:    ى  ->  ي
      - Unify hamza forms:     ؤ ئ
      - Convert teh-marbuta:   ة  ->  ه
      - Remove non-Arabic characters and extra spaces
      - Collapse repeated letters (elongation)
    
    Example:
        Input : "هَذِهِ الدِّرَاسَةُ تَهْدِفُ إلى التَّحْليل"
        Output: "هذه الدراسه تهدف الي التحليل"
    """
    if text is None:
        return ""
    text = str(text)
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = NON_ARABIC.sub(" ", text)
    text = ELONGATION.sub(r"\1", text)
    text = EXTRA_SPACES.sub(" ", text).strip()
    return text


# Common Arabic stop-words (function words that don't carry meaning).
# Removing them helps the model focus on content words.

ARABIC_STOPWORDS = {
    "من", "في", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "اللاتي", "ما", "لا", "لم", "لن", "قد",
    "كان", "كانت", "يكون", "تكون", "هو", "هي", "هم", "هن", "نحن",
    "أنا", "أنت", "أنتم", "كل", "بعض", "بين", "حيث", "حين", "عند",
    "أو", "ثم", "بل", "إذا", "إن", "أن", "كما", "لكن", "غير", "سوف",
    "ايضا", "كذلك", "حتى", "اذا", "ولا", "ولم", "وما", "وقد",
    "ولكن", "فقد", "فإن", "فيها", "فيه", "بها", "به", "لها", "له",
    "منها", "منه", "عليها", "عليه", "اليها", "اليه", "هناك", "هنا",
}


def remove_stopwords(text):
    """Remove common Arabic stop-words from the text."""
    if text is None:
        return ""
    return " ".join(t for t in str(text).split() if t not in ARABIC_STOPWORDS)


def isri_stem(text):
    """
    Apply the ISRI stemmer to reduce Arabic words to their roots.
    
    ISRI = Information Science Research Institute (NLTK package).
    
    Example:
        Input : "الكاتب الكتاب الكتابة"
        Output: "كتب كتب كتب"   (all share the root ك-ت-ب)
    
    Stemming helps because it groups different word forms together,
    which improves TF-IDF and classification accuracy.
    """
    try:
        from nltk.stem.isri import ISRIStemmer
    except Exception:
        # If NLTK isn't installed, just return the text unchanged
        return text
    if text is None:
        return ""
    stemmer = ISRIStemmer()
    return " ".join(stemmer.stem(t) for t in str(text).split())


def preprocess_full(text):
    """Apply the full preprocessing pipeline in one call."""
    return remove_stopwords(normalize_arabic(text))
