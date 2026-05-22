"""
modeling.py
===========
Phase 3b: Train 3 classifiers on the features using Spark MLlib.

================================================================
WHERE SPARK IS USED HERE:
  - Spark MLlib (Spark's machine learning library) is used for:
       * VectorAssembler  : combine our 5 features into ONE vector
       * LogisticRegression
       * RandomForestClassifier
       * GBTClassifier (Gradient Boosted Trees)
  - All training is DISTRIBUTED: Spark splits the training set
    across CPU cores (or cluster nodes) and trains in parallel.
  - We use Spark MLlib evaluators for metrics:
       * BinaryClassificationEvaluator  : ROC-AUC
       * MulticlassClassificationEvaluator : accuracy, precision, recall, F1

WHERE HADOOP / HDFS IS USED:
  - Trained models can be saved to HDFS using:
       model.write().save("hdfs:///user/marwah/models/random_forest_spark")
  - For this project they're saved locally.

WHY SPARK MLLIB INSTEAD OF SCIKIT-LEARN:
  Scikit-learn runs on ONE machine in memory.
  Spark MLlib partitions data across executors and trains in parallel.
  The project brief explicitly asks for Spark MLlib (5 points weight).
================================================================
"""

import os
import shutil
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import functions as F

from src.utils import get_spark, PROCESSED_DIR, MODELS_DIR, ensure_dirs


# ================================================================
# The 5 stylometric feature columns (computed in feature_engineering.py)
# ================================================================

STYLO_COLS = [
    "repeated_letter_words",
    "avg_words_per_paragraph",
    "top100_embedding_count",
    "burstiness",
    "roberta_probability",
]


# ================================================================
# HELPERS
# ================================================================

def stratified_split(df, seed=42):
    """
    Split into Train (70%) / Validation (15%) / Test (15%),
    keeping the class balance in each split.
    
    Spark's built-in randomSplit() does NOT stratify by class,
    so we split each class separately and union them back.
    """
    train_parts, val_parts, test_parts = [], [], []
    for lbl in [0, 1]:
        sub = df.filter(F.col("label") == lbl)
        tr, va, te = sub.randomSplit([0.70, 0.15, 0.15], seed=seed)
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)

    train = train_parts[0].unionByName(train_parts[1])
    val = val_parts[0].unionByName(val_parts[1])
    test = test_parts[0].unionByName(test_parts[1])
    return train, val, test


def evaluate_model(predictions_df, name):
    """
    Compute the 4 standard metrics + AUC using Spark MLlib evaluators.
    
    Spark MLlib has TWO evaluators we need:
      - BinaryClassificationEvaluator  : for AUC (uses the score, not the prediction)
      - MulticlassClassificationEvaluator : for accuracy / precision / recall / F1
    """
    # AUC needs the raw score (probability), not the 0/1 prediction
    auc_eval = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = auc_eval.evaluate(predictions_df)

    # These need the 0/1 prediction column
    acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="accuracy").evaluate(predictions_df)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="f1").evaluate(predictions_df)
    prec = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="weightedPrecision").evaluate(predictions_df)
    rec = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="weightedRecall").evaluate(predictions_df)

    print(f"  {name:<22} acc={acc:.4f}  prec={prec:.4f}  "
          f"rec={rec:.4f}  f1={f1:.4f}  auc={auc:.4f}")

    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc}


def save_predictions(predictions_df, name):
    """
    Save predictions to a parquet file so evaluation.py
    can make confusion matrices and ROC curves from them.
    """
    # The 'probability' column is a Spark Vector. We extract element [1]
    # which is P(label=1) using vector_to_array.
    out = predictions_df.select(
        "label",
        F.col("prediction").cast("int").alias("prediction"),
        vector_to_array(F.col("probability")).getItem(1).alias("score"),
    )
    path = os.path.join(PROCESSED_DIR, f"preds_{name}.parquet")
    if os.path.isdir(path):
        shutil.rmtree(path)
    out.write.mode("overwrite").parquet(path)


def save_spark_model(model, name):
    """Save a trained Spark MLlib pipeline so it can be loaded later."""
    path = os.path.join(MODELS_DIR, f"{name}_spark")
    if os.path.isdir(path):
        shutil.rmtree(path)
    model.write().overwrite().save(path)


# ================================================================
# MAIN
# ================================================================

def run():
    ensure_dirs()
    spark = get_spark("ArabicAIGT-Modeling")

    # ---- Load features (output of feature_engineering.py) ----
    print("Loading features parquet from disk...")
    df = spark.read.parquet(os.path.join(PROCESSED_DIR, "features.parquet"))
    n = df.count()
    print(f"  rows: {n}, columns: {len(df.columns)}")
    print("  class distribution:")
    df.groupBy("label").count().show()

    # ---- Stratified 70/15/15 split ----
    train, val, test = stratified_split(df, seed=42)
    print(f"Split sizes  ->  train={train.count()}  "
          f"val={val.count()}  test={test.count()}")

    # ---- Build the feature assembler ----
    # VectorAssembler takes our 5 separate columns and combines them
    # into ONE vector column called 'features' that the classifiers expect.
    assembler = VectorAssembler(
        inputCols=STYLO_COLS,
        outputCol="features",
        handleInvalid="keep",  # don't crash on nulls
    )

    # ---- The 3 classifiers we want to compare ----
    # All three follow the same Spark MLlib API:
    #   classifier.fit(train_df)  ->  PipelineModel
    #   model.transform(test_df)  ->  DataFrame with predictions
    classifiers = [
        ("logistic_regression",
         LogisticRegression(
             featuresCol="features", labelCol="label",
             maxIter=200, regParam=0.0)),

        ("random_forest",
         RandomForestClassifier(
             featuresCol="features", labelCol="label",
             numTrees=100, maxDepth=10, seed=42)),

        ("gbt",
         GBTClassifier(
             featuresCol="features", labelCol="label",
             maxIter=100, maxDepth=6, seed=42)),
    ]

    # ---- Train and evaluate each classifier ----
    print("\n=== Training Spark MLlib classifiers ===")
    results = []

    for name, clf in classifiers:
        print(f"\n--- {name} ---")

        # Pipeline = [VectorAssembler]  ->  [Classifier]
        # This packages preprocessing + model together.
        pipeline = Pipeline(stages=[assembler, clf])

        # Fit on training data (this is where Spark distributes the work)
        model = pipeline.fit(train)

        # Score the test set
        preds_test = model.transform(test)

        # Compute and save metrics
        result = evaluate_model(preds_test, name)
        results.append(result)

        # Save predictions for evaluation.py to use
        save_predictions(preds_test, name)
        # Save the trained Spark pipeline for streaming_pipeline.py
        save_spark_model(model, name)

        print(f"  saved model -> models/{name}_spark/")
        print(f"  saved preds -> data/processed/preds_{name}.parquet")

    # ---- Save summary CSV for the report ----
    summary = pd.DataFrame(results)
    summary_path = os.path.join(MODELS_DIR, "results_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary -> {summary_path}")
    print(summary.to_string(index=False))

    spark.stop()
    print("\nModelling phase complete.")


if __name__ == "__main__":
    run()
