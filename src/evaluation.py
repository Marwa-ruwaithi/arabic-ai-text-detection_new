import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.utils import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, ensure_dirs


def read_parquet_dir(path):
    if os.path.isdir(path):
        files = sorted([os.path.join(path, f) for f in os.listdir(path)
                        if f.endswith(".parquet") and not f.startswith(".")
                        and not f.startswith("_")])
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Original", "Generated"],
                yticklabels=["Original", "Generated"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc(y_true, y_score, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="#4C72B0", lw=2,
             label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(out_path, top_n=15):
    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    if not os.path.exists(rf_path):
        print(f"Skipping feature importance: {rf_path} not found")
        return
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)

    importances = rf.feature_importances_
    feature_names = [
        "repeated_letter_words", "avg_words_per_paragraph",
        "top100_embedding_count", "burstiness", "roberta_probability",
    ]

    idx = np.argsort(importances)[::-1][:top_n]
    labels = []
    for i in idx:
        if i < len(feature_names):
            labels.append(feature_names[i])
        else:
            labels.append(f"tfidf_dim_{i - len(feature_names)}")
    vals = importances[idx]

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(labels)), vals, color="#4C72B0")
    plt.yticks(range(len(labels)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Top Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run():
    ensure_dirs()
    models = [
        ("Baseline", "preds_baseline.parquet"),
        ("Random Forest", "preds_random_forest.parquet"),
        ("GBT", "preds_gbt.parquet"),
    ]
    for name, fname in models:
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found")
            continue
        df = read_parquet_dir(path)
        y_true = df["label"].values
        y_pred = df["prediction"].values
        y_score = df["score"].values

        slug = name.replace(" ", "_").lower()
        plot_confusion_matrix(y_true, y_pred, f"Confusion Matrix - {name}",
                              os.path.join(FIGURES_DIR, f"cm_{slug}.png"))
        plot_roc(y_true, y_score, f"ROC Curve - {name}",
                 os.path.join(FIGURES_DIR, f"roc_{slug}.png"))
        print(f"Saved plots for {name}")

    summary_path = os.path.join(MODELS_DIR, "results_summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        plt.figure(figsize=(8, 5))
        m = df.melt(id_vars="model",
                    value_vars=["accuracy", "precision", "recall", "f1"],
                    var_name="metric", value_name="value")
        sns.barplot(data=m, x="model", y="value", hue="metric")
        plt.title("Model Performance Comparison")
        plt.ylim(0, 1)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=150)
        plt.close()
        print("Saved model comparison plot")

    plot_feature_importance(
        os.path.join(FIGURES_DIR, "fig_09_feature_importance.png")
    )
    print("Saved feature importance plot")
    print("Done")


if __name__ == "__main__":
    run()