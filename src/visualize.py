"""
visualize.py
------------
Generate every chart used in the README and the analysis notebook.

Two groups of figures are produced:

1. Exploratory Data Analysis (EDA) - built directly from the raw training data.
2. Model evaluation - built from models/eval_artifacts.pkl (produced by
   train_model.py), so run train_model.py first.

All figures are saved as PNGs in images/.

    python src/visualize.py
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_preprocessing import (
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    TARGET,
    clean,
    load_raw,
)

IMAGES_DIR = PROJECT_ROOT / "images"
MODELS_DIR = PROJECT_ROOT / "models"

sns.set_theme(style="whitegrid")
PALETTE = {0: "#4C9F70", 1: "#E4572E"}  # retained (green) vs churned (orange-red)
CHURN_LABELS = {0: "Retained", 1: "Churned"}


def _save(fig, name):
    path = IMAGES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# 1. EDA figures
# ---------------------------------------------------------------------------
def eda_figures(df: pd.DataFrame):
    df = df.copy()
    df["Churn_Label"] = df[TARGET].map(CHURN_LABELS)

    # 1a. Overall churn distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = df["Churn_Label"].value_counts()
    ax.bar(counts.index, counts.values,
           color=[PALETTE[0] if k == "Retained" else PALETTE[1] for k in counts.index])
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v/total:.1%})", ha="center", va="bottom", fontsize=10)
    ax.set_title("Customer Churn Distribution")
    ax.set_ylabel("Number of Customers")
    ax.margins(y=0.15)
    _save(fig, "01_churn_distribution.png")

    # 1b. Churn rate by categorical drivers
    cats = ["Contract Length", "Subscription Type", "Gender"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, cats):
        rate = df.groupby(col)[TARGET].mean().sort_values(ascending=False)
        ax.bar(rate.index, rate.values, color="#3D5A80")
        for i, v in enumerate(rate.values):
            ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"Churn Rate by {col}")
        ax.set_ylabel("Churn Rate")
        ax.set_ylim(0, 1)
    fig.suptitle("Churn Rate Across Categorical Drivers", fontsize=13)
    _save(fig, "02_churn_by_category.png")

    # 1c. Support calls vs churn (a strong behavioural signal)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rate = df.groupby("Support Calls")[TARGET].mean()
    ax.plot(rate.index, rate.values, marker="o", color=PALETTE[1])
    ax.set_title("Churn Rate by Number of Support Calls")
    ax.set_xlabel("Support Calls")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, 1)
    _save(fig, "03_churn_by_support_calls.png")

    # 1d. Numeric feature distributions split by churn
    key_num = ["Age", "Tenure", "Total Spend", "Payment Delay"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), key_num):
        for churn_val, sub in df.groupby(TARGET):
            ax.hist(sub[col], bins=30, alpha=0.6,
                    label=CHURN_LABELS[churn_val], color=PALETTE[churn_val])
        ax.set_title(f"{col} by Churn")
        ax.set_xlabel(col)
        ax.legend()
    fig.suptitle("Numeric Feature Distributions by Churn", fontsize=13)
    _save(fig, "04_numeric_distributions.png")

    # 1e. Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6.5))
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap (numeric features + churn)")
    _save(fig, "05_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# 2. Model evaluation figures
# ---------------------------------------------------------------------------
def eval_figures():
    artifacts_path = MODELS_DIR / "eval_artifacts.pkl"
    if not artifacts_path.exists():
        print("  [skip] models/eval_artifacts.pkl not found - run train_model.py first")
        return
    with open(artifacts_path, "rb") as fh:
        art = pickle.load(fh)

    # 2a. ROC curves for all models
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, d in art["roc_data"].items():
        ax.plot(d["fpr"], d["tpr"], label=f"{name} (AUC = {d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_title("ROC Curves - Model Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    _save(fig, "06_roc_curves.png")

    # 2b. Confusion matrix for the best model
    cm = art["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"], ax=ax)
    ax.set_title(f"Confusion Matrix - {art['best_name']}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    _save(fig, "07_confusion_matrix.png")

    # 2c. Feature importance for the best model
    imp = art["feature_importance"].sort_values("Importance")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(imp["Feature"], imp["Importance"], color="#3D5A80")
    ax.set_title(f"Feature Importance - {art['best_name']}")
    ax.set_xlabel("Importance")
    _save(fig, "08_feature_importance.png")

    # 2d. Metrics comparison bar chart
    m = art["metrics_df"].set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    m.plot(kind="bar", ax=ax, colormap="viridis", rot=0)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    _save(fig, "09_model_comparison.png")


def main():
    IMAGES_DIR.mkdir(exist_ok=True)
    print("Building EDA figures ...")
    train, _ = load_raw()
    train = clean(train)
    eda_figures(train)
    print("Building model-evaluation figures ...")
    eval_figures()
    print("Done.")


if __name__ == "__main__":
    main()
