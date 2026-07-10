"""
train_model.py
--------------
Train and compare three classifiers for customer-churn prediction, select the
best one by ROC-AUC, and persist everything the rest of the project needs:

    models/best_model.pkl        -> the winning scikit-learn Pipeline
    models/eval_artifacts.pkl    -> data used by visualize.py (ROC curves,
                                    confusion matrix, feature importances)
    reports/model_metrics.csv    -> side-by-side metrics table
    data/powerbi_churn_ready.csv -> scored test set for the Power BI dashboard

Run from anywhere:

    python src/train_model.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    FEATURES,
    PROJECT_ROOT,
    build_preprocessor,
    load_clean_split,
    load_scoring_set,
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"
RANDOM_STATE = 42


def get_models():
    """Return the candidate models, each wrapped in its own preprocessing
    pipeline so scaling/encoding is fit on the training fold only."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=16,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate(name, model, X_test, y_test):
    """Fit-agnostic scoring of an already-trained model on the test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return metrics, y_pred, y_proba, (fpr, tpr)


def feature_importance(best_name, best_model, X_test, y_test):
    """Return a tidy DataFrame of feature importances for the best model.

    Tree models expose native importances; for others we fall back to a
    permutation importance on a sample of the test set.
    """
    clf = best_model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        # Map transformed-feature importances back to the original columns by
        # grouping one-hot columns under their source categorical feature.
        prep = best_model.named_steps["prep"]
        names = prep.get_feature_names_out()
        importances = clf.feature_importances_
        agg = {}
        for feat, imp in zip(names, importances):
            base = feat.split("__", 1)[-1]
            for original in FEATURES:
                if base.startswith(original):
                    agg[original] = agg.get(original, 0.0) + imp
                    break
        imp_df = pd.DataFrame(
            {"Feature": list(agg.keys()), "Importance": list(agg.values())}
        )
    else:
        sample = X_test.sample(min(5000, len(X_test)), random_state=RANDOM_STATE)
        y_sample = y_test.loc[sample.index]
        result = permutation_importance(
            best_model, sample, y_sample, n_repeats=5, random_state=RANDOM_STATE
        )
        imp_df = pd.DataFrame(
            {"Feature": FEATURES, "Importance": result.importances_mean}
        )

    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    return imp_df


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Loading and cleaning data ...")
    # In-distribution evaluation: stratified split of the training file.
    X_train, X_test, y_train, y_test = load_clean_split()
    print(f"  train: {X_train.shape[0]:,} rows | validation: {X_test.shape[0]:,} rows")

    all_metrics = []
    roc_data = {}
    fitted = {}

    for name, model in get_models().items():
        print(f"\nTraining: {name} ...")
        model.fit(X_train, y_train)
        metrics, y_pred, y_proba, roc = evaluate(name, model, X_test, y_test)
        all_metrics.append(metrics)
        roc_data[name] = {"fpr": roc[0], "tpr": roc[1], "auc": metrics["ROC_AUC"]}
        fitted[name] = {"model": model, "y_pred": y_pred, "y_proba": y_proba}
        print(
            "  Accuracy={Accuracy:.3f}  Precision={Precision:.3f}  "
            "Recall={Recall:.3f}  F1={F1:.3f}  ROC-AUC={ROC_AUC:.3f}".format(**metrics)
        )

    metrics_df = pd.DataFrame(all_metrics).sort_values("ROC_AUC", ascending=False)
    metrics_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    print("\n=== Model comparison (sorted by ROC-AUC) ===")
    print(metrics_df.to_string(index=False))

    best_name = metrics_df.iloc[0]["Model"]
    best_model = fitted[best_name]["model"]
    best_pred = fitted[best_name]["y_pred"]
    best_proba = fitted[best_name]["y_proba"]
    print(f"\nBest model: {best_name}")

    # Persist the winning model.
    with open(MODELS_DIR / "best_model.pkl", "wb") as fh:
        pickle.dump(best_model, fh)

    # Feature importance + confusion matrix for the winner.
    imp_df = feature_importance(best_name, best_model, X_test, y_test)
    cm = confusion_matrix(y_test, best_pred)

    artifacts = {
        "best_name": best_name,
        "metrics_df": metrics_df,
        "roc_data": roc_data,
        "confusion_matrix": cm,
        "feature_importance": imp_df,
    }
    with open(MODELS_DIR / "eval_artifacts.pkl", "wb") as fh:
        pickle.dump(artifacts, fh)

    # Build the Power BI dataset from the held-out validation split. These are
    # unseen, in-distribution customers, so the predictions are accurate and the
    # dashboard tells a coherent churn story (unlike the provided testing file,
    # which is affected by data drift - see data_preprocessing.load_clean_split).
    print("\nExporting scored validation set for Power BI ...")
    scored = X_test.copy()
    scored["Actual_Churn"] = y_test.values
    scored["Predicted_Churn"] = best_pred
    scored["Churn_Probability"] = np.round(best_proba, 4)
    scored["Risk_Band"] = pd.cut(
        scored["Churn_Probability"],
        bins=[-0.001, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
    )
    scored.to_csv(DATA_DIR / "powerbi_churn_ready.csv", index=False)

    # For transparency, also report how the winning model behaves on the drifted
    # testing file (used only as a data-drift demonstration, not for the model
    # metrics above or the dashboard).
    X_drift, y_drift = load_scoring_set()
    drift_acc = accuracy_score(y_drift, best_model.predict(X_drift))
    print(
        f"  [note] Accuracy on the provided (drifted) testing file: {drift_acc:.3f} "
        "- expected to be low due to distribution shift."
    )

    print("\nArtifacts written:")
    print("  models/best_model.pkl")
    print("  models/eval_artifacts.pkl")
    print("  reports/model_metrics.csv")
    print("  data/powerbi_churn_ready.csv")


if __name__ == "__main__":
    main()
