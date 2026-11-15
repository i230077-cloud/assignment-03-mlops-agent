"""
ML Training Pipeline Script with MLflow Tracking & DVC Provenance Recording
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import sys
import json
import time
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

def get_git_commit_and_tag():
    """Retrieve current git commit hash and nearest tag."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown_commit"
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--always"], text=True
        ).strip()
    except Exception:
        tag = "v2.0"
    return commit, tag

def get_dvc_hash(dvc_file_path="data/raw/customer_churn.csv.dvc"):
    """Extract md5 hash from dvc file."""
    if os.path.exists(dvc_file_path):
        try:
            with open(dvc_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if "md5:" in line:
                        return line.split("md5:")[1].strip()
        except Exception:
            pass
    return "dvc_md5_placeholder"

def train(
    data_path="data/raw/customer_churn.csv",
    model_type="random_forest",
    experiment_name="customer-churn-prediction",
    n_estimators=100,
    max_depth=5,
    register_model=False,
):
    print(f"=== Starting Training for {model_type} ===")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Feature definitions
    categorical_features = ["contract_type"]
    numerical_features = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
    target = "churn"

    X = df[numerical_features + categorical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    commit_hash, git_tag = get_git_commit_and_tag()
    dvc_hash = get_dvc_hash()

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{model_type}_run") as run:
        run_id = run.info.run_id
        print(f"Active MLflow Run ID: {run_id}")

        # Measure fit latency
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_latency_sec = time.time() - start_time

        # Predictions
        inference_start = time.time()
        y_pred = pipeline.predict(X_test)
        inference_latency_ms = (time.time() - inference_start) / len(X_test) * 1000.0
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")

        # Log parameters
        mlflow.log_params({
            "model_type": model_type,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "dataset_path": data_path,
            "dataset_version": git_tag,
            "dataset_git_commit": commit_hash[:8],
            "dataset_dvc_hash": dvc_hash,
        })

        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "training_latency_sec": training_latency_sec,
            "inference_latency_ms": inference_latency_ms,
        })

        # Log tags for provenance
        mlflow.set_tags({
            "dvc.dataset_version": git_tag,
            "dvc.hash": dvc_hash,
            "git.commit": commit_hash,
            "pipeline.stage": "training",
            "environment": "production-candidate",
        })

        # Artifact: Confusion matrix plot
        os.makedirs("artifacts", exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.6)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va="center", ha="center")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_type}")
        cm_path = f"artifacts/cm_{model_type}.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="evaluation_charts")

        # Artifact: Evaluation summary JSON
        summary = {
            "model_type": model_type,
            "dataset_version": git_tag,
            "git_commit": commit_hash,
            "dvc_hash": dvc_hash,
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc_auc,
            },
            "training_samples": len(X_train),
            "test_samples": len(X_test),
        }
        summary_path = f"artifacts/summary_{model_type}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path, artifact_path="model_summary")

        # Log Model with pickle/cloudpickle serialization
        kwargs = {"serialization_format": mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE}
        if register_model:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                registered_model_name="CustomerChurnModel",
                **kwargs,
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                **kwargs,
            )

        print(f"Run {run_id} logged to MLflow experiment '{experiment_name}'.")
        return run_id, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="random_forest")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    train(
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        register_model=args.register,
    )
