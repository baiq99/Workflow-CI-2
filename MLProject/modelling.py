import pandas as pd
import numpy as np
import os
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, f1_score, precision_score, recall_score
)
import joblib

# ========== Argument Parser ==========
parser = argparse.ArgumentParser(description="Train ML model for CI workflow")
parser.add_argument(
    "--data_path",
    type=str,
    default="dataset_preprocessed/online_shoppers_intention_preprocessed.csv",
    help="Path to preprocessed CSV file"
)
args = parser.parse_args()

# ========== Load Dataset ==========
data = pd.read_csv(args.data_path)
X = data.drop("Revenue", axis=1)
y = data["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========== Model Training ==========
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True)
}

best_model = None
best_score = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"\n📌 Model: {name}")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("ROC AUC:", roc_auc)
    print("Precision:", precision)
    print("Recall:", recall)
    print(classification_report(y_test, y_pred))

    mlflow.log_metric(f"{name}_accuracy", acc)
    mlflow.log_metric(f"{name}_f1", f1)
    mlflow.log_metric(f"{name}_precision", precision)
    mlflow.log_metric(f"{name}_recall", recall)
    mlflow.log_metric(f"{name}_roc_auc", roc_auc)

    if acc > best_score:
        best_model = model
        best_model_name = name
        best_score = acc

# ========== Simpan model ke outputs ==========
os.makedirs("outputs", exist_ok=True)
joblib.dump(best_model, "outputs/best_model.pkl")

# ========== Logging model MLflow agar bisa digunakan build-docker ==========
mlflow.sklearn.log_model(best_model, artifact_path="mlflow_model")

print(f"\n✅ Model terbaik: {best_model_name} (Accuracy: {best_score:.4f})")
