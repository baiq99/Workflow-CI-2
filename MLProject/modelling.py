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
    accuracy_score,
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
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
def load_data(path):
    return pd.read_csv(path)

data = load_data(args.data_path)
X = data.drop("Revenue", axis=1)
y = data["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========== MLflow Setup ==========
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Eksperimen_Modeling_CI")

# ========== Model Training ==========
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True)
}

best_model = None
best_score = 0
best_model_name = ""

# NOTE: start_run() dihapus karena mlflow run sudah memulai run otomatis
# with mlflow.start_run():

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

# Save best model to MLflow and as local file
mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name="BestCIModel")
print(f"\n✅ Model terbaik: {best_model_name} (Accuracy: {best_score:.4f})")

# Save to outputs directory
os.makedirs("outputs", exist_ok=True)
joblib.dump(best_model, "outputs/best_model.pkl")
mlflow.log_artifact("outputs/best_model.pkl")
