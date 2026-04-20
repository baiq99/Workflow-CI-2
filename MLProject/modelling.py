import os
import argparse
import warnings
import joblib
import pandas as pd
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
    recall_score,
)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML model for CI workflow")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset_preprocessed/online_shoppers_intention_preprocessed.csv",
        help="Path to preprocessed CSV file",
    )
    return parser.parse_args()


def validate_dataset(df: pd.DataFrame):
    if df.empty:
        raise ValueError("Dataset kosong. Pastikan file CSV berisi data.")

    if "Revenue" not in df.columns:
        raise ValueError("Kolom target 'Revenue' tidak ditemukan pada dataset.")

    if df["Revenue"].nunique() < 2:
        raise ValueError("Target 'Revenue' harus memiliki minimal 2 kelas.")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset masih mengandung missing values. Selesaikan preprocessing terlebih dahulu.")


def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan: {args.data_path}")

    print(f"Loading dataset from: {args.data_path}")
    data = pd.read_csv(args.data_path)

    validate_dataset(data)

    X = data.drop("Revenue", axis=1)
    y = data["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ),
        "SVM": SVC(
            probability=True,
            kernel="rbf",
            random_state=42
        )
    }

    os.makedirs("outputs", exist_ok=True)

    best_model = None
    best_model_name = None
    best_score = -1.0

    mlflow.set_experiment("workflow-ci-experiment")

    with mlflow.start_run(run_name="train_models_ci"):
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("n_features", X.shape[1])

        for name, model in models.items():
            print(f"\n===== Training {name} =====")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)

            report = classification_report(y_test, y_pred, zero_division=0)

            print(f"Model      : {name}")
            print(f"Accuracy   : {acc:.4f}")
            print(f"F1 Score   : {f1:.4f}")
            print(f"Precision  : {precision:.4f}")
            print(f"Recall     : {recall:.4f}")
            print(f"ROC AUC    : {roc_auc:.4f}")
            print("Classification Report:")
            print(report)

            mlflow.log_metric(f"{name}_accuracy", acc)
            mlflow.log_metric(f"{name}_f1", f1)
            mlflow.log_metric(f"{name}_precision", precision)
            mlflow.log_metric(f"{name}_recall", recall)
            mlflow.log_metric(f"{name}_roc_auc", roc_auc)

            report_path = f"outputs/{name}_classification_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            mlflow.log_artifact(report_path)

            if acc > best_score:
                best_score = acc
                best_model = model
                best_model_name = name

        if best_model is None:
            raise RuntimeError("Tidak ada model yang berhasil dilatih.")

        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_metric("best_model_accuracy", best_score)

        mlflow.sklearn.log_model(best_model, artifact_path="model")

        best_model_pkl_path = "outputs/best_model.pkl"
        joblib.dump(best_model, best_model_pkl_path)
        mlflow.log_artifact(best_model_pkl_path)

        summary_path = "outputs/best_model_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Best Accuracy: {best_score:.4f}\n")
        mlflow.log_artifact(summary_path)

        print(f"\nBest model: {best_model_name}")
        print(f"Best accuracy: {best_score:.4f}")
        print(f"Saved pickle model to: {best_model_pkl_path}")


if __name__ == "__main__":
    main()