import argparse
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
)
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Load model + schema
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def load_model_and_schema(cancer_type):
    model_file = MODELS_DIR / f"{cancer_type}_model.joblib"
    schema_file = MODELS_DIR / f"{cancer_type}_schema.json"

    print(f"Detected model file: {model_file.name}")
    print(f"Detected schema file: {schema_file.name}")

    model = joblib.load(model_file)
    with open(schema_file, "r") as f:
        schema = json.load(f)

    return model, schema


# -------------------------
# Evaluation function
# -------------------------
def evaluate(cancer_type, data_csv):
    model, schema = load_model_and_schema(cancer_type)

    # Load dataset
    df = pd.read_csv(data_csv)

    # Detect target column
    target_col = schema.get("target", "target")
    if target_col not in df.columns:
        possible_targets = ["target", "diagnosis", "class", "label"]
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        else:
            raise ValueError(
                f"Target column not found. Available columns: {list(df.columns)}"
            )

    print(f"Detected target column: '{target_col}'")

    # Features + Target
    features = [c for c in df.columns if c != target_col]
    X = df[features].copy()
    y = df[target_col].copy()

    # Replace '?' with NaN
    X.replace("?", np.nan, inplace=True)

    # -------------------------
    # Impute Missing Values
    # -------------------------
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in X.columns if c not in numeric_cols]

    if numeric_cols:
        imputer_num = SimpleImputer(strategy="median")
        X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])

    if non_numeric_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        X[non_numeric_cols] = imputer_cat.fit_transform(X[non_numeric_cols])

    # -------------------------
    # Predictions
    # -------------------------
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
    else:
        # fallback for models without predict_proba
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        y_pred_encoded = le.transform(y_pred)
        y_proba = np.zeros((len(y), len(le.classes_)))
        for i, cls in enumerate(y_pred_encoded):
            y_proba[i, cls] = 1

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(y, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix (raw counts):")
    print(cm)

    # Confusion matrix plots
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    cm_file = BASE_DIR / f"confusion_matrix_{cancer_type}.png"
    plt.savefig(cm_file)
    plt.close()
    print(f"Saved confusion matrix to: {cm_file}")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    cm_norm_file = BASE_DIR / f"confusion_matrix_{cancer_type}_norm.png"
    plt.savefig(cm_norm_file)
    plt.close()
    print(f"Saved confusion matrix to: {cm_norm_file}")

    # ROC AUC (multiclass)
    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)
    if y_bin.shape[1] == 1:
        auc = roc_auc_score(y, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    print(f"Multiclass ROC AUC (macro, OVR): {auc:.4f}")

    # Log loss
    try:
        ll = log_loss(y, y_proba)
        print(f"Log loss: {ll:.4f}")
    except ValueError:
        print("Log loss could not be computed (probably missing probabilities).")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cancer detection models")
    parser.add_argument("cancer_type", type=str, help="Cancer type (lung, breast, skin)")
    parser.add_argument("data_csv", type=str, help="Path to dataset CSV")
    args = parser.parse_args()

    evaluate(args.cancer_type, args.data_csv)
