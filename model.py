# ============================================================
#   Student Lifestyle & Stress – Complete ML Pipeline
#   Includes: EDA, Preprocessing, Training, Testing,
#             Prediction, Accuracy, Save & Load Model
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
CSV_PATH      = "student-lifestyle-and-stress-dataset.csv"   # <- change if needed
TARGET_COLUMN = "Stress_Level"                                # <- target label column
MODEL_PATH    = "student_stress_model.pkl"
SCALER_PATH   = "student_stress_scaler.pkl"
ENCODER_PATH  = "student_stress_encoders.pkl"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2


# ════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ════════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    """Load CSV and show basic info."""
    print("\n" + "="*60)
    print("  STEP 1 – LOADING DATA")
    print("="*60)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please place the CSV file in the same folder as this script."
        )

    df = pd.read_csv(path)
    print(f"  Rows    : {df.shape[0]}")
    print(f"  Columns : {df.shape[1]}")
    print(f"\n  Columns : {df.columns.tolist()}")
    print(f"\n  First 3 rows:\n{df.head(3).to_string()}")
    print(f"\n  Missing values:\n{df.isnull().sum().to_string()}")
    return df


# ════════════════════════════════════════════════════════════
# 2.  EXPLORATORY DATA ANALYSIS  (saves plots as PNG)
# ════════════════════════════════════════════════════════════
def run_eda(df: pd.DataFrame, target: str) -> None:
    print("\n" + "="*60)
    print("  STEP 2 – EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # Target distribution
    plt.figure(figsize=(7, 4))
    ax = df[target].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
    ax.set_title("Stress Level Distribution")
    ax.set_xlabel("Stress Level")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("eda_target_distribution.png", dpi=100)
    plt.close()
    print("  Saved: eda_target_distribution.png")

    # Correlation heatmap (numeric only)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 1:
        plt.figure(figsize=(10, 7))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("eda_correlation_heatmap.png", dpi=100)
        plt.close()
        print("  Saved: eda_correlation_heatmap.png")

    print(f"\n  Target value counts:\n{df[target].value_counts().to_string()}")


# ════════════════════════════════════════════════════════════
# 3.  PREPROCESSING
# ════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame, target: str):
    """
    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoders
    """
    print("\n" + "="*60)
    print("  STEP 3 – PREPROCESSING")
    print("="*60)

    df = df.copy()

    # 3a. Drop rows with missing target
    df.dropna(subset=[target], inplace=True)

    # 3b. Fill missing numeric with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # 3c. Encode categorical columns (except target)
    label_encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  Encoded : {col}  →  classes = {list(le.classes_)}")

    # 3d. Split features / target
    X = df.drop(columns=[target])
    y = df[target]

    print(f"\n  Features  : {X.columns.tolist()}")
    print(f"  Target    : {target}")
    print(f"  X shape   : {X.shape}")
    print(f"  y shape   : {y.shape}")

    # 3e. Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # 3f. Feature scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("  Scaling done (StandardScaler)")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, label_encoders, X.columns.tolist()


# ════════════════════════════════════════════════════════════
# 4.  MODEL TRAINING  (multiple models → pick best)
# ════════════════════════════════════════════════════════════
def train_models(X_train, y_train):
    print("\n" + "="*60)
    print("  STEP 4 – MODEL TRAINING")
    print("="*60)

    candidates = {
        "Random Forest"        : RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=150, random_state=RANDOM_STATE),
        "Logistic Regression"  : LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM (RBF)"            : SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        results[name] = {
            "model"    : model,
            "cv_mean"  : cv_scores.mean(),
            "cv_std"   : cv_scores.std(),
        }
        print(f"  {name:<25}  CV Acc = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best_model = results[best_name]["model"]
    print(f"\n  ✅ Best model : {best_name}  (CV Acc = {results[best_name]['cv_mean']:.4f})")

    # Final fit on full training set
    best_model.fit(X_train, y_train)
    return best_model, best_name, results


# ════════════════════════════════════════════════════════════
# 5.  MODEL TESTING & ACCURACY
# ════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, model_name: str):
    print("\n" + "="*60)
    print("  STEP 5 – MODEL TESTING & ACCURACY")
    print("="*60)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n  Model         : {model_name}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    plt.savefig("model_confusion_matrix.png", dpi=100)
    plt.close()
    print("  Saved: model_confusion_matrix.png")
    return acc, y_pred


# ════════════════════════════════════════════════════════════
# 6.  SAVE MODEL
# ════════════════════════════════════════════════════════════
def save_model(model, scaler, label_encoders, feature_cols):
    print("\n" + "="*60)
    print("  STEP 6 – SAVING MODEL")
    print("="*60)

    joblib.dump(model,         MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(
        {"encoders": label_encoders, "feature_cols": feature_cols},
        ENCODER_PATH
    )

    print(f"  Model   saved → {MODEL_PATH}")
    print(f"  Scaler  saved → {SCALER_PATH}")
    print(f"  Encoders saved → {ENCODER_PATH}")


# ════════════════════════════════════════════════════════════
# 7.  LOAD MODEL
# ════════════════════════════════════════════════════════════
def load_model():
    print("\n" + "="*60)
    print("  STEP 7 – LOADING SAVED MODEL")
    print("="*60)

    for path in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}. Run save_model() first.")

    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    meta          = joblib.load(ENCODER_PATH)
    label_encoders = meta["encoders"]
    feature_cols   = meta["feature_cols"]

    print(f"  Model   loaded ← {MODEL_PATH}")
    print(f"  Scaler  loaded ← {SCALER_PATH}")
    print(f"  Encoders loaded ← {ENCODER_PATH}")
    return model, scaler, label_encoders, feature_cols


# ════════════════════════════════════════════════════════════
# 8.  PREDICTION  (single sample or batch)
# ════════════════════════════════════════════════════════════
def predict(input_data: dict, model=None, scaler=None,
            label_encoders=None, feature_cols=None):
    """
    Predict stress level for one or more samples.

    Parameters
    ----------
    input_data : dict  – {column_name: value}  for a SINGLE sample, or
                 list of dicts for multiple samples.
    model, scaler, label_encoders, feature_cols – if None, loaded from disk.

    Returns
    -------
    predictions : list of predicted class labels
    probabilities : list of probability arrays (if model supports predict_proba)
    """
    print("\n" + "="*60)
    print("  STEP 8 – PREDICTION")
    print("="*60)

    # Auto-load if not passed
    if model is None:
        model, scaler, label_encoders, feature_cols = load_model()

    # Normalise to list of dicts
    samples = input_data if isinstance(input_data, list) else [input_data]

    df_input = pd.DataFrame(samples)

    # Encode categoricals using saved encoders
    for col in df_input.columns:
        if col in label_encoders:
            le = label_encoders[col]
            df_input[col] = df_input[col].astype(str).apply(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )

    # Align columns to training order
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_cols]

    # Fill any NaN
    df_input.fillna(0, inplace=True)

    # Scale
    X_scaled = scaler.transform(df_input)

    # Predict
    predictions = model.predict(X_scaled)

    # Probabilities (if supported)
    try:
        probs = model.predict_proba(X_scaled)
    except AttributeError:
        probs = [None] * len(predictions)

    # Decode target if encoder exists
    target_le = label_encoders.get(TARGET_COLUMN)

    for i, (pred, prob) in enumerate(zip(predictions, probs)):
        label = target_le.inverse_transform([pred])[0] if target_le else pred
        print(f"  Sample {i+1}:")
        print(f"    Predicted Stress Level : {label}")
        if prob is not None:
            classes = (target_le.classes_ if target_le else model.classes_)
            prob_str = "  |  ".join(f"{c}: {p:.3f}" for c, p in zip(classes, prob))
            print(f"    Probabilities          : {prob_str}")

    return list(predictions), list(probs)


# ════════════════════════════════════════════════════════════
# 9.  FEATURE IMPORTANCE PLOT
# ════════════════════════════════════════════════════════════
def plot_feature_importance(model, feature_cols):
    if not hasattr(model, "feature_importances_"):
        print("\n  (Feature importance not available for this model)")
        return

    print("\n" + "="*60)
    print("  STEP 9 – FEATURE IMPORTANCE")
    print("="*60)

    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance.sort_values(ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))
    importance.plot(kind="bar", color="darkorange", edgecolor="black")
    plt.title("Feature Importance")
    plt.ylabel("Importance Score")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=100)
    plt.close()
    print("  Saved: feature_importance.png")
    print(f"\n  Top features:\n{importance.head(10).to_string()}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 1. Load ───────────────────────────────────────────
    df = load_data(CSV_PATH)

    # ── 2. EDA ────────────────────────────────────────────
    run_eda(df, TARGET_COLUMN)

    # ── 3. Preprocess ─────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_cols = \
        preprocess(df, TARGET_COLUMN)

    # ── 4. Train ──────────────────────────────────────────
    best_model, best_name, all_results = train_models(X_train, y_train)

    # ── 5. Test / Accuracy ────────────────────────────────
    accuracy, y_pred = evaluate_model(best_model, X_test, y_test, best_name)

    # ── 6. Save model ─────────────────────────────────────
    save_model(best_model, scaler, label_encoders, feature_cols)

    # ── 7. Load model (verification) ──────────────────────
    loaded_model, loaded_scaler, loaded_encoders, loaded_cols = load_model()

    # ── 8. Predict – sample new student data ──────────────
    #   Update these values to match your dataset's columns!
    sample_student = {
        "Age"                    : 20,
        "Gender"                 : "Male",
        "Study_Hours_Per_Day"    : 6,
        "Sleep_Hours_Per_Day"    : 7,
        "Physical_Activity_Hours": 1.5,
        "Social_Hours_Per_Day"   : 2,
        "GPA"                    : 3.4,
        "Extracurricular_Activities": "Yes",
    }

    predictions, probabilities = predict(
        sample_student,
        model          = loaded_model,
        scaler         = loaded_scaler,
        label_encoders = loaded_encoders,
        feature_cols   = loaded_cols,
    )

    # ── 9. Feature importance ─────────────────────────────
    plot_feature_importance(best_model, feature_cols)

    # ── Summary ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  ✅  PIPELINE COMPLETE")
    print("="*60)
    print(f"  Best Model    : {best_name}")
    print(f"  Test Accuracy : {accuracy*100:.2f}%")
    print(f"  Saved files   : {MODEL_PATH}, {SCALER_PATH}, {ENCODER_PATH}")
    print(f"  EDA plots     : eda_target_distribution.png, eda_correlation_heatmap.png")
    print(f"  CM plot       : model_confusion_matrix.png")
    print(f"  FI plot       : feature_importance.png")
    print("="*60)