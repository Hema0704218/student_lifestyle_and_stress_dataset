import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from flask import (
    Flask, render_template, request,
    jsonify, redirect, url_for, flash, session
)
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
import joblib

# ──────────────────────────────────────────────
# APP CONFIG
# ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "student_stress_secret_key_2024"

UPLOAD_FOLDER  = "uploads"
MODEL_FOLDER   = "models"
ALLOWED_EXT    = {"csv"}
TARGET_COLUMN  = "Stress_Level"
RANDOM_STATE   = 42
TEST_SIZE      = 0.20

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER,  exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB

# Paths
MODEL_PATH   = os.path.join(MODEL_FOLDER, "stress_model.pkl")
SCALER_PATH  = os.path.join(MODEL_FOLDER, "stress_scaler.pkl")
META_PATH    = os.path.join(MODEL_FOLDER, "stress_meta.pkl")


# ──────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="none", transparent=True)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def load_model_files():
    """Load model, scaler, meta from disk. Returns None if not found."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, META_PATH]):
        return None, None, None
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta   = joblib.load(META_PATH)
    return model, scaler, meta


def model_is_trained():
    return all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, META_PATH])


# ──────────────────────────────────────────────
# ML PIPELINE FUNCTIONS
# ──────────────────────────────────────────────
def preprocess_df(df, target_col):
    df = df.copy()
    df.dropna(subset=[target_col], inplace=True)

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def train_pipeline(df, target_col):
    df_proc, encoders = preprocess_df(df, target_col)

    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    candidates = {
        "Random Forest"      : RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=150, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM"                : SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    cv_results = {}
    for name, mdl in candidates.items():
        scores = cross_val_score(mdl, X_train_sc, y_train, cv=5, scoring="accuracy")
        cv_results[name] = {"mean": round(float(scores.mean()), 4),
                            "std" : round(float(scores.std()),  4)}

    best_name  = max(cv_results, key=lambda k: cv_results[k]["mean"])
    best_model = candidates[best_name]
    best_model.fit(X_train_sc, y_train)

    y_pred  = best_model.predict(X_test_sc)
    test_acc = round(float(accuracy_score(y_test, y_pred)), 4)
    report  = classification_report(y_test, y_pred, output_dict=True)

    # Confusion matrix
    cm  = confusion_matrix(y_test, y_pred)
    classes = sorted(y.unique())
    target_le = encoders.get(target_col)
    if target_le is not None:
        class_labels = [str(c) for c in target_le.inverse_transform(classes)]
    else:
        class_labels = [str(c) for c in classes]

    fig_cm, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix – {best_name}", fontsize=13, pad=10)
    cm_img = fig_to_base64(fig_cm)

    # Feature importance
    fi_img = None
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig_fi, ax2 = plt.subplots(figsize=(7, max(4, len(fi)*0.4)))
        fi.plot(kind="barh", color="#4f8ef7", edgecolor="none", ax=ax2)
        ax2.set_title("Feature Importance", fontsize=13)
        ax2.set_xlabel("Importance Score")
        fi_img = fig_to_base64(fig_fi)

    # Save artefacts
    meta = {
        "feature_cols"  : feature_cols,
        "encoders"      : encoders,
        "target_col"    : target_col,
        "best_model_name": best_name,
        "cv_results"    : cv_results,
        "test_accuracy" : test_acc,
        "class_labels"  : class_labels,
        "report"        : report,
        "cm_img"        : cm_img,
        "fi_img"        : fi_img,
    }
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)
    joblib.dump(meta,       META_PATH)

    return meta


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

# ── Home ──────────────────────────────────────
@app.route("/")
def index():
    trained = model_is_trained()
    meta    = joblib.load(META_PATH) if trained else None
    return render_template("index.html", trained=trained, meta=meta)


# ── Upload & Train ────────────────────────────
@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        if "dataset" not in request.files:
            flash("No file selected.", "danger")
            return redirect(request.url)

        file = request.files["dataset"]
        if file.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Only CSV files are allowed.", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)

            if TARGET_COLUMN not in df.columns:
                flash(f"Column '{TARGET_COLUMN}' not found. Available: {', '.join(df.columns)}", "danger")
                return redirect(request.url)

            meta = train_pipeline(df, TARGET_COLUMN)
            flash(f"✅ Model trained! Best: {meta['best_model_name']} | Accuracy: {meta['test_accuracy']*100:.2f}%", "success")
            return redirect(url_for("results"))

        except Exception as e:
            flash(f"Training failed: {str(e)}", "danger")
            return redirect(request.url)

    return render_template("train.html")


# ── Results / Dashboard ───────────────────────
@app.route("/results")
def results():
    if not model_is_trained():
        flash("No trained model found. Please train first.", "warning")
        return redirect(url_for("train"))

    meta = joblib.load(META_PATH)
    return render_template("results.html", meta=meta)


# ── Predict ───────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not model_is_trained():
        flash("No trained model found. Please train first.", "warning")
        return redirect(url_for("train"))

    model, scaler, meta = load_model_files()
    feature_cols = meta["feature_cols"]
    encoders     = meta["encoders"]
    target_col   = meta["target_col"]

    prediction   = None
    confidence   = None
    class_probs  = None

    if request.method == "POST":
        try:
            input_data = {}
            for col in feature_cols:
                val = request.form.get(col, "0")
                try:
                    input_data[col] = float(val)
                except ValueError:
                    # categorical – encode
                    if col in encoders:
                        le = encoders[col]
                        if val in le.classes_:
                            input_data[col] = float(le.transform([val])[0])
                        else:
                            input_data[col] = 0.0
                    else:
                        input_data[col] = 0.0

            row    = pd.DataFrame([input_data])[feature_cols]
            row_sc = scaler.transform(row)

            pred_raw = model.predict(row_sc)[0]
            target_le = encoders.get(target_col)
            if target_le:
                prediction = str(target_le.inverse_transform([int(pred_raw)])[0])
            else:
                prediction = str(pred_raw)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(row_sc)[0]
                class_labels = meta["class_labels"]
                class_probs  = [{"label": cl, "prob": round(float(p)*100, 1)}
                                 for cl, p in zip(class_labels, probs)]
                confidence   = round(float(max(probs)) * 100, 1)

        except Exception as e:
            flash(f"Prediction error: {str(e)}", "danger")

    # Build form fields
    form_fields = []
    for col in feature_cols:
        if col in encoders:
            form_fields.append({
                "name"   : col,
                "label"  : col.replace("_", " ").title(),
                "type"   : "select",
                "options": list(encoders[col].classes_),
            })
        else:
            form_fields.append({
                "name" : col,
                "label": col.replace("_", " ").title(),
                "type" : "number",
            })

    return render_template(
        "predict.html",
        form_fields  = form_fields,
        prediction   = prediction,
        confidence   = confidence,
        class_probs  = class_probs,
    )


# ── API: Predict JSON ─────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not model_is_trained():
        return jsonify({"error": "Model not trained yet."}), 400

    model, scaler, meta = load_model_files()
    feature_cols = meta["feature_cols"]
    encoders     = meta["encoders"]
    target_col   = meta["target_col"]

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body."}), 400

    try:
        input_data = {}
        for col in feature_cols:
            val = data.get(col, 0)
            if col in encoders:
                le = encoders[col]
                val_str = str(val)
                input_data[col] = float(le.transform([val_str])[0]) if val_str in le.classes_ else 0.0
            else:
                input_data[col] = float(val)

        row    = pd.DataFrame([input_data])[feature_cols]
        row_sc = scaler.transform(row)
        pred   = model.predict(row_sc)[0]

        target_le = encoders.get(target_col)
        label = str(target_le.inverse_transform([int(pred)])[0]) if target_le else str(pred)

        result = {"prediction": label}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(row_sc)[0]
            result["probabilities"] = {cl: round(float(p), 4)
                                        for cl, p in zip(meta["class_labels"], probs)}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Delete / Reset ────────────────────────────
@app.route("/reset", methods=["POST"])
def reset():
    for p in [MODEL_PATH, SCALER_PATH, META_PATH]:
        if os.path.exists(p):
            os.remove(p)
    flash("Model reset. Please upload a new dataset to retrain.", "info")
    return redirect(url_for("index"))


# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)