

import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "models", "artifacts.pkl"), "rb") as f:
    artifacts = pickle.load(f)

FEATURES = artifacts["features"]
scaler = artifacts["scaler"]
reg_model = artifacts["best_regressor"]
clf_model = artifacts["best_classifier"]
feature_importance = artifacts["feature_importance"] or {}
focus_mean = artifacts["focus_hours_mean"]

app = Flask(__name__, static_folder=None)


def build_recommendation(focus_pred, is_high_prod, prob_high, inputs):
    reasons = []
    if inputs["sleep_hours"] < 6.5:
        reasons.append("low sleep")
    if inputs["social_media_hours"] > 3.5:
        reasons.append("high social media use")
    if inputs["is_exam_week"]:
        reasons.append("exam week (usually boosts focus)")
    if inputs["prev_day_focus_hours"] < focus_mean * 0.6:
        reasons.append("low momentum from yesterday")
    if inputs["energy_self_log"] < 2.5:
        reasons.append("low self-reported energy")

    if is_high_prod:
        headline = "This looks like a high-productivity day."
        if inputs["sleep_hours"] >= 7:
            best_window = "morning (9am-12pm)"
        else:
            best_window = "late morning to early afternoon (11am-2pm)"
        tip = f"Good conditions — consider tackling your hardest task during your {best_window} window."
    else:
        headline = "This looks like a lower-focus day."
        tip = "Consider lighter tasks (review, admin) rather than deep work, or address: " + (", ".join(reasons) if reasons else "no single strong driver — likely just noise.")

    return {
        "headline": headline,
        "tip": tip,
        "contributing_factors": reasons if reasons else ["No strong risk factors detected"],
    }


@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "app", "static"), "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(os.path.join(BASE_DIR, "app", "static"), path)


@app.route("/api/feature_importance")
def get_feature_importance():
    return jsonify(feature_importance)


@app.route("/api/model_info")
def model_info():
    return jsonify({
        "regressor": artifacts["best_regressor_name"],
        "classifier": artifacts["best_classifier_name"],
        "results": artifacts["results"],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        row = {
            "is_weekend": int(data.get("is_weekend", 0)),
            "is_exam_week": int(data.get("is_exam_week", 0)),
            "sleep_hours": float(data.get("sleep_hours", 7)),
            "social_media_hours": float(data.get("social_media_hours", 2)),
            "study_app_hours": float(data.get("study_app_hours", 2)),
            "class_hours": float(data.get("class_hours", 4)),
            "energy_self_log": float(data.get("energy_self_log", 3)),
            "prev_day_focus_hours": float(data.get("prev_day_focus_hours", focus_mean)),
            "rolling_3day_focus_avg": float(data.get("rolling_3day_focus_avg", focus_mean)),
        }
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    X = pd.DataFrame([row])[FEATURES]
    X_scaled = scaler.transform(X)

    focus_pred = float(reg_model.predict(X_scaled)[0])
    focus_pred = max(0, round(focus_pred, 2))

    clf_pred = int(clf_model.predict(X_scaled)[0])
    if hasattr(clf_model, "predict_proba"):
        prob_high = float(clf_model.predict_proba(X_scaled)[0][1])
    else:
        prob_high = float(clf_pred)

    rec = build_recommendation(focus_pred, clf_pred, prob_high, row)

    return jsonify({
        "predicted_focus_hours": focus_pred,
        "high_productivity_day": bool(clf_pred),
        "probability_high_productivity": round(prob_high, 3),
        "recommendation": rec,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
