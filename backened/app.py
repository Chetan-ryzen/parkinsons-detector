from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import librosa
import cv2

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL (VOICE FEATURES)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    return "Backend Running ✅"

# =========================
# EXISTING FEATURE-BASED PREDICTION
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    model_type = data.get("type", "voice")
    features = data.get("features", [])

    if model_type == "voice":
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)[0]

        result = "Parkinson Detected" if prediction == 1 else "Healthy"

    elif model_type == "handwriting":
        result = "Handwriting model coming soon"

    else:
        result = "Invalid type"

    return jsonify({"prediction": result})

# =========================
# 🎤 REAL VOICE PREDICTION
# =========================
@app.route("/predict_voice", methods=["POST"])
def predict_voice():
    try:
        file = request.files["file"]

        file_path = os.path.join(BASE_DIR, "temp_audio.wav")
        file.save(file_path)

        # Load audio
        y, sr = librosa.load(file_path)

        # Extract MFCC features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0)

        # Ensure 22 features (match model)
        if len(mfcc) > 22:
            mfcc = mfcc[:22]
        else:
            mfcc = np.pad(mfcc, (0, 22 - len(mfcc)))

        features = scaler.transform([mfcc])
        prediction = model.predict(features)[0]

        result = "Parkinson Detected" if prediction == 1 else "Healthy"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# ✍️ BASIC HANDWRITING DETECTION
# =========================
@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        file = request.files["file"]

        file_path = os.path.join(BASE_DIR, "temp_image.png")
        file.save(file_path)

        # Load image (grayscale)
        img = cv2.imread(file_path, 0)

        # Resize
        img = cv2.resize(img, (64, 64))

        # Simple feature
        mean_intensity = np.mean(img)

        # Dummy logic (placeholder)
        if mean_intensity < 127:
            result = "Possible Parkinson Pattern"
        else:
            result = "Healthy Pattern"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    print("Starting server...")
    app.run(debug=True)