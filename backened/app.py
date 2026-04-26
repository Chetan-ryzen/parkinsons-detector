from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os

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
# 🎤 VOICE FILE UPLOAD (COMING SOON MODEL)
# =========================
@app.route("/predict_voice", methods=["POST"])
def predict_voice():
    try:
        file = request.files["file"]

        # Save temporarily (optional)
        file_path = os.path.join(BASE_DIR, "temp_audio.wav")
        file.save(file_path)

        # 🚧 Future: extract features using librosa here

        return jsonify({"prediction": "Voice model coming soon"})

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# ✍️ HANDWRITING IMAGE UPLOAD (COMING SOON MODEL)
# =========================
@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        file = request.files["file"]

        # Save temporarily
        file_path = os.path.join(BASE_DIR, "temp_image.png")
        file.save(file_path)

        # 🚧 Future: load CNN model and predict here

        return jsonify({"prediction": "Handwriting model coming soon"})

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    print("Starting server...")
    app.run(debug=True)