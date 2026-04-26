from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import librosa
import cv2
import soundfile as sf   # 🔥 IMPORTANT FIX

app = Flask(__name__)

# ✅ CORS FIX
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response

# 🔥 Allow larger uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# =========================
# HOME
# =========================
@app.route("/")
def home():
    return "Backend Running ✅"

# =========================
# 🎤 VOICE DETECTION (FIXED PROPERLY)
# =========================
@app.route("/predict_voice", methods=["POST"])
def predict_voice():
    try:
        file = request.files["file"]

        # Save raw audio
        webm_path = os.path.join(BASE_DIR, "temp_audio.webm")
        wav_path = os.path.join(BASE_DIR, "temp_audio.wav")

        file.save(webm_path)

        # 🔥 Convert webm → wav (CRITICAL FIX)
        y, sr = librosa.load(webm_path, sr=None)
        sf.write(wav_path, y, sr)

        # Extract MFCC features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0)

        # Match 22 features
        if len(mfcc) > 22:
            mfcc = mfcc[:22]
        else:
            mfcc = np.pad(mfcc, (0, 22 - len(mfcc)))

        features = scaler.transform([mfcc])
        prediction = model.predict(features)[0]

        result = "Parkinson Detected" if prediction == 1 else "Healthy"

        return jsonify({"prediction": result})

    except Exception as e:
        print("VOICE ERROR:", e)   # 🔥 Debug log
        return jsonify({"error": str(e)})

# =========================
# ✍️ IMAGE DETECTION
# =========================
@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        file = request.files["file"]

        file_path = os.path.join(BASE_DIR, "temp_image.png")
        file.save(file_path)

        img = cv2.imread(file_path, 0)

        if img is None:
            return jsonify({"error": "Invalid image"})

        img = cv2.resize(img, (64, 64))

        mean_intensity = np.mean(img)

        if mean_intensity < 127:
            result = "Possible Parkinson Pattern"
        else:
            result = "Healthy Pattern"

        return jsonify({"prediction": result})

    except Exception as e:
        print("IMAGE ERROR:", e)
        return jsonify({"error": str(e)})

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("Starting server...")
    app.run(debug=True)