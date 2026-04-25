from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def home():
    return "Backend Running ✅"

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

if __name__ == "__main__":
    print("Starting server...")   # DEBUG LINE
    app.run(debug=True)