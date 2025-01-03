from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model at startup
model = joblib.load("model_training/fraud_model.pkl")

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """Health check endpoint for testing"""
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint for fraud detection"""
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    return jsonify({'fraud_prediction': int(prediction[0]), 'fraud_probability': probability})

if __name__ == '__main__':
    app.run(debug=True)
