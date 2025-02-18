from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
from .logging_config import configure_logging

# Initialize Flask app
app = Flask(__name__)
configure_logging(app)

# Load models
MODELS = {
    "credit_card": {
        "cnn": None,
        "random_forest": None
    },
    "ecommerce": {
        "lstm": None,
        "gradient_boosting": None
    }
}

def load_models():
    """Load all models into memory"""
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
        # Credit Card Models
        MODELS["credit_card"]["cnn"] = load_model(
            os.path.join(base_dir, "notebooks/models/CNN/CNN_model.keras")
        )
        MODELS["credit_card"]["random_forest"] = joblib.load(
            os.path.join(base_dir, "notebooks/models/RandomForest/RandomForest_model.pkl")
        )
        
        # E-commerce Models
        MODELS["ecommerce"]["lstm"] = load_model(
            os.path.join(base_dir, "notebooks/models/LSTM/LSTM_model.keras")
        )
        MODELS["ecommerce"]["gradient_boosting"] = joblib.load(
            os.path.join(base_dir, "notebooks/models/GradientBoosting/GradientBoosting_model.pkl")
        )
        
        app.logger.info("All models loaded successfully")
        
    except Exception as e:
        app.logger.error(f"Model loading failed: {str(e)}")
        sys.exit(1)

# Load models at startup        
load_models()

@app.route('/predict/credit_card', methods=['POST'])
def predict_credit_card():
    """Endpoint for credit card fraud detection"""
    try:
        data = request.json
        app.logger.info(f"Credit card prediction request: {data}")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Select model type from input
        model_type = data.get('model_type', 'random_forest')
        model = MODELS["credit_card"].get(model_type)
        
        if not model:
            return jsonify({"error": "Invalid model type"}), 400
            
        # Preprocessing
        amount = input_df['Amount'].values[0]
        v_features = input_df[[f'V{i}' for i in range(1, 29)]].values.astype(np.float32)
        
        # Prediction
        if model_type == 'cnn':
            prediction = model.predict(v_features.reshape(1, 28, 1))[0][0]
        else:
            prediction = model.predict_proba(input_df)[0][1]
            
        return jsonify({
            "fraud_probability": float(prediction),
            "model_used": model_type,
            "amount": amount
        })
        
    except Exception as e:
        app.logger.error(f"Credit card prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/ecommerce', methods=['POST'])
def predict_ecommerce():
    """Endpoint for e-commerce fraud detection"""
    try:
        data = request.json
        app.logger.info(f"E-commerce prediction request: {data}")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Select model type
        model_type = data.get('model_type', 'gradient_boosting')
        model = MODELS["ecommerce"].get(model_type)
        
        if not model:
            return jsonify({"error": "Invalid model type"}), 400
            
        # Preprocessing
        purchase_value = input_df['purchase_value'].values[0]
        
        # Prediction
        if model_type == 'lstm':
            features = input_df.drop(columns=['purchase_value']).values.astype(np.float32)
            prediction = model.predict(features.reshape(1, 1, features.shape[1]))[0][0]
        else:
            prediction = model.predict_proba(input_df)[0][1]
            
        return jsonify({
            "fraud_probability": float(prediction),
            "model_used": model_type,
            "purchase_value": purchase_value
        })
        
    except Exception as e:
        app.logger.error(f"E-commerce prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)