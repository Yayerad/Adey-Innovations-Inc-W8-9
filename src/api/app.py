# src/api/app.py
import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import requests  # Added missing import
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from flask.logging import default_handler
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Initialize Flask application
app = Flask(__name__)

# Configure logging
def configure_logging():
    """Configure logging for Flask application"""
    # Remove default handler
    app.logger.removeHandler(default_handler)
    
    # File handler
    file_handler = logging.FileHandler('fraud_detection.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Add handlers
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.DEBUG)

configure_logging()

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
        # Get the correct base directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
        # Credit Card Models
        MODELS["credit_card"]["cnn"] = load_model(
            os.path.join(base_dir, "notebooks/models/CNN/CNN_model.keras")  # Add .keras extension
        )
        MODELS["credit_card"]["random_forest"] = joblib.load(
            os.path.join(base_dir, "notebooks/models/RandomForest/RandomForest_model.pkl")
        )
        
        # E-commerce Models
        MODELS["ecommerce"]["lstm"] = load_model(
            os.path.join(base_dir, "notebooks/models/LSTM/LSTM_model.keras")  # Add .keras extension
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

# API Endpoints
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

# Dashboard Data Endpoints
@app.route('/api/summary')
def get_summary():
    """Get summary statistics"""
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        df = pd.read_csv(os.path.join(base_dir, "src/data/fraud_data_processed.csv"))
        total = len(df)
        fraud = df['class'].sum()
        return jsonify({
            "total_transactions": total,
            "total_fraud": int(fraud),
            "fraud_percentage": (fraud/total)*100
        })
    except Exception as e:
        app.logger.error(f"Summary data error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/fraud_trend')
def get_fraud_trend():
    """Get fraud trend over time"""
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        df = pd.read_csv(os.path.join(base_dir, "src/data/creditcard_processed.csv"))
        df['date'] = pd.to_datetime(df['Time'], unit='s').dt.date
        trend = df.groupby('date')['Class'].sum().reset_index()
        return trend.rename(columns={'Class': 'fraud_cases'}).to_json(orient='records')
    except Exception as e:
        app.logger.error(f"Fraud trend error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/geo_fraud')
def get_geo_fraud():
    """Get fraud by country"""
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        df = pd.read_csv(os.path.join(base_dir, "src/data/fraud_data_processed.csv"))
        geo = df.groupby('country')['class'].sum().reset_index()
        return geo.rename(columns={'class': 'fraud_cases'}).to_json(orient='records')
    except Exception as e:
        app.logger.error(f"Geo fraud error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/device_fraud')
def get_device_fraud():
    """Get fraud by device"""
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        df = pd.read_csv(os.path.join(base_dir, "src/data/fraud_data_processed.csv"))
        device = df.groupby('device_id')['class'].sum().reset_index()
        return device.rename(columns={'class': 'fraud_cases'}).to_json(orient='records')
    except Exception as e:
        app.logger.error(f"Device fraud error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/browser_fraud')
def get_browser_fraud():
    """Get fraud by browser"""
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        df = pd.read_csv(os.path.join(base_dir, "src/data/fraud_data_processed.csv"))
        browser_cols = [c for c in df.columns if 'browser_' in c]
        browser = df[browser_cols].sum().reset_index()
        browser.columns = ['browser', 'fraud_cases']
        browser['browser'] = browser['browser'].str.replace('browser_', '')
        return browser.to_json(orient='records')
    except Exception as e:
        app.logger.error(f"Browser fraud error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Dash Application
def init_dash_app(flask_app):
    """Create a Dash application"""
    dash_app = dash.Dash(
        server=flask_app,
        url_base_pathname='/dashboard/',
        external_stylesheets=['/static/styles.css']
    )
    
    dash_app.title = "Fraud Detection Dashboard"
    
    # Define layout
    dash_app.layout = html.Div([
        html.H1("Fraud Detection Analytics", style={'textAlign': 'center'}),
        
        # Summary Cards
        html.Div([
            html.Div(id='total-transactions-card', className='card'),
            html.Div(id='fraud-cases-card', className='card'),
            html.Div(id='fraud-percentage-card', className='card')
        ], className='row'),
        
        # Main Charts
        html.Div([
            dcc.Graph(id='fraud-trend-chart', className='six columns'),
            dcc.Graph(id='geo-map', className='six columns')
        ], className='row'),
        
        # Device/Browser Charts
        html.Div([
            dcc.Graph(id='device-fraud-chart', className='six columns'),
            dcc.Graph(id='browser-fraud-chart', className='six columns')
        ], className='row'),
        
        dcc.Interval(
            id='interval-component',
            interval=60*1000,  # 1 minute
            n_intervals=0
        )
    ])
    
    # Register callbacks
    @dash_app.callback(
        [Output('total-transactions-card', 'children'),
         Output('fraud-cases-card', 'children'),
         Output('fraud-percentage-card', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_summary(_):
        try:
            response = requests.get('http://localhost:5000/api/summary')
            response.raise_for_status()
            data = response.json()
            return [
                html.Div([
                    html.H3(f"{data['total_transactions']:,}"),
                    html.P("Total Transactions")
                ]),
                html.Div([
                    html.H3(f"{data['total_fraud']:,}", style={'color': '#FF4B4B'}),
                    html.P("Fraud Cases")
                ]),
                html.Div([
                    html.H3(f"{data['fraud_percentage']:.2f}%"),
                    html.P("Fraud Percentage")
                ])
            ]
        except Exception as e:
            app.logger.error(f"Dashboard summary error: {str(e)}")
            return ["N/A", "N/A", "N/A"]

    # Add other callbacks with similar error handling...

    return dash_app

# Initialize Dash app
dash_app = init_dash_app(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
