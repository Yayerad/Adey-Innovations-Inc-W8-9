# scripts/explain.py
import os
import sys
import shap
import lime
import mlflow
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

def load_data(dataset_path, target_col):
    """Load and preprocess data without external dependencies"""
    df = pd.read_csv(dataset_path)
    
    # Basic preprocessing
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Convert bool columns to int
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)
    
    return X, y

def explain_model(model_path, dataset_path, target_col, model_type):
    """Explain a single model using SHAP and LIME"""
    # Get model name from path
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_dir)
    
    # Load data
    X, y = load_data(dataset_path, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Scale numerical features
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled = scaler.transform(X_test[numerical_cols])
    
    # Load model
    if model_type == "sklearn":
        model = joblib.load(model_path)
    elif model_type == "keras":
        model = load_model(model_path)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"Explain_{model_name}"):
        # SHAP Explanation
        explain_shap(model, X_train_scaled, X_test_scaled, model_type)
        
        # LIME Explanation
        explain_lime(model, X_train_scaled, X_test_scaled, X.columns)

def explain_shap(model, X_train, X_test, model_type):
    """Generate SHAP explanations"""
    try:
        # Sample 100 instances for faster computation
        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        test_sample = X_test[:100]
        
        if model_type == "sklearn":
            # Ensure data is in the correct format
            if isinstance(background, pd.DataFrame):
                background = background.values
            if isinstance(test_sample, pd.DataFrame):
                test_sample = test_sample.values
            
            explainer = shap.TreeExplainer(model, background, check_additivity=False)
            shap_values = explainer.shap_values(test_sample)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
            
        elif model_type == "keras":
            explainer = shap.DeepExplainer(model, background.astype(np.float32))
            shap_values = explainer.shap_values(test_sample.astype(np.float32))
        
        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, test_sample, show=False)
        plt.title("SHAP Summary Plot")
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")
        plt.close()
    except Exception as e:
        print(f"SHAP failed: {str(e)}")
        
def explain_lime(model, X_train, X_test, feature_names):
    """Generate LIME explanations"""
    try:
        explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=['Legit', 'Fraud'],
            mode='classification'
        )
        
        # Explain first test instance
        def predict_fn(x):
            if hasattr(model, 'predict_proba'):  # sklearn models
                return model.predict_proba(x)
            return model.predict(x.astype(np.float32))  # keras models
        
        exp = explainer.explain_instance(
            X_test[0],
            predict_fn,
            num_features=10
        )
        
        # Save explanation
        fig = exp.as_pyplot_figure()
        fig.suptitle("LIME Explanation")
        fig.savefig("lime_explanation.png")
        mlflow.log_artifact("lime_explanation.png")
        plt.close(fig)
    except Exception as e:
        print(f"LIME failed: {str(e)}")

if __name__ == "__main__":
    # Define paths relative to the parent directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Example usage - adjust paths as needed
    models_to_explain = [
        (os.path.join(base_dir, "notebooks", "models", "RandomForest", "RandomForest_model.pkl"),
         os.path.join(base_dir, "src", "data", "fraud_data_processed.csv"), "class", "sklearn"),
         
        (os.path.join(base_dir, "notebooks", "models", "LSTM", "LSTM_model.keras"),
         os.path.join(base_dir, "src", "data", "fraud_data_processed.csv"), "class", "keras")
    ]
    
    for model_path, data_path, target, m_type in models_to_explain:
        explain_model(model_path, data_path, target, m_type)