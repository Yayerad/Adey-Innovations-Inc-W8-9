import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
from pathlib import Path
import joblib

def train_sklearn_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train and evaluate scikit-learn models."""
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Log metrics
        mlflow.log_metrics({
            "roc_auc": roc_auc_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        })
        
        # ======== NEW CODE TO SAVE MODELS ========
        # Create directory if it doesn't exist
        model_dir = Path(f"models/{model_name}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model to disk (joblib format)
        joblib.dump(model, model_dir / f"{model_name}_model.pkl")
        # ======== END NEW CODE ========
        
        # Log model to MLflow (existing functionality)
        mlflow.sklearn.log_model(model, model_name)