import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import os
from pathlib import Path

def train_keras_model(model, model_name, X_train, y_train, X_test, y_test, epochs=10):
    """Train and evaluate Keras models."""
    with mlflow.start_run(run_name=model_name):
        early_stop = EarlyStopping(monitor='val_loss', patience=3)
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=64,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        
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
        
        # Save model to disk (TensorFlow SavedModel format)
        model.save(model_dir / f"{model_name}_model.keras")
        
        # ======== END NEW CODE ========
        
        # Log model to MLflow (existing functionality)
        mlflow.tensorflow.log_model(model, model_name)