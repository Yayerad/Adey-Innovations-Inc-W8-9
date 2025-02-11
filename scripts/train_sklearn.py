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
        
        # Log model
        mlflow.sklearn.log_model(model, model_name)