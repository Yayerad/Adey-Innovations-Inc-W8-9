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
        
        # Log model
        mlflow.tensorflow.log_model(model, model_name)