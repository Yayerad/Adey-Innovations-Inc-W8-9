def scale_features(X_train, X_test, numerical_features):
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    return X_train_scaled, X_test_scaled