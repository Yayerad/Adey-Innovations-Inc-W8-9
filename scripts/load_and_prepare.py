def load_and_prepare_data(dataset_path, target_column):
    """Load data and split into features/target."""
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y