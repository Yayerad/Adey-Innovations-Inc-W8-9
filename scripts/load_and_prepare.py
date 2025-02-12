import pandas as pd
def load_and_prepare_data(dataset_path, target_column):
    """Load data and split into features/target."""
    data = pd.read_csv(dataset_path)
    
    # Drop non-numeric columns
    non_numeric_cols = [
        'user_id', 'signup_time', 'purchase_time', 
        'device_id', 'ip_address', 'ip_address_int'
    ]
    data = data.drop(columns=non_numeric_cols, errors='ignore')
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y