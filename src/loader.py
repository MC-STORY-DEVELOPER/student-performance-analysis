import pandas as pd
import os

def load_data(filepath):
    """
    Loads the dataset from the specified filepath.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
    return df

def validate_data(df):
    """
    Performs basic validation on the dataframe.
    """
    required_columns = ['Exam_Score'] # Target variable
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError("Dataframe is empty.")
    
    return True
