import pandas as pd
from data_processing import process_data  # Import your function
import os
from pathlib import Path

def test_process_data(tmp_path):
    # Create a sample dataset with all necessary columns
    sample_data = pd.DataFrame({
        "Time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "V1": [1.0, -1.2, 0.5, 0.8, -0.6, 1.2, -0.3, 0.4, -1.0, 0.7],
        "V2": [0.5, -0.8, 1.0, 0.3, -0.2, 0.6, -0.4, 1.2, -0.9, 0.8],
        "V3": [1.1, 0.2, -0.5, 0.4, 0.0, -0.7, 0.3, -0.8, 1.4, 0.6],
        "V4": [0.3, -0.1, 0.8, 0.2, -0.5, 0.6, -0.3, 0.1, -0.7, 0.5],
        "V5": [0.2, 0.1, -0.3, 0.6, -0.8, 0.9, -0.4, 0.2, -0.6, 0.7],
        "V6": [0.4, -0.6, 0.7, -0.2, 0.3, -0.5, 0.8, -0.1, 0.9, -0.4],
        "V7": [0.1, 0.0, -0.2, 0.5, -0.3, 0.6, -0.7, 0.2, -0.4, 0.8],
        "V8": [1.5, -0.9, 0.3, -0.8, 0.7, -0.6, 0.4, -0.1, 0.9, -0.3],
        "V9": [0.8, -0.2, 0.6, -0.1, 0.2, -0.4, 0.3, 0.5, -0.6, 0.7],
        "V10": [-0.3, 0.4, 0.5, -0.8, 0.9, -0.1, 0.2, -0.5, 0.6, -0.7],
        "V11": [0.7, -0.4, 0.1, -0.9, 0.3, 0.6, -0.8, 0.4, 0.2, -0.3],
        "V12": [-0.2, 0.6, -0.1, 0.4, -0.3, 0.8, -0.6, 0.2, -0.4, 0.5],
        "V13": [0.9, -1.0, 0.4, 0.3, -0.7, 0.6, -0.2, 0.5, -0.3, 0.8],
        "V14": [0.3, 0.1, -0.7, 0.2, -0.4, 0.9, -0.5, 0.8, -0.6, 0.7],
        "V15": [-0.5, 0.8, 0.2, -0.6, 0.4, -0.3, 0.7, -0.2, 0.6, -0.4],
        "V16": [0.6, -0.3, 0.0, 0.8, -0.9, 0.5, -0.7, 0.1, -0.4, 0.3],
        "V17": [1.1, -0.4, 0.5, 0.9, -0.2, 0.3, -0.8, 0.7, -0.6, 0.2],
        "V18": [-0.6, 0.2, 0.8, -0.3, 0.4, -0.1, 0.9, -0.7, 0.6, -0.2],
        "V19": [0.4, -0.1, -0.3, 0.5, -0.7, 0.6, -0.4, 0.2, 0.9, -0.6],
        "V20": [0.0, 0.7, -0.5, 0.8, -0.6, 0.2, -0.3, 0.4, -0.8, 0.9],
        "V21": [0.5, -0.2, 0.6, -0.3, 0.9, -0.1, 0.4, -0.7, 0.3, 0.8],
        "V22": [1.2, -0.4, 0.3, -0.6, 0.8, -0.5, 0.7, -0.9, 0.6, -0.2],
        "V23": [0.3, 0.8, -0.1, 0.2, -0.3, 0.9, -0.4, 0.6, -0.5, 0.7],
        "V24": [0.6, -0.5, 0.7, -0.8, 0.4, -0.2, 0.8, -0.6, 0.3, 0.9],
        "V25": [-0.1, 0.2, 0.4, -0.7, 0.6, -0.3, 0.5, -0.8, 0.9, -0.4],
        "V26": [0.7, -0.3, 0.0, -0.6, 0.9, -0.4, 0.6, -0.2, 0.5, 0.8],
        "V27": [1.3, -0.8, 0.9, -0.2, 0.3, 0.7, -0.5, 0.4, 0.8, -0.6],
        "V28": [0.0, 0.1, -0.2, 0.6, -0.7, 0.9, -0.4, 0.3, 0.5, -0.3],
        "Amount": [100.5, 200.0, 150.0, 300.0, 250.0, 400.0, 500.0, 350.0, 450.0, 100.0],
        "Class": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]  # Target variable
    })
    
    tmp_path = Path(tmp_path)
    # Save sample data to a temporary CSV file
    input_csv = tmp_path / "sample_data.csv"
    print("input csv :", input_csv)
    sample_data.to_csv(input_csv, index=False)

    # Create a temporary output directory
    output_dir = tmp_path / "processed_data"
    output_dir.mkdir(exist_ok=True)

    # Call the process_data function
    process_data(input_csv, output_dir)

    # Check if output files exist
    assert (output_dir / "X_train.csv").exists(), "X_train.csv is missing"
    assert (output_dir / "y_train.csv").exists(), "y_train.csv is missing"
    assert (output_dir / "X_test.csv").exists(), "X_test.csv is missing"
    assert (output_dir / "y_test.csv").exists(), "y_test.csv is missing"

# Run the function manually
test_process_data("data_processing")