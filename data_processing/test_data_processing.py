import os
import pandas as pd
from data_processing import process_data

def test_process_data(tmp_path):
    # Create sample input data
    sample_data = pd.DataFrame({
        "Amount": [100, 200, 300],
        "Time": [1, 2, 3],
        "Class": [0, 1, 0]
    })
    input_csv = tmp_path / "sample_data.csv"
    output_dir = tmp_path / "processed_data"
    output_dir.mkdir()

    # Save sample data
    sample_data.to_csv(input_csv, index=False)

    # Run data processing
    process_data(input_csv, output_dir)

    # Check if processed files exist
    assert os.path.exists(output_dir / "X_train.csv")
    assert os.path.exists(output_dir / "y_train.csv")
    assert os.path.exists(output_dir / "X_test.csv")
    assert os.path.exists(output_dir / "y_test.csv")

    # Verify data integrity
    X_train = pd.read_csv(output_dir / "X_train.csv")
    y_train = pd.read_csv(output_dir / "y_train.csv")
    assert not X_train.empty
    assert not y_train.empty
