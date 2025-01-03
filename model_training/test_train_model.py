import os
from pathlib import Path
from training import train_and_evaluate_model
import joblib
from sklearn.ensemble import RandomForestClassifier

def test_train_and_evaluate_model(tmp_path):
    # Create mock processed data
    tmp_path = Path(tmp_path)

    X_train_csv = tmp_path / "X_train.csv"
    y_train_csv = tmp_path / "y_train.csv"
    X_test_csv = tmp_path / "X_test.csv"
    y_test_csv = tmp_path / "y_test.csv"

    X_train_csv.write_text("Amount,Time\n0.5,1.0\n-0.5,-1.0\n0.2,0.3\n-0.3,-0.7")
    y_train_csv.write_text("Class\n1\n0\n1\n0")
    X_test_csv.write_text("Amount,Time\n0.2,0.5\n-0.3,-0.7\n0.6,0.8\n-0.4,-0.6")
    y_test_csv.write_text("Class\n1\n0\n1\n0")



    # Define model path
    model_path = 'model_training/fraud_model.pkl'

    # Run model training and evaluation with dynamically created paths
    train_and_evaluate_model(
        model_path,
        X_train_csv,
        y_train_csv,
        X_test_csv,
        y_test_csv,
    )

    # Check if model file is created
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

    # Verify model loading
    model = joblib.load(model_path)
    assert isinstance(model, RandomForestClassifier)



test_train_and_evaluate_model('model_training')