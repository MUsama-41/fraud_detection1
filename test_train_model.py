import os
from training import train_and_evaluate_model
from sklearn.ensemble import RandomForestClassifier
import joblib

def test_train_and_evaluate_model(tmp_path):
    # Create mock processed data
    X_train_csv = tmp_path / "X_train.csv"
    y_train_csv = tmp_path / "y_train.csv"
    X_test_csv = tmp_path / "X_test.csv"
    y_test_csv = tmp_path / "y_test.csv"
    
    X_train_csv.write_text("Amount,Time\n0.5,1.0\n-0.5,-1.0")
    y_train_csv.write_text("Class\n1\n0")
    X_test_csv.write_text("Amount,Time\n0.2,0.5\n-0.3,-0.7")
    y_test_csv.write_text("Class\n1\n0")

    # Define model path
    model_path = tmp_path / "fraud_model.pkl"

    # Run model training and evaluation
    train_and_evaluate_model(tmp_path, model_path)

    # Check if model file is created
    assert os.path.exists(model_path)

    # Verify model loading
    model = joblib.load(model_path)
    assert isinstance(model, RandomForestClassifier)