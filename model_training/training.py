from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import pandas as pd
from pathlib import Path


def train_and_evaluate_model(model_path, X_train_path, y_train_path, X_test_path, y_test_path):
    """
    Train a RandomForest model and evaluate it on the test dataset.

    Args:
        data_dir (str): Directory containing processed train/test CSV files.
        model_path (str): Path to save the trained model.

    Returns:
        None
    """
    # Load data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    # Save model
    joblib.dump(model, model_path)

# Example Usage:

tmp_path = 'model_training'
tmp_path = Path(tmp_path)

X_train_csv = tmp_path / "X_train.csv"
y_train_csv = tmp_path / "y_train.csv"
X_test_csv = tmp_path / "X_test.csv"
y_test_csv = tmp_path / "y_test.csv"

#train_and_evaluate_model('model_training/fraud_model.pkl') 
train_and_evaluate_model(
        'model_training/fraud_model.pkl',
        X_train_csv,
        y_train_csv,
        X_test_csv,
        y_test_csv,
    )