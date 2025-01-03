from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score,classification_report
import pandas as pd
import joblib
from pathlib import Path


def train_and_evaluate_model(model_path, X_train_path, y_train_path, X_test_path, y_test_path):
    # Load data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    # Handle predict_proba for single-class cases
    if model.n_classes_ > 1:  # Check if there are multiple classes
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"AUC: {auc}")
    else:
        print("AUC calculation skipped due to single class in test data.")

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