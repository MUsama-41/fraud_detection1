from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import pandas as pd

def train_and_evaluate_model(model_path):
    """
    Train a RandomForest model and evaluate it on the test dataset.

    Args:
        data_dir (str): Directory containing processed train/test CSV files.
        model_path (str): Path to save the trained model.

    Returns:
        None
    """
    # Load data
    X_train = pd.read_csv(f'X_train.csv')
    y_train = pd.read_csv(f'y_train.csv').squeeze()
    X_test = pd.read_csv(f'X_test.csv')
    y_test = pd.read_csv(f'y_test.csv').squeeze()

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
train_and_evaluate_model('model_training/fraud_model.pkl') 