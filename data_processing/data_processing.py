import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def process_data(input_csv, output_dir):
    """
    Preprocess the input dataset and save processed train/test splits.

    Args:
        input_csv (str): Path to the input CSV file.
        output_dir (str): Directory to save the processed files.

    Returns:
        None
    """
    # Load dataset
    data = pd.read_csv(input_csv)

    # Normalize features
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

    # Split data
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Handle class imbalance
    #smote = SMOTE(random_state=42)
    #X_train, y_train = smote.fit_resample(X_train, y_train)

    # Save processed data
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

# Example Usage:
# process_data('creditcard.csv', './processed_data')
input_csv = "data_processing/creditcard.csv"
output_dir = "model_training"
process_data(input_csv, output_dir)
print("x")