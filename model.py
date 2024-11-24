import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_dataset(filepath):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(filepath)
        print("Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {filepath}. Please check the path.")


def analyze_dataset(data, target_column):
    """Perform initial analysis on the dataset."""
    print("\nDataset Overview:")
    print(data.head())

    print("\nDataset Information:")
    print(data.info())

    print("\nClass Distribution:")
    if target_column in data.columns:
        print(data[target_column].value_counts())
    else:
        raise ValueError(f"The target column '{target_column}' is missing.")

    print("\nMissing Values:")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])

def preprocess_data(data, target_column, categorical_features, numeric_features):
    """Preprocess the data, including feature encoding, splitting, and scaling."""
    encoding_maps = {}

    # Frequency encode categorical features
    for col in categorical_features:
        encoding_maps[col] = data[col].value_counts(normalize=True).to_dict()
        data[col] = data[col].map(encoding_maps[col])

    # Separate features and target
    X = data[categorical_features + numeric_features]
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Standardize the numeric features
    scaler = StandardScaler()
    X_train_resampled[numeric_features] = scaler.fit_transform(X_train_resampled[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    return X_train_resampled, X_test, y_train_resampled, y_test, scaler, X_train.columns, encoding_maps


def train_model(X_train, y_train):
    """Train the Logistic Regression model."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    print("\nModel training completed.")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

def predict_transaction(model, scaler, numeric_features, encoding_maps):
    """Interactively predict whether a transaction is fraudulent."""
    print("\nEnter transaction details for prediction:")

    try:
        # Input fields
        amt = float(input("Amount: "))
        city = input("City: ").strip()
        merchant = input("Merchant Name: ").strip()
        category = input("Category: ").strip()
    except ValueError:
        print("Invalid input. Please provide numeric values for 'Amount' and valid strings for other fields.")
        return

    # Create input dictionary
    input_data = {
        'amt': amt,
        'city': city,
        'merchant': merchant,
        'category': category
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features using precomputed frequency maps
    for col, mapping in encoding_maps.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].map(mapping).fillna(0)

    # Scale numeric features
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    # Ensure all features match the model's input
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)
    print("\nPrediction:", "Fraud" if prediction[0] == 1 else "Non-Fraud")

if __name__ == "__main__":
    # Filepath to dataset
    dataset_path = 'dataset.csv'
    target_column = 'is_fraud'
    categorical_features = ['city', 'merchant', 'category']
    numeric_features = ['amt']

    # Load and analyze dataset
    data = load_dataset(dataset_path)
    analyze_dataset(data, target_column)

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_columns, encoding_maps = preprocess_data(
    data, target_column, categorical_features, numeric_features)


    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Predict transactions in a loop
    while True:
        predict_transaction(model, scaler, numeric_features, encoding_maps)
        cont = input("\nDo you want to predict another transaction? (yes/no): ").strip().lower()
        if cont != 'yes':
            break

