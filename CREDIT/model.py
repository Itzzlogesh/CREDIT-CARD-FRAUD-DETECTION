import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import os

def train_model(csv_file, target_column='Class', test_size=0.2, n_estimators=100, random_state=42):
    """
    Train a fraud detection model on any CSV file
    
    Parameters:
    - csv_file: Path to the CSV file
    - target_column: Name of the column containing the target variable (fraud/not fraud)
    - test_size: Proportion of data to use for testing
    - n_estimators: Number of trees in the Random Forest
    - random_state: Random seed for reproducibility
    """
    print(f"Loading data from {csv_file}...")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Load dataset
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")
    
    # Check if target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the CSV file. Available columns: {', '.join(data.columns)}")
    
    print(f"Data loaded successfully. Shape: {data.shape}")
    
    # Drop rows where target column is NaN
    data = data.dropna(subset=[target_column])
    
    # Prepare features (keep all columns except target column)
    feature_columns = [col for col in data.columns if col != target_column]
    X = data[feature_columns]
    y = data[target_column]
    
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Handle missing values in features
    print("Handling missing values...")
    X = X.fillna(X.mean())
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split data
    print(f"Splitting data with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    
    # Train model
    print(f"Training Random Forest model with {n_estimators} trees...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Model accuracy - Training: {train_score:.4f}, Testing: {test_score:.4f}")
    
    # Save model, scaler, and feature columns
    print("Saving model and preprocessing objects...")
    joblib.dump(model, 'fraud_detector.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_columns, 'feature_columns.pkl')
    
    print("✅ Model saved as fraud_detector.pkl")
    print(f"Features required: {', '.join(feature_columns)}")
    
    return model, scaler, feature_columns

if __name__ == '__main__':
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Train a fraud detection model on any CSV file')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--target', '-t', default='Class', help='Name of the target column (default: Class)')
    parser.add_argument('--test-size', '-s', type=float, default=0.2, help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--n-estimators', '-n', type=int, default=100, help='Number of trees in the Random Forest (default: 100)')
    parser.add_argument('--random-state', '-r', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Train model with provided arguments
    train_model(
        args.csv_file,
        target_column=args.target,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
