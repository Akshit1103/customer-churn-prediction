import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the raw churn data."""
    df = pd.read_csv(filepath)
    
    # Basic cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    
    return df

def prepare_features_target(df, target_col='Churn'):
    """Splits data into features and target, and applies preprocessing."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # One-Hot Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Scale numeric features
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    scaler = StandardScaler()
    X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
    
    # Save fitted objects
    joblib.dump(le, 'models/label_encoder.pkl')
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    
    return X_processed, y_encoded, le, scaler

if __name__ == "__main__":
    df = load_and_preprocess_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X, y, le, scaler = prepare_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data preprocessing completed successfully.")