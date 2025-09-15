import joblib
import pandas as pd
import numpy as np

def predict_churn(customer_data, model_path='models/best_churn_model.pkl', 
                  scaler_path='models/feature_scaler.pkl', le_path='models/label_encoder.pkl'):
    """
    Loads a trained model and makes a prediction on new customer data.
    
    Args:
        customer_data (dict or DataFrame): New customer data to predict.
        
    Returns:
        prediction (str): 'Churn' or 'No Churn'
        probability (float): Confidence of the prediction
    """
    # Load the trained model, scaler, and label encoder
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([customer_data])
    
    # Preprocess the input identical to training
    categorical_cols = input_df.select_dtypes(include=['object']).columns.tolist()
    input_processed = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Ensure the input has all the columns the model was trained on
    # This is a simplified version. In production, you need to align columns exactly.
    train_columns = model.feature_names_in_
    for col in train_columns:
        if col not in input_processed.columns:
            input_processed[col] = 0 # Add missing columns with 0
    
    # Reorder columns to match training
    input_processed = input_processed[train_columns]
    
    # Scale the numeric features (assuming the same numeric cols as before)
    numeric_cols = [col for col in input_processed.columns if col in ['tenure', 'MonthlyCharges', 'TotalCharges']]
    if numeric_cols:
        input_processed[numeric_cols] = scaler.transform(input_processed[numeric_cols])
    
    # Make prediction
    prediction_proba = model.predict_proba(input_processed)[0]
    prediction = model.predict(input_processed)[0]
    
    # Decode the prediction
    churn_label = le.inverse_transform([prediction])[0]
    churn_probability = prediction_proba[1] # Probability of class 'Yes'
    
    return churn_label, churn_probability

# Example usage
if __name__ == "__main__":
    # Example new customer data (must have all original features)
    new_customer = {
        'tenure': 12,
        'MonthlyCharges': 59.9,
        'TotalCharges': 719.0,
        'PhoneService': 'Yes',
        'Contract': 'One year',
        # ... include all other necessary features
    }
    
    label, proba = predict_churn(new_customer)
    print(f"Prediction: {label} | Confidence: {proba:.2f}")