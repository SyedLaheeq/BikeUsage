from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from methods.XGBoost import XGBoostModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    try:
        dataset = pd.read_csv(file_path)
        X = dataset.drop('increase_stock', axis=1)
        y = dataset['increase_stock']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Encode categorical variables if any
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        return X, y
    except Exception as e:
        logging.error(f"Error in data loading: {str(e)}")
        raise

def main():
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data('training_data_fall2024.csv')
        
        # Encode target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Initialize model
        xgb_model = XGBoostModel()
        
        # Train model
        logging.info("Training XGBoost model...")
        xgb_model.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate initial model
        logging.info("\nInitial Model Performance:")
        logging.info("Training Metrics:")
        train_metrics = xgb_model.evaluate(X_train, y_train)
        
        logging.info("\nValidation Metrics:")
        val_metrics = xgb_model.evaluate(X_val, y_val)
        
        # Perform hyperparameter tuning
        logging.info("\nPerforming hyperparameter tuning...")
        best_params = xgb_model.tune_hyperparameters(X_train, y_train)
        
        # Evaluate tuned model
        logging.info("\nTuned Model Performance:")
        logging.info("Validation Metrics After Tuning:")
        final_metrics = xgb_model.evaluate(X_val, y_val)
        
        # Feature importance analysis
        xgb_model.plot_feature_importance(X.columns)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()