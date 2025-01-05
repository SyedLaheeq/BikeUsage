# train_bagging.py
from methods.bagging import Bagging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    try:
        dataset = pd.read_csv(file_path)
        X = dataset.drop('increase_stock', axis=1)
        y = dataset['increase_stock']
        return X.values, y.values
    except Exception as e:
        logging.error(f"Error in data loading: {str(e)}")
        raise

def tune_base_model(X, y, param_grid):
    """Tune hyperparameters for the base decision tree model."""
    try:
        base_model = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X, y)
        
        logging.info(f"Best parameters found: {grid_search.best_params_}")
        logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    except Exception as e:
        logging.error(f"Error in hyperparameter tuning: {str(e)}")
        raise

def evaluate_model(model, X_train, X_val, y_train, y_val):
    """Evaluate the model on training and validation sets."""
    try:
        # Training predictions
        y_train_pred = model.predict(X_train)
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        
        # Print classification reports
        logging.info("\nTraining Data Evaluation:")
        logging.info("\n" + classification_report(y_train, y_train_pred))
        
        logging.info("\nValidation Data Evaluation:")
        logging.info("\n" + classification_report(y_val, y_val_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        logging.info(f"\nCross-validation scores: {cv_scores}")
        logging.info(f"Mean CV accuracy: {cv_scores.mean():.4f}")
        
        # Calculate accuracy scores
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        logging.info(f"\nAccuracy on training data: {train_score:.4f}")
        logging.info(f"Accuracy on validation data: {val_score:.4f}")
        
        return {
            'train_score': train_score,
            'val_score': val_score,
            'cv_scores': cv_scores,
            'train_report': classification_report(y_train, y_train_pred),
            'val_report': classification_report(y_val, y_val_pred)
        }
    except Exception as e:
        logging.error(f"Error in model evaluation: {str(e)}")
        raise

def main():
    try:
        # Load dataset
        X, y = load_and_preprocess_data('training_data_fall2024.csv')
        
        # Split the dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define hyperparameter grid for base model
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],  # Removed 'auto' as it's no longer supported
            'criterion': ['gini', 'entropy']
        }
        
        # Tune base model
        logging.info("Tuning base decision tree model...")
        best_base_model, best_params = tune_base_model(X_train, y_train, param_grid)
        
        # Create and train bagging model
        logging.info("\nTraining bagging model...")
        bagging_params = {
            'estimator': best_base_model,  # Changed from 'base_estimator' to 'estimator'
            'n_estimators': 50,
            'random_state': 42
        }
        
        bagging_model = Bagging(**bagging_params)
        bagging_model.fit((X_train, y_train))
        
        # Evaluate model
        logging.info("\nEvaluating bagging model...")
        evaluation_results = evaluate_model(
            bagging_model.model,
            X_train, X_val,
            y_train, y_val
        )
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()
