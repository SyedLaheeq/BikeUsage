from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd


class RandomForestModel:
    def __init__(self):
        # Initialize the model and hyperparameter grid
        self.model = RandomForestClassifier(random_state=42)
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],  # Removed 'auto' as it's deprecated in sklearn 1.2+
            'bootstrap': [True, False]
        }

    def fit(self, X_train, y_train):
        """Fit the model to the training data with hyperparameter tuning."""
        # Perform GridSearchCV to find the best hyperparameters
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)
        
        # Set the best model after hyperparameter tuning
        self.model = grid_search.best_estimator_
        print(f"Best Hyperparameters: {grid_search.best_params_}")

    def eval(self, X, y, dataset_type="Dataset"):
        """Evaluate the model with accuracy, precision, recall, and F1-score."""
        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"\n{dataset_type} Accuracy: {accuracy:.4f}")
        print(f"{dataset_type} Classification Report:")
        print(classification_report(y, predictions))  # Includes precision, recall, and F1-score

    def cross_validate(self, X, y):
        """Perform k-fold cross-validation on the model."""
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")

    def train_and_evaluate(self, data_file):
        """Train the model and evaluate it using 80:20 split for training and validation datasets."""
        print("Loading data...")
        data = pd.read_csv(data_file)

        # Replace 'increase_stock' with the actual target column name
        X = data.drop('increase_stock', axis=1)  
        y = data['increase_stock']

        # Split the data into 80% training and 20% validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        print("\nTraining the model...")
        self.fit(X_train, y_train)

        # Perform cross-validation
        print("\nPerforming cross-validation...")
        self.cross_validate(X_train, y_train)

        # Evaluate on training data
        print("\nEvaluating on Training Data:")
        self.eval(X_train, y_train, dataset_type="Training")

        # Evaluate on validation data
        print("\nEvaluating on Validation Data:")
        self.eval(X_val, y_val, dataset_type="Validation")


if __name__ == '__main__':
    # Initialize and run the model
    model = RandomForestModel()
    model.train_and_evaluate('training_data_fall2024.csv')  # Replace with actual dataset path
