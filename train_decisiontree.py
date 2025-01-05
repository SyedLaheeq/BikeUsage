import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from data import BikeDemandDataset_v01 as BikeDemandDataset
from methods.DecisionTree import DecisionTree as Model  # Assuming DecisionTree is implemented correctly


def tune_hyperparameters(model_class, X_train, y_train, param_grid):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    # Instantiate the model
    model = model_class()

    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=model.model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1
    )

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Return the best model and hyperparameters
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X, y, dataset_type="Training"):
    """
    Evaluate the model using accuracy, F1-score, precision, and recall.
    """
    predictions = model.predict(X)

    # Print classification metrics
    print(f"\n{dataset_type} Evaluation:")
    print(classification_report(y, predictions))  # Includes Precision, Recall, and F1-score


if __name__ == '__main__':
    # Load dataset
    dataset = BikeDemandDataset('training_data_fall2024.csv')

    # Extract features and labels
    X = np.array([i[0] for i in dataset])
    y = np.array([i[1] for i in dataset])

    # Split data into 80% training and 20% validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameter grid for Decision Tree
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }

    # Hyperparameter tuning
    print("Tuning hyperparameters...")
    best_model, best_params = tune_hyperparameters(Model, X_train, y_train, param_grid)
    print(f"Best parameters: {best_params}")

    # Perform cross-validation on training data
    print("\nPerforming Cross-Validation...")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")

    # Evaluate performance on training data
    evaluate_model(best_model, X_train, y_train, dataset_type="Training")

    # Evaluate performance on validation data
    evaluate_model(best_model, X_val, y_val, dataset_type="Validation")

    # Training accuracy
    train_score = best_model.score(X_train, y_train)
    print(f"Training Accuracy: {train_score * 100:.2f}%")

    # Validation accuracy
    val_score = best_model.score(X_val, y_val)
    print(f"Validation Accuracy: {val_score * 100:.2f}%")
