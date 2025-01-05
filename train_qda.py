import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report
from data import BikeDemandDataset_v01 as BikeDemandDataset
from methods.qda import QDAModel  # Assuming the QDAModel class is defined in qda_model.py


def tune_hyperparameters(model_class, X_train, y_train, param_grid):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    # Create an instance of the model class
    model = model_class()

    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        error_score='raise'  # Raise errors to debug easily
    )
    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Return the best model and hyperparameters
    return grid_search.best_estimator_, grid_search.best_params_


def perform_cross_validation(model, X, y, cv=5):
    """
    Perform k-fold cross-validation.
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores) * 100:.2f}%")
    return cv_scores.mean()


def evaluate_model(model, X, y, dataset_type="Training"):
    """
    Evaluate model using accuracy, F1-score, precision, and recall.
    """
    predictions = model.predict(X)

    # Print classification metrics
    print(f"\n{dataset_type} Evaluation:")
    print(classification_report(y, predictions))  # Precision, Recall, F1-score, and Support


if __name__ == '__main__':
    # Load dataset
    dataset = BikeDemandDataset('training_data_fall2024.csv')

    # Extract features and labels
    X = np.array([i[0] for i in dataset])
    y = np.array([i[1] for i in dataset])

    # Split data into 80% training and 20% validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameter grid for QDAModel
    param_grid = {
        'priors': [None],  # None uses class frequencies as prior probabilities
        'reg_param': [0.0, 0.1, 0.5, 1.0],  # Regularization parameter
        'store_covariance': [False, True]  # Whether to store covariance matrices
    }

    # Hyperparameter tuning
    print("Tuning hyperparameters...")
    best_model, best_params = tune_hyperparameters(QDAModel, X_train, y_train, param_grid)
    print(f"Best hyperparameters: {best_params}")

    # Perform cross-validation
    print("Performing Cross-Validation...")
    mean_cv_score = perform_cross_validation(best_model, X_train, y_train, cv=5)

    # Evaluate performance on training data
    print("\nEvaluating on Training Data:")
    evaluate_model(best_model, X_train, y_train, dataset_type="Training")

    # Evaluate performance on validation data
    print("\nEvaluating on Validation Data:")
    evaluate_model(best_model, X_val, y_val, dataset_type="Validation")

    # Training accuracy
    train_score = best_model.score(X_train, y_train)
    print(f"Training Accuracy: {train_score * 100:.2f}%")

    # Validation accuracy
    val_score = best_model.score(X_val, y_val)
    print(f"Validation Accuracy: {val_score * 100:.2f}%")
