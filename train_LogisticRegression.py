from methods.LogisticRegression import LogisticRegressionModel as Model
from data import BikeDemandDataset_v01 as BikeDemandDataset
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Hyperparameter tuning function
def tune_hyperparameters(model, X_train, y_train, param_grid):
    # GridSearchCV setup
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        verbose=1
    )
    # Perform grid search
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Return best model
    return grid_search.best_estimator_, grid_search.best_params_

if __name__ == '__main__':
    # Load dataset
    dataset = BikeDemandDataset('test_data_fall2024.csv')

    # Prepare features (X) and labels (y)
    X = np.array([i[0] for i in dataset])
    y = np.array([i[1] for i in dataset])

    # Split data into 80:20 ratio
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1, random_state=42)

    # Initialize Logistic Regression model
    logistic_model = Model()

    # Hyperparameter grid
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }

    print("Tuning hyperparameters...")
    # Tune hyperparameters
    best_model, best_params = tune_hyperparameters(logistic_model, X_train, y_train, param_grid)

    # Cross-validation on the best model
    print("Performing cross-validation...")
    cv_scores = cross_val_score(best_model.model, X_train, y_train, cv=5, scoring='accuracy')
    print("Cross-validation scores:", cv_scores)
    print("Mean CV Accuracy:", np.mean(cv_scores))

    # Train the best model on training data
    print("Training best model...")
    best_model.fit(X_train, y_train)

    # Evaluate performance on training data
    print('Training Evaluation:')
    y_train_pred = best_model.predict(X_train)
    print(classification_report(y_train, y_train_pred))
    print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred) * 100:.2f}%")

    # Evaluate performance on validation data
    print('Validation Evaluation:')
    y_val_pred = best_model.predict(X_val)
    print(classification_report(y_val, y_val_pred))
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred) * 100:.2f}%")
