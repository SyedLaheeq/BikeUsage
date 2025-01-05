from data import BikeDemandDataset_v01 as BikeDemandDataset
from methods.AdaBoost import AdaBoost
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report
import numpy as np

def tune_hyperparameters(model_class, dataset, param_grid):
    # Instantiate the model
    model = model_class()  # Use the custom AdaBoost class

    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=model.model,  # Use the internal AdaBoostClassifier
        param_grid=param_grid,  # Hyperparameter grid
        cv=5,                   # 5-fold cross-validation
        scoring='accuracy'      # Evaluation metric
    )

    # Fit the grid search to the data
    grid_search.fit(
        np.array([i[0] for i in dataset]),  # Features
        np.array([i[1] for i in dataset])   # Labels
    )

    # Return the best model and parameters
    return grid_search.best_estimator_, grid_search.best_params_

def cross_validate_model(model, X, y):
    """Perform k-fold cross-validation on the model"""
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

if __name__ == '__main__':
    # Load datasets
    dataset = BikeDemandDataset('training_data_fall2024.csv')
    eval_dataset = BikeDemandDataset('training_data_fall2024.csv', data_type='validation')
    
    # Define the hyperparameter grid for AdaBoost
    param_grid = {
        'n_estimators': [50, 100, 200],        # Number of estimators
        'learning_rate': [0.01, 0.1, 0.2],     # Learning rate
        'algorithm': ['SAMME', 'SAMME.R'],      # Algorithm type
    }

    # Perform hyperparameter tuning
    best_model, best_params = tune_hyperparameters(AdaBoost, dataset, param_grid)
    
    # Print the best parameters
    print(f"Best parameters: {best_params}")

    # Prepare the training and validation data
    X = np.array([i[0] for i in dataset])
    y = np.array([i[1] for i in dataset])
    
    # Split the dataset into 80% training and 20% validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform cross-validation on the best model
    print("\nPerforming cross-validation...")
    cross_validate_model(best_model, X_train, y_train)
    
    # Evaluate the best model on training data
    print("\nEvaluating on training data:")
    best_model.fit(X_train, y_train)
    y_train_pred = best_model.predict(X_train)
    print(f"Score on training data: {best_model.score(X_train, y_train)}")
    
    # Print classification report (Precision, Recall, F1-score)
    print("Training Data Evaluation:")
    print(classification_report(y_train, y_train_pred))
    
    # Evaluate the best model on validation data
    print("\nEvaluating on validation data:")
    y_val_pred = best_model.predict(X_val)
    print(f"Score on validation data: {best_model.score(X_val, y_val)}")
    
    # Print classification report (Precision, Recall, F1-score)
    print("Validation Data Evaluation:")
    print(classification_report(y_val, y_val_pred))
