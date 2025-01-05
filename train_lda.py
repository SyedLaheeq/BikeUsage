from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np


class LDAClassification:
    def __init__(self, solver='svd', shrinkage=None, n_components=10):
        """
        Initialize the LDA classifier with PCA preprocessing.
        """
        self.solver = solver
        self.shrinkage = shrinkage
        self.n_components = n_components

        # Define the pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Normalize features
            ('pca', PCA(n_components=n_components)),  # Dimensionality reduction
            ('lda', LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage))  # LDA classifier
        ])

    def fit(self, X, y):
        """
        Fit the LDA model with PCA preprocessing.
        """
        self.pipeline.fit(X, y)

    def eval(self, X, y):
        """
        Evaluate the LDA model using precision, recall, F1-score, and accuracy.
        """
        predictions = self.pipeline.predict(X)
        print(classification_report(y, predictions))  # Prints Precision, Recall, F1-score, and Support
        return predictions

    def cross_validate(self, X, y, cv=5):
        """
        Perform k-fold cross-validation.
        """
        scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='accuracy')
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV accuracy: {np.mean(scores):.2f}")
        return scores

    def tune_hyperparameters(self, X, y, param_grid, cv=5):
        """
        Tune hyperparameters using GridSearchCV.
        """
        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            verbose=1
        )
        grid_search.fit(X, y)

        # Update pipeline with best parameters
        self.pipeline = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)
        print("Best CV Accuracy:", grid_search.best_score_)

        return grid_search.best_params_


# Example usage
if __name__ == '__main__':
    # Example: Load dataset
    from data import BikeDemandDataset_v01 as BikeDemandDataset

    # Load data
    dataset = BikeDemandDataset('training_data_fall2024.csv')

    # Extract features and labels
    X = np.array([i[0] for i in dataset])
    y = np.array([i[1] for i in dataset])

    # Split dataset into 80:20 ratio
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize LDA classifier
    lda_model = LDAClassification()

    # Hyperparameter grid
    param_grid = {
        'pca__n_components': [5, 10, 15],
        'lda__solver': ['lsqr', 'eigen'],
        'lda__shrinkage': ['auto', None]
    }

    # Hyperparameter tuning
    print("Tuning hyperparameters...")
    best_params = lda_model.tune_hyperparameters(X_train, y_train, param_grid)

    # Training with best parameters
    print("Training the best model...")
    lda_model.fit(X_train, y_train)

    # Evaluate on training data
    print("Training Evaluation:")
    lda_model.eval(X_train, y_train)

    # Evaluate on validation data
    print("Validation Evaluation:")
    lda_model.eval(X_val, y_val)

    # Cross-validation
    print("Performing Cross-Validation:")
    lda_model.cross_validate(X_train, y_train, cv=5)
