from sklearn.linear_model import LogisticRegression
import numpy as np
from .abstract import AbstractClass

class LogisticRegressionModel(AbstractClass):
    def __init__(self, penalty='l2', C=1.0, solver='lbfgs'):
        # Initialize with hyperparameters
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.model = LogisticRegression(penalty=self.penalty, C=self.C, solver=self.solver)

    # Fit the model
    def fit(self, X, y):
        self.model.fit(X, y)

    # Evaluate the model - make sure score is used from the sklearn model
    def eval(self, dataset):
        X = np.array([i[0] for i in dataset])
        y = np.array([i[1] for i in dataset])
        score = self.model.score(X, y)  # This uses LogisticRegression's score method
        print(f"Model accuracy: {score}")
        return score

    # Custom Score method to ensure it's available in the wrapper
    def score(self, X, y):
        return self.model.score(X, y)  # Use the score method of the sklearn model

    # Predict method
    def predict(self, input):
        return self.model.predict(input)

    # REQUIRED for GridSearchCV: Get hyperparameters
    def get_params(self, deep=True):
        return {
            'penalty': self.penalty,
            'C': self.C,
            'solver': self.solver
        }

    # REQUIRED for GridSearchCV: Set hyperparameters
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # Update internal model
        self.model = LogisticRegression(penalty=self.penalty, C=self.C, solver=self.solver)
        return self