from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
from .abstract import AbstractClass

class QDAModel(AbstractClass):
    def __init__(self,  reg_param=0.1, solver='lsqr', store_covariance=False):
        # Initialize with hyperparameters
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        # QDA does not use 'solver' directly in initialization, it's a valid argument in set_params
        self.model = QuadraticDiscriminantAnalysis(
                                                  reg_param=self.reg_param, 
                                                  store_covariance=self.store_covariance)

    # Fit the model
    def fit(self, X, y):
        self.model.fit(X, y)

    # Evaluate the model
    def eval(self, dataset):
        X = np.array([i[0] for i in dataset])
        y = np.array([i[1] for i in dataset])
        score = self.model.score(X, y)
        print(f"Model accuracy: {score}")
        return score

    # Predict method
    def predict(self, input):
        return self.model.predict(input)

    # REQUIRED for GridSearchCV: Get hyperparameters
    def get_params(self, deep=True):
        return {
            'reg_param': self.reg_param,
            'store_covariance': self.store_covariance
        }

    # REQUIRED for GridSearchCV: Set hyperparameters
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # Update internal model with the adjusted parameters
        self.model = QuadraticDiscriminantAnalysis(
                                                  reg_param=self.reg_param, 
                                                  store_covariance=self.store_covariance)
        return self
    
    def score(self, X, y):
        # Assuming you have a predict method
        predictions = self.predict(X)
        accuracy = (predictions == y).mean()
        return accuracy