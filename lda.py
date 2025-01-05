from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
class LDA_classification:
    def __init__(self, solver='svd', shrinkage=None):
        """
        Initialize the LDA classifier.
        """
        self.model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    def fit(self, dataset):
        """
        Fit the LDA model on the dataset.
        """
        X = np.array([i[0] for i in dataset])  # Features
        y = np.array([i[1] for i in dataset])  # Labels
        self.model.fit(X, y)

    def eval(self, dataset):
        """
        Evaluate the LDA model on the dataset.
        """
        X = np.array([i[0] for i in dataset])  # Features
        y = np.array([i[1] for i in dataset])  # Labels
        score = self.model.score(X, y)  # Accuracy
        print(f"Model accuracy: {score}")
        return score
    
