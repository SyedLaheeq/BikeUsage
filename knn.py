from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNN_classification:
    def __init__(self, n_neighbors=3, weights='uniform', metric='euclidean'):
        """
        Initialize the KNN classifier.
        """
        # Create the KNeighborsClassifier model
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

    def fit(self, dataset):
        """
        Fit the KNN model on the dataset.
        """
        X = np.array([i[0] for i in dataset])  # Features
        y = np.array([i[1] for i in dataset])  # Labels
        self.model.fit(X, y)

    def eval(self, dataset):
        """
        Evaluate the KNN model on the dataset.
        """
        X = np.array([i[0] for i in dataset])  # Features
        y = np.array([i[1] for i in dataset])  # Labels
        score = self.model.score(X, y)  # Return accuracy score
        print(f"Model accuracy: {score}")
        return score
