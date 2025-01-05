from sklearn.tree import DecisionTreeClassifier
import numpy as np
from .abstract import AbstractClass

class DecisionTree(AbstractClass):
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def predict(self, input):
        return self.model.predict(input)

    def fit(self, dataset):
        self.model.fit(
            np.array([i[0] for i in dataset]),
            np.array([i[1] for i in dataset])
            )
