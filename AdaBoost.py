from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from .abstract import AbstractClass

class AdaBoost(AbstractClass):
    def __init__(self):
        self.model = AdaBoostClassifier()

    def predict(self, input):
        return self.model.predict(input)
    
    def fit(self, dataset):
        self.model.fit(
            np.array([i[0] for i in dataset]),
            np.array([i[1] for i in dataset])
            )