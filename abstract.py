import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

class AbstractClass:    
    def eval(self, dataset):
        input = np.array([i[0] for i in dataset])
        grt = np.array([i[1] for i in dataset])
        output = self.predict(input)

        print('Accuracy:', accuracy_score(grt, output))
        print('Precision:', precision_score(grt, output, average='macro'))
        print('Recall:', recall_score(grt, output, average='macro'))
        print('Report:', classification_report(grt, output))
