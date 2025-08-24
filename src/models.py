import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyModel:
    def __init__(self):
        self.model = IsolationForest()

    def train(self, X, epochs=1):
        self.model.fit(X)
        print("Model trained.")

    def evaluate(self, X):
        scores = -self.model.decision_function(X)
        print("Evaluation complete. Sample scores:", scores[:5])
        return scores
