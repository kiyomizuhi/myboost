import numpy as np

from .base import BaseRegressor
from .criterion import MSE


class Leaf(BaseRegressor):
    def __init__(self):
        pass

    def fit(self, X: np.array, y: np.array):
        return self

    def predict(self, X: np.array, y: np.array) -> np.array:
        pass
