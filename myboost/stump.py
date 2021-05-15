import numpy as np

from .base import BaseRegressor
from .criterion import CRITERION_REG
from .leaf import Leaf


class Stump(BaseRegressor):

    criterion_reg = CRITERION_REG

    def __init__(self, criterion: str, leaf: Leaf):
        self.criterion = criterion
        self.leaf = leaf

    def fit(self, X: np.array, y: np.array):
        return self

    def predict(self, X: np.array, y: np.array) -> np.array:
        pass
