import numpy

from .base import BaseRegressor
from .loss import LOSS_REG


class Node(BaseRegressor):

    loss_reg = LOSS_REG

    def __init__(self, loss: str):
        self.loss = loss

    def fit(self, X: numpy.ndarray, y: numpy.ndarray):
        pass

    def predict(self, X: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        pass
