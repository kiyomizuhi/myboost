LOSS_REG = ["MSE"]


class BaseLoss:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def gradient(self):
        pass

    def curvature(self):
        pass


class BaseRegressorLoss(BaseLoss):
    def __init__(self):
        pass


class MeanSquaredError(BaseRegressorLoss):
    def __init__(self):
        pass
