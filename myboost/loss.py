class BaseLoss:
    def __init__(self):
        pass

    def fit(self):
        pass

    def gradient(self):
        pass

    def curvature(self):
        pass


class BaseRegressorCriterion(BaseLoss):
    def __init__(self):
        pass


class MeanSquaredError(BaseRegressorCriterion):
    def __init__(self):
        pass
