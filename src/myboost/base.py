import abc
import inspect

import numpy


class BaseEstimatorMixin:
    """Base class for all estimators in myboost."""

    def __repr__(self):
        NotImplementedError()

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # inspect the constructor arguments to find the model parameters
        init_signature = inspect.signature(cls.__init__)
        # consider the constructor parameters except for 'self'
        names = []
        for name, arg in init_signature.parameters.items():
            if name == "self":
                continue
            elif arg.kind != arg.POSITIONAL_OR_KEYWORD:
                raise RuntimeError("specifiy all the constructor arguments")
            names.append(name)
        return names

    def get_params(self):
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params):
        NotImplementedError()

    def _check_n_features(self, X, reset):
        NotImplementedError()

    def _validate_data(self, X, y, reset=True):
        NotImplementedError()


class RegressorMixin:
    """Mixin class for all regression estimators in myboost"""

    _estimator_type = "regressor"

    def score(self, X: numpy.array, y: numpy.array):
        from .metrics import mse

        y_pred = self.predict(X)
        return mse(y, y_pred)


class BaseRegressor(RegressorMixin, BaseEstimatorMixin, metaclass=abc.ABCMeta):
    @abs.abstractmethod
    def fit(self):
        pass

    @abs.abstractmethod
    def predict(self):
        pass
