from typing import Dict, Iterable, Optional, Text, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBRFRegressor


class RatingsModel(object):
    """An interface to implement an XGBoost regression model"""

    def __init__(self, model: Union[XGBRegressor, XGBRFRegressor]):
        super(RatingsModel, self).__init__()
        self.model = model
        self.is_fitted = False

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Iterable[np.float64], **kwargs
    ):
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True

    def predict(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        assert self.is_fitted, "Fit before you predict 🙃"
        return self.model.predict(X, **kwargs)

    def get_mse(
        self, y_true: Iterable[np.float64], y_pred: Iterable[np.float64], **kwargs
    ) -> np.float64:
        return mean_squared_error(y_true, y_pred, **kwargs)

    def get_shap_values(self):
        pass

    def save_results(self, path: Text):
        """model object, mse, shap values per variable"""
        pass


class StackedModel(object):
    """An interface to implement a stacked model"""

    def __init__(self, model: Union[SGDRegressor, PassiveAggressiveRegressor]):
        super(StackedModel, self).__init__()
        self.model = model
        self.is_fitted = False
        self.coefs = []

    def fit(
        self,
        model_1: RatingsModel,
        X_1: Union[np.ndarray, pd.DataFrame],
        y_1: Iterable[np.float64],
        model_2: RatingsModel,
        X_2: Union[np.ndarray, pd.DataFrame],
        y_2: Iterable[np.float64],
        kwargs_1: Optional[Dict] = None,
        kwargs_2: Optional[Dict] = None,
        kwargs: Optional[Dict] = None,
    ):
        if kwargs_1 is not None:
            model_1.fit(X_1, y_1, **kwargs_1)
        else:
            model_1.fit(X_1, y_1)
        m1_pred = model_1.predict(X_2)

        if kwargs_2 is not None:
            model_2.fit(X_2, y_2, **kwargs_2)
        else:
            model_2.fit(X_2, y_2)
        m2_pred = model_2.predict(X_2)

        X_stacked = np.vstack((m1_pred, m2_pred)).T

        if kwargs is not None:
            if self.is_fitted:
                self.model.partial_fit(X_stacked, y_2, **kwargs)
            else:
                self.model.fit(X_stacked, y_2, **kwargs)
        else:
            if self.is_fitted:
                self.model.partial_fit(X_stacked, y_2)
            else:
                self.model.fit(X_stacked, y_2)

        self.is_fitted = True
        self.coefs.append(tuple(self.model.coef_))
        self.model_1 = model_1
        self.model_2 = model_2

    def predict(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        assert self.is_fitted, "Fit before you predict 🙃"
        m1_pred = self.model_1.predict(X)
        m2_pred = self.model_2.predict(X)
        X_stacked = np.vstack((m1_pred, m2_pred)).T
        return self.model.predict(X_stacked, **kwargs)

    def get_mse(
        self, y_true: Iterable[np.float64], y_pred: Iterable[np.float64], **kwargs
    ) -> np.float64:
        return mean_squared_error(y_true, y_pred, **kwargs)

    def get_coefs(self) -> Tuple[Tuple[np.float64]]:
        assert self.coefs, "No coefs yet 🤨 Get back to training"
        model_1_coefs, model_2_coefs = zip(*self.coefs)
        return (model_1_coefs, model_2_coefs)
