from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import pandas as pd

class ToLassoORnotToLasso(BaseEstimator, ClassifierMixin):
    def __init__(self, lasso_features, no_lasso_features, lasso_params=None):
        self.lasso_features = lasso_features
        self.no_lasso_features = no_lasso_features
        self.lasso_params = lasso_params if lasso_params is not None else {}
        self.lasso_model = LogisticRegressionCV(penalty='l1', solver='saga', **self.lasso_params)
        self.no_lasso_model = LogisticRegressionCV(penalty='l2', solver='lbfgs', **self.lasso_params)

    def fit(self, X, y):
        X_lasso = X[self.lasso_features]
        X_no_lasso = X[self.no_lasso_features]
        
        self.lasso_model.fit(X_lasso, y)
        self.no_lasso_model.fit(X_no_lasso, y)
        return self

    def predict(self, X):
        y_combined = self._combine_predictions(X)
        return (y_combined >= 0.5).astype(np.int8)

    def predict_proba(self, X):
        y_combined = self._combine_predictions(X)
        return np.vstack([1 - y_combined, y_combined]).T

    def _combine_predictions(self, X):
        X_lasso = X[self.lasso_features]
        X_no_lasso = X[self.no_lasso_features]
        
        y_lasso = self.lasso_model.predict_proba(X_lasso)[:, 1]
        y_no_lasso = self.no_lasso_model.predict_proba(X_no_lasso)[:, 1]
        
        y_combined = y_lasso + y_no_lasso
        return y_combined / 2  