"""
linear_regression.py

Ordinary Least Squares Linear Regression via gradient descent.

Math:
    Hypothesis:  y_hat = X @ w + b
    Cost:        J(w,b) = (1/2m) * sum((y_hat - y)^2)
    Gradient:    dJ/dw  = (1/m) * X.T @ (y_hat - y)
                 dJ/db  = (1/m) * sum(y_hat - y)
    Update:      w = w - lr * dJ/dw
                 b = b - lr * dJ/db
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from utils.metrics import mse, r2_score


class LinearRegression:
    """
    Linear Regression via gradient descent.

    Parameters
    ----------
    learning_rate : float, default 0.01
    n_iterations  : int, default 1000
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights: np.ndarray = None
        self.bias: float = 0.0
        self.loss_history: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Train the model using gradient descent.

        Parameters
        ----------
        X : shape (n_samples, n_features)
        y : shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iter):
            y_pred = self._forward(X)
            error  = y_pred - y

            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

            self.loss_history.append(mse(y, y_pred))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted values for X."""
        return self._forward(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² on given data."""
        return r2_score(y, self.predict(X))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias
