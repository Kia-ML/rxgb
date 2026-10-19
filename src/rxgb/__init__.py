"""RXGB: Random eXtreme Gradient Boosting.

A bagging ensemble framework that combines Random Forest's bagging
strategy with depth-constrained XGBoost learners as base estimators.

The framework provides two scikit-learn compatible estimators:

- :class:`RXGBClassifier` for classification
- :class:`RXGBRegressor` for regression

Example
-------
>>> from rxgb import RXGBClassifier
>>> from sklearn.datasets import load_iris
>>> X, y = load_iris(return_X_y=True)
>>> clf = RXGBClassifier(n_estimators=25, max_depth=4)
>>> clf.fit(X, y)
RXGBClassifier(...)
"""

from .core import RXGBClassifier, RXGBRegressor

__version__ = "0.1.0"
__author__ = "Kiarash Nouri"
__email__ = "kiarash.nr.ml@gmail.com"
__all__ = ["RXGBClassifier", "RXGBRegressor"]