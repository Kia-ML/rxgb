"""RXGB: Advanced Ensembles of Gradient-Boosted Learners.

A unified framework that provides classic bagging ensembles and deep 
cascaded architectures built on top of XGBoost.

The framework provides four scikit-learn compatible estimators:

- :class:`RXGBClassifier` for classic bagging-based classification.
- :class:`RXGBRegressor` for classic bagging-based regression.
- :class:`DeepRXGBClassifier` for deep, multi-layered classification.
- :class:`DeepRXGBRegressor` for deep, multi-layered regression.

Examples
--------
>>> # Example 1: Classic RXGB (Bagging)
>>> from rxgb import RXGBClassifier
>>> from sklearn.datasets import load_iris
>>> X, y = load_iris(return_X_y=True)
>>> clf = RXGBClassifier(n_estimators=25, max_depth=4)
>>> clf.fit(X, y)
RXGBClassifier(...)

>>> # Example 2: Deep RXGB (Cascaded Network)
>>> from rxgb import DeepRXGBClassifier
>>> from sklearn.datasets import make_classification
>>> X_deep, y_deep = make_classification(n_samples=1000, n_features=20, random_state=42)
>>> deep_clf = DeepRXGBClassifier(n_layers=3, mode='sequential')
>>> deep_clf.fit(X_deep, y_deep)
DeepRXGBClassifier(...)
"""

from .core import (
    RXGBClassifier, 
    RXGBRegressor, 
    DeepRXGBClassifier, 
    DeepRXGBRegressor
)

__version__ = "0.2.1"
__author__ = "Kiarash Nouri"
__email__ = "kiarash.nr.ml@gmail.com"

__all__ = [
    "RXGBClassifier",
    "RXGBRegressor",
    "DeepRXGBClassifier",
    "DeepRXGBRegressor",
]
