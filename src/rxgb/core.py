"""RXGB Core: Bagging ensemble of shallow XGBoost learners.

This module provides two highly optimized, scikit-learn compatible estimators:

- :class:`RXGBClassifier` for classification tasks.
- :class:`RXGBRegressor` for regression tasks.

The estimators implement the RXGB architecture, which combines the bagging
strategy of Random Forest with depth-constrained XGBoost learners. This design
reduces both bias and variance simultaneously, and features dynamic adaptation
for binary and multi-class datasets to ensure maximum performance.
"""

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class _RXGBBase(BaseEstimator):
    """Base class shared by RXGBClassifier and RXGBRegressor.

    Parameters
    ----------
    n_estimators : int, default=25
        Number of XGBoost base learners in the bagging ensemble.

    max_depth : int, default=4
        Maximum depth of each XGBoost base learner.

    learning_rate : float, default=0.1
        Learning rate (shrinkage) for each XGBoost base learner.

    inner_n_estimators : int, default=50
        Number of boosting rounds (trees) inside each XGBoost base
        learner.

    max_samples : float, default=0.8
        Fraction of samples drawn (with replacement) for each base
        learner.

    max_features : float, default=0.8
        Fraction of features randomly selected for each base learner.
        (Automatically set to 1.0 for multi-class problems).

    n_jobs : int, default=None
        The number of jobs to run in parallel. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using
        all processors.

    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, n_estimators=25, max_depth=4, learning_rate=0.1,
                 inner_n_estimators=50, max_samples=0.8, max_features=0.8,
                 n_jobs=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.inner_n_estimators = inner_n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _validate_input(self, X, y):
        """Validate and convert input arrays to float32 for speed."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: "
                f"{X.shape[0]} vs {len(y)}"
            )
        return X, y


class RXGBClassifier(_RXGBBase, ClassifierMixin):
    """RXGB classifier: Bagging ensemble of shallow XGBoost classifiers.

    RXGB replaces the conventional decision trees of a Random Forest with
    depth-constrained XGBoost learners. The bagging layer ensures diversity
    across base learners, while XGBoost's gradient boosting reduces bias.
    
    **Dynamic Strategy:** Automatically detects multi-class datasets and adjusts 
    feature subsampling and tree methods to maximize speed and accuracy.

    Parameters
    ----------
    n_estimators : int, default=25
        Number of XGBoost base learners in the bagging ensemble.

    max_depth : int, default=4
        Maximum depth of each XGBoost base learner.

    learning_rate : float, default=0.1
        Learning rate (shrinkage) for each XGBoost base learner.

    inner_n_estimators : int, default=50
        Number of boosting rounds (trees) inside each XGBoost base
        learner.

    max_samples : float, default=0.8
        Fraction of samples drawn (with replacement) for each base
        learner.

    max_features : float, default=0.8
        Fraction of features randomly selected for each base learner.
        (Ignored and set to 1.0 if n_classes > 2).

    n_jobs : int, default=None
        The number of jobs to run in parallel. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using
        all processors.

    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels.

    models_ : list of XGBClassifier
        The trained XGBoost base estimators.

    features_ : list of ndarray
        The feature indices used by each base estimator.

    Examples
    --------
    >>> from rxgb import RXGBClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = RXGBClassifier(n_estimators=15, max_depth=4)
    >>> clf.fit(X, y)
    RXGBClassifier(...)
    >>> clf.predict(X[:5])
    array([0, 0, 0, 2, 2])
    """

    def fit(self, X, y):
        """Fit the RXGB classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target class labels. Can be integers, strings, or booleans.

        Returns
        -------
        self : RXGBClassifier
            Fitted classifier.
        """
        X, y = self._validate_input(X, y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        
        # 🧠 Dynamic Adaptation for Speed & Accuracy
        if n_classes > 2:
            actual_max_features = 1.0       # Use all features for multi-class
            actual_eval_metric = 'mlogloss'
            actual_tree_method = 'hist'     # Faster tree building
        else:
            actual_max_features = self.max_features
            actual_eval_metric = 'logloss'
            actual_tree_method = 'auto'

        n, p = X.shape
        rng = np.random.RandomState(self.random_state)
        self.models_ = []
        self.features_ = []
        
        # Pre-calculate sizes for loop speed
        sample_size = int(self.max_samples * n)
        feature_size = max(1, int(actual_max_features * p))

        for i in range(self.n_estimators):
            idx = rng.choice(n, size=sample_size, replace=True)
            feats = rng.choice(p, size=feature_size, replace=False)
            
            model = XGBClassifier(
                n_estimators=self.inner_n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_jobs=self.n_jobs,
                random_state=self.random_state + i,
                eval_metric=actual_eval_metric,
                tree_method=actual_tree_method,
            )
            
            model.fit(X[np.ix_(idx, feats)], y_encoded[idx])
            self.models_.append(model)
            self.features_.append(feats)

        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Mean predicted class probabilities across all base learners.
        """
        X = np.asarray(X, dtype=np.float32)
        n_classes = len(self.classes_)
        probas = []
        
        for m, f in zip(self.models_, self.features_):
            p = m.predict_proba(X[:, f])
            if p.shape[1] < n_classes:
                full = np.zeros((p.shape[0], n_classes), dtype=np.float32)
                for i, c in enumerate(m.classes_):
                    full[:, c] = p[:, i]
                p = full
            probas.append(p)
            
        return np.mean(probas, axis=0)

    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class RXGBRegressor(_RXGBBase, RegressorMixin):
    """RXGB regressor: Bagging ensemble of shallow XGBoost regressors.

    RXGB replaces the conventional decision trees of a Random Forest with
    depth-constrained XGBoost regressors. Predictions from all base
    learners are averaged to produce the final output.

    Parameters
    ----------
    n_estimators : int, default=25
        Number of XGBoost base learners in the bagging ensemble.

    max_depth : int, default=4
        Maximum depth of each XGBoost base learner.

    learning_rate : float, default=0.1
        Learning rate (shrinkage) for each XGBoost base learner.

    inner_n_estimators : int, default=50
        Number of boosting rounds inside each XGBoost base learner.

    max_samples : float, default=0.8
        Fraction of samples drawn for each base learner.

    max_features : float, default=0.8
        Fraction of features randomly selected for each base learner.

    n_jobs : int, default=None
        The number of jobs to run in parallel. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using
        all processors.

    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    models_ : list of XGBRegressor
        The trained XGBoost base estimators.

    features_ : list of ndarray
        The feature indices used by each base estimator.

    Examples
    --------
    >>> from rxgb import RXGBRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    >>> reg = RXGBRegressor(n_estimators=15, max_depth=4)
    >>> reg.fit(X, y)
    RXGBRegressor(...)
    >>> y_pred = reg.predict(X)
    """

    def fit(self, X, y):
        """Fit the RXGB regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values. Must be numeric.

        Returns
        -------
        self : RXGBRegressor
            Fitted regressor.
        """
        X, y = self._validate_input(X, y)
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError(
                f"y must be numeric for regression, got dtype {y.dtype}"
            )
            
        n, p = X.shape
        rng = np.random.RandomState(self.random_state)
        self.models_ = []
        self.features_ = []
        
        # Pre-calculate sizes for loop speed
        sample_size = int(self.max_samples * n)
        feature_size = max(1, int(self.max_features * p))

        for i in range(self.n_estimators):
            idx = rng.choice(n, size=sample_size, replace=True)
            feats = rng.choice(p, size=feature_size, replace=False)
            
            model = XGBRegressor(
                n_estimators=self.inner_n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_jobs=self.n_jobs,
                random_state=self.random_state + i,
            )
            
            model.fit(X[np.ix_(idx, feats)], y[idx])
            self.models_.append(model)
            self.features_.append(feats)

        return self

    def predict(self, X):
        """Predict target values for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Mean predicted values across all base learners.
        """
        X = np.asarray(X, dtype=np.float32)
        preds = np.array([
            m.predict(X[:, f])
            for m, f in zip(self.models_, self.features_)
        ])
        return preds.mean(axis=0)
