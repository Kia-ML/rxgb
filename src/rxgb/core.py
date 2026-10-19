"""RXGB Core: Bagging ensemble of shallow XGBoost learners.

This module provides two scikit-learn compatible estimators:

- :class:`RXGBClassifier` for classification tasks.
- :class:`RXGBRegressor` for regression tasks.

Both estimators implement the RXGB architecture, which combines the
bagging strategy of Random Forest with depth-constrained XGBoost
learners as base estimators. This design reduces both bias and variance
simultaneously, which neither standalone Random Forest nor standalone
XGBoost can achieve.
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

    n_jobs : int, default=None
        The number of jobs to run in parallel. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using
        all processors. See :term:`Glossary <n_jobs>` for more details.

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
        """Validate the input arrays X and y."""
        X = np.asarray(X)
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

    def _make_base_model(self, seed):
        """Create a single base estimator. Must be overridden."""
        raise NotImplementedError

    def fit(self, X, y):
        """Build the RXGB ensemble from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = self._validate_input(X, y)
        n, p = X.shape
        rng = np.random.RandomState(self.random_state)

        self.models_ = []
        self.features_ = []

        for i in range(self.n_estimators):
            idx = rng.choice(
                n, size=int(self.max_samples * n), replace=True
            )
            feats = rng.choice(
                p, size=max(1, int(self.max_features * p)), replace=False
            )
            model = self._make_base_model(seed=self.random_state + i)
            model.fit(X[np.ix_(idx, feats)], y[idx])
            self.models_.append(model)
            self.features_.append(feats)

        return self


class RXGBClassifier(_RXGBBase, ClassifierMixin):
    """RXGB classifier: Bagging ensemble of shallow XGBoost classifiers.

    RXGB replaces the conventional decision trees of a Random Forest with
    depth-constrained XGBoost learners. The bagging layer (bootstrap
    sampling of rows and random subsampling of features) ensures diversity
    across base learners, while XGBoost's gradient boosting reduces bias.

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

    n_jobs : int, default=None
        The number of jobs to run in parallel. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using
        all processors. See :term:`Glossary <n_jobs>` for more details.

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
    array([0, 0, 0, 0, 0])
    """

    def _make_base_model(self, seed):
        return XGBClassifier(
            n_estimators=self.inner_n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            random_state=seed,
        )

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
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        super().fit(X, y_encoded)
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
        X = np.asarray(X)
        n_classes = len(self.classes_)
        probas = []
        for m, f in zip(self.models_, self.features_):
            p = m.predict_proba(X[:, f])
            if p.shape[1] < n_classes:
                full = np.zeros((p.shape[0], n_classes))
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

    def _make_base_model(self, seed):
        return XGBRegressor(
            n_estimators=self.inner_n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            random_state=seed,
        )

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
        X = np.asarray(X)
        y = np.asarray(y)
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError(
                f"y must be numeric for regression, got dtype {y.dtype}"
            )
        super().fit(X, y)
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
        X = np.asarray(X)
        preds = np.array([
            m.predict(X[:, f])
            for m, f in zip(self.models_, self.features_)
        ])
        return preds.mean(axis=0)
