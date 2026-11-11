"""RXGB Core: Bagging ensemble of shallow XGBoost learners with Attention.

This module provides two highly optimized, scikit-learn compatible estimators:

- :class:`RXGBClassifier` for classification tasks.
- :class:`RXGBRegressor` for regression tasks.

Both estimators feature a Dynamic Weighting (Attention) mechanism by default.
Instead of a simple average, a meta-learner learns the optimal weight for each 
base learner based on the input, making the model exceptionally robust against 
noise, complex patterns, and highly overlapping classes.
"""

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge


class _RXGBBase(BaseEstimator):
    """Base class shared by RXGBClassifier and RXGBRegressor.

    Parameters
    ----------
    n_estimators : int, default=25
        The number of independent XGBoost base learners in the bagging 
        ensemble. A higher value increases stability but takes longer to train.

    max_depth : int, default=4
        The maximum depth of each XGBoost tree. It is recommended to keep 
        this value small (between 3 and 6) to prevent overfitting and 
        ensure diversity among the ensemble members.

    learning_rate : float, default=0.1
        The step size shrinkage (learning rate) used in each boosting round 
        inside the XGBoost models. It prevents overfitting by making the 
        optimization process more conservative.

    inner_n_estimators : int, default=50
        The number of boosting rounds (trees) built *inside* each base 
        XGBoost model. This dictates the internal complexity of each learner.

    max_samples : float, default=0.8
        The fraction of samples drawn (with replacement) to train each 
        base learner. A value less than 1.0 introduces diversity (variance 
        reduction) through bootstrap sampling.

    max_features : float, default=0.8
        The fraction of input features randomly selected to train each 
        base learner. Subsampling features ensures that the models do not 
        all look identical.

    dynamic_weighting : bool, default=True
        If True, activates the Attention mechanism. Instead of a simple 
        majority vote or average, a meta-learner dynamically learns the 
        optimal weight for each base learner's prediction. 
        This significantly boosts performance on noisy and overlapping data.

    n_jobs : int, default=None
        The number of parallel jobs to run for training and prediction. 
        ``None`` means 1 (no parallelism), while ``-1`` uses all available 
        processor cores.

    random_state : int, default=42
        Controls the randomness of the bootstrap sampling, feature 
        selection, and internal model seeds. Set to a fixed integer for 
        reproducible outputs.
    """

    def __init__(self, n_estimators=25, max_depth=4, learning_rate=0.1,
                 inner_n_estimators=50, max_samples=0.8, max_features=0.8,
                 dynamic_weighting=True, n_jobs=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.inner_n_estimators = inner_n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.dynamic_weighting = dynamic_weighting
        self.n_jobs = n_jobs
        self.random_state = random_state

    def validate_input(self, X, y):
        """Validate and convert input arrays to float32 for speed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Validated float32 array for X.
        y : ndarray of shape (n_samples,)
            Validated array for y.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(y) != X.shape[0]:
            raise ValueError(f"X and y have inconsistent lengths: {X.shape[0]} vs {len(y)}")
        return X, y

    # Backwards-compatible alias used by the existing regressor implementation.
    _validate_input = validate_input


class RXGBClassifier(_RXGBBase, ClassifierMixin):
    """RXGB Classifier: A robust ensemble of shallow XGBoost classifiers.

    RXGB replaces the decision trees of a Random Forest with depth-constrained 
    XGBoost learners. By default, it uses an Attention mechanism (dynamic 
    weighting) to intelligently combine the predictions of each base learner.

    Parameters
    ----------
    n_estimators : int, default=25
        Number of independent XGBoost base learners in the ensemble.

    max_depth : int, default=4
        Maximum tree depth used by each XGBoost base learner.

    learning_rate : float, default=0.1
        Learning rate used by each XGBoost base learner.

    inner_n_estimators : int, default=50
        Number of boosting rounds inside each XGBoost base learner.

    max_samples : float, default=0.8
        Fraction of training samples drawn for each base learner.

    max_features : float, default=0.8
        Fraction of input features selected for each base learner.

    dynamic_weighting : bool, default=True
        If True, uses a Logistic Regression meta-learner to dynamically 
        weight the predictions of base learners. Highly recommended for 
        noisy, complex, or imbalanced datasets.

    n_jobs : int, default=None
        Number of parallel jobs. ``None`` means 1, ``-1`` uses all processors.

    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels found in the training data.

    models_ : list of XGBClassifier
        The trained XGBoost base estimators.

    features_ : list of ndarray
        The specific feature indices used by each corresponding base estimator.

    meta_learner_ : LogisticRegression
        The trained meta-learner (only available if dynamic_weighting=True).

    Examples
    --------
    >>> from rxgb import RXGBClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score
    >>> 
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> 
    >>> # The Attention mechanism is enabled by default
    >>> clf = RXGBClassifier(n_estimators=25, max_depth=4, random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> 
    >>> preds = clf.predict(X_test)
    >>> print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    """

    def _validate_classifier_parameters(self):
        """Validate parameters that control classifier ensemble construction."""
        for name in ("n_estimators", "inner_n_estimators", "max_depth"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        for name in ("max_samples", "max_features"):
            value = getattr(self, name)
            if not isinstance(value, (int, float, np.integer, np.floating)) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in the interval (0, 1], got {value!r}")

        if not isinstance(self.dynamic_weighting, (bool, np.bool_)):
            raise ValueError("dynamic_weighting must be a boolean")

    def fit(self, X, y):
        """Fit the RXGB classifier to the training data.

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
        X, y = self.validate_input(X, y)
        self._validate_classifier_parameters()

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("y must contain at least two distinct classes")

        n, p = X.shape
        self.n_features_in_ = p
        sample_size = max(1, int(self.max_samples * n))
        if sample_size < n_classes:
            raise ValueError(
                "max_samples is too small to include every class in a base learner; "
                "increase max_samples or provide more training samples"
            )

        rng = np.random.RandomState(self.random_state)
        self.models_ = []
        self.features_ = []
        
        if self.dynamic_weighting:
            meta_features = np.zeros((n, self.n_estimators * n_classes), dtype=np.float32)

        feature_size = max(1, int(self.max_features * p))
        class_indices = [np.flatnonzero(y_encoded == class_id) for class_id in range(n_classes)]
        class_probabilities = np.bincount(y_encoded, minlength=n_classes) / n

        for i in range(self.n_estimators):
            # Stratified bootstrap: guarantee that every base learner sees every class.
            class_draws = rng.multinomial(sample_size - n_classes, class_probabilities) + 1
            idx = np.concatenate([
                rng.choice(indices, size=draws, replace=True)
                for indices, draws in zip(class_indices, class_draws)
            ])
            rng.shuffle(idx)
            feats = rng.choice(p, size=feature_size, replace=False)
            
            model = XGBClassifier(
                n_estimators=self.inner_n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_jobs=self.n_jobs,
                random_state=int(rng.randint(np.iinfo(np.int32).max)),
                eval_metric='logloss' if n_classes == 2 else 'mlogloss',
            )
            
            model.fit(X[np.ix_(idx, feats)], y_encoded[idx])
            self.models_.append(model)
            self.features_.append(feats)
            
            if self.dynamic_weighting:
                probas = model.predict_proba(X[:, feats])
                if probas.shape[1] < n_classes:
                    full = np.zeros((n, n_classes), dtype=np.float32)
                    for j, c in enumerate(model.classes_):
                        full[:, c] = probas[:, j]
                    probas = full
                meta_features[:, i*n_classes:(i+1)*n_classes] = probas

        # Train the Attention Mechanism (Meta-Learner)
        if self.dynamic_weighting:
            self.meta_learner_ = LogisticRegression(max_iter=1000, random_state=self.random_state)
            self.meta_learner_.fit(meta_features, y_encoded)

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
            The predicted class probabilities.
        """
        if not hasattr(self, "classes_"):
            raise ValueError("This RXGBClassifier instance is not fitted yet. Call 'fit' before prediction.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but RXGBClassifier was fitted with "
                f"{self.n_features_in_} features"
            )
        n_classes = len(self.classes_)
        
        if self.dynamic_weighting:
            meta_features = np.zeros((X.shape[0], self.n_estimators * n_classes), dtype=np.float32)
            for i, (m, f) in enumerate(zip(self.models_, self.features_)):
                probas = m.predict_proba(X[:, f])
                if probas.shape[1] < n_classes:
                    full = np.zeros((X.shape[0], n_classes), dtype=np.float32)
                    for j, c in enumerate(m.classes_):
                        full[:, c] = probas[:, j]
                    probas = full
                meta_features[:, i*n_classes:(i+1)*n_classes] = probas
            return self.meta_learner_.predict_proba(meta_features)
        else:
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
            The predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class RXGBRegressor(_RXGBBase, RegressorMixin):
    """RXGB Regressor: A robust ensemble of shallow XGBoost regressors.

    This model is the regression counterpart to RXGBClassifier. It uses
    a Ridge regression meta-learner to dynamically combine the predictions 
    of multiple XGBoost regressors.

    Parameters
    ----------
    n_estimators : int, default=25
        Number of independent XGBoost base learners in the ensemble.

    max_depth : int, default=4
        Maximum tree depth used by each XGBoost base learner.

    learning_rate : float, default=0.1
        Learning rate used by each XGBoost base learner.

    inner_n_estimators : int, default=50
        Number of boosting rounds inside each XGBoost base learner.

    max_samples : float, default=0.8
        Fraction of training samples drawn for each base learner.

    max_features : float, default=0.8
        Fraction of input features selected for each base learner.

    dynamic_weighting : bool, default=True
        If True, activates the Attention mechanism for regression using a 
        Ridge meta-learner. Highly recommended for noisy regression data.

    n_jobs : int, default=None
        Number of parallel jobs. ``None`` means 1, ``-1`` uses all processors.

    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    models_ : list of XGBRegressor
        The trained XGBoost base estimators.

    features_ : list of ndarray
        The feature indices used by each base estimator.

    meta_learner_ : Ridge
        The trained meta-learner (only available if dynamic_weighting=True).

    Examples
    --------
    >>> from rxgb import RXGBRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.metrics import r2_score
    >>> 
    >>> X, y = make_regression(n_samples=1000, n_features=10, noise=15.0, random_state=42)
    >>> 
    >>> # Initialize with default parameters (Attention enabled)
    >>> reg = RXGBRegressor(n_estimators=25, max_depth=4, random_state=42)
    >>> reg.fit(X, y)
    >>> 
    >>> preds = reg.predict(X)
    >>> print(f"R2 Score: {r2_score(y, preds):.4f}")
    """

    def fit(self, X, y):
        """Fit the RXGB regressor to the training data.

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
            raise ValueError(f"y must be numeric for regression, got dtype {y.dtype}")
            
        n, p = X.shape
        rng = np.random.RandomState(self.random_state)
        self.models_ = []
        self.features_ = []
        
        if self.dynamic_weighting:
            meta_features = np.zeros((n, self.n_estimators), dtype=np.float32)

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
            
            if self.dynamic_weighting:
                meta_features[:, i] = model.predict(X[:, feats])

        # Train the Attention Mechanism (Meta-Learner)
        if self.dynamic_weighting:
            self.meta_learner_ = Ridge(alpha=1.0)
            self.meta_learner_.fit(meta_features, y)

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
            The predicted target values.
        """
        X = np.asarray(X, dtype=np.float32)
        
        if self.dynamic_weighting:
            meta_features = np.zeros((X.shape[0], self.n_estimators), dtype=np.float32)
            for i, (m, f) in enumerate(zip(self.models_, self.features_)):
                meta_features[:, i] = m.predict(X[:, f])
            return self.meta_learner_.predict(meta_features)
        else:
            preds = np.array([
                m.predict(X[:, f])
                for m, f in zip(self.models_, self.features_)
            ])
            return preds.mean(axis=0)
