"""RXGB Core: Advanced Ensembles of Gradient-Boosted Learners.

This module provides four powerful, scikit-learn compatible estimators:

- :class:`RXGBClassifier` for bagging-based classification tasks.
- :class:`RXGBRegressor` for bagging-based regression tasks.
- :class:`DeepRXGBClassifier` for deep cascaded/parallel classification.
- :class:`DeepRXGBRegressor` for deep cascaded/parallel regression.

Both the standard and deep architectures are designed to reduce bias and 
variance simultaneously, going beyond the capabilities of standalone XGBoost.
"""

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


# ==============================================================================
# BASE CLASSES (Internal use)
# ==============================================================================
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
        """Build the RXGB ensemble from the training data."""
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


class _DeepRXGBBase(BaseEstimator):
    """Base class shared by DeepRXGBClassifier and DeepRXGBRegressor.

    Parameters
    ----------
    n_layers : int, default=3
        The maximum number of layers (depth) in the network.

    models_per_layer : int, default=3
        Number of models trained in parallel in each layer. 
        Ignored if ``mode='sequential'``.

    base_estimator : {'rxgb', 'xgb'}, default='rxgb'
        The type of model to use as the "neuron". 'rxgb' uses the 
        bagging ensemble, 'xgb' uses a standard XGBoost.

    early_stopping : bool, default=True
        If True, stops adding layers if validation performance drops.

    mode : {'parallel', 'sequential'}, default='parallel'
        The architecture type. 'parallel' averages multiple models per 
        layer. 'sequential' passes the output of one model to the next.

    n_jobs : int, default=None
        The number of jobs to run in parallel for XGBoost training.

    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, n_layers=3, models_per_layer=3, base_estimator='rxgb',
                 early_stopping=True, mode='parallel', n_jobs=None, 
                 random_state=42):
        self.n_layers = n_layers
        self.models_per_layer = models_per_layer
        self.base_estimator = base_estimator
        self.early_stopping = early_stopping
        self.mode = mode
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _create_base_model(self, seed):
        if self.base_estimator == 'rxgb':
            return RXGBClassifier(n_estimators=10, max_depth=4, 
                                  inner_n_estimators=30, n_jobs=self.n_jobs, 
                                  random_state=seed)
        else:
            return XGBClassifier(n_estimators=50, max_depth=4, 
                                 n_jobs=self.n_jobs, 
                                 random_state=seed, eval_metric='logloss')

    def _create_reg_model(self, seed):
        if self.base_estimator == 'rxgb':
            return RXGBRegressor(n_estimators=10, max_depth=4, 
                                 inner_n_estimators=30, n_jobs=self.n_jobs, 
                                 random_state=seed)
        else:
            return XGBRegressor(n_estimators=50, max_depth=4, 
                                n_jobs=self.n_jobs, random_state=seed)


# ==============================================================================
# 1. RXGB CLASSIFIER
# ==============================================================================
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
        """Fit the RXGB classifier."""
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        super().fit(X, y_encoded)
        return self

    def predict_proba(self, X):
        """Predict class probabilities for X."""
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
        """Predict class labels for X."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ==============================================================================
# 2. RXGB REGRESSOR
# ==============================================================================
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
        """Fit the RXGB regressor."""
        X = np.asarray(X)
        y = np.asarray(y)
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError(
                f"y must be numeric for regression, got dtype {y.dtype}"
            )
        super().fit(X, y)
        return self

    def predict(self, X):
        """Predict target values for X."""
        X = np.asarray(X)
        preds = np.array([
            m.predict(X[:, f])
            for m, f in zip(self.models_, self.features_)
        ])
        return preds.mean(axis=0)


# ==============================================================================
# 3. DEEP RXGB CLASSIFIER
# ==============================================================================
class DeepRXGBClassifier(_DeepRXGBBase, ClassifierMixin):
    """Deep RXGB classifier: A neural network built from ML ensembles.

    This estimator stacks multiple layers of XGBoost or RXGB models. 
    Information flows from one layer to the next. The original input 
    features are concatenated with the outputs of each layer (Skip Connections).
    An internal validation set is used to automatically stop adding layers 
    if the performance starts to degrade (Early Stopping).

    Parameters
    ----------
    n_layers : int, default=3
        The maximum number of layers (depth) in the network. The model will 
        automatically stop before reaching this limit if performance plateaus.

    models_per_layer : int, default=3
        The number of parallel models trained in each layer. Their outputs 
        are averaged. This parameter is ignored if ``mode='sequential'``.

    base_estimator : {'rxgb', 'xgb'}, default='rxgb'
        The type of block used as a neuron. 
        - 'rxgb': Uses the powerful RXGB bagging ensemble.
        - 'xgb': Uses a standard single XGBoost model.

    early_stopping : bool, default=True
        If True, the network splits off 15% of the data for validation. It 
        stops adding new layers if the validation accuracy fails to improve.

    mode : {'parallel', 'sequential'}, default='parallel'
        The architecture of the network:
        - 'parallel': Multiple models per layer vote together (Deep Forest).
        - 'sequential': Exactly one model per layer, acting like a cascaded 
          correction mechanism (Iterative Stacking).

    n_jobs : int, default=None
        The number of jobs to run in parallel for XGBoost training.
        ``None`` means 1, ``-1`` means using all processors.

    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.

    layers_ : list of lists
        The trained models organized by layer. The length of this list shows 
        the optimal number of layers found by the early stopping mechanism.

    Examples
    --------
    >>> from rxgb import DeepRXGBClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    >>> # Parallel Mode (Deep Forest style)
    >>> clf = DeepRXGBClassifier(n_layers=4, models_per_layer=3, mode='parallel')
    >>> clf.fit(X, y)
    >>> print(f"Network depth: {len(clf.layers_)} layers")
    
    >>> # Sequential Mode (Cascaded chain)
    >>> clf_seq = DeepRXGBClassifier(n_layers=5, mode='sequential')
    >>> clf_seq.fit(X, y)
    >>> preds = clf_seq.predict(X)
    """

    def fit(self, X, y):
        """Fit the Deep RXGB classifier."""
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        
        if self.early_stopping and self.n_layers > 1:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=0.15, random_state=self.random_state, stratify=y_encoded
            )
        else:
            X_train, y_train = X, y_encoded
            X_val, y_val = X, y_encoded

        current_input_train = X_train.copy()
        current_input_val = X_val.copy()
        
        self.layers_ = []
        best_val_acc = 0.0
        best_layer_count = 0

        for layer_idx in range(self.n_layers):
            layer_models = []
            
            if self.mode == 'parallel':
                train_probs = []
                val_probs = []
                for i in range(self.models_per_layer):
                    seed = self.random_state + layer_idx * 100 + i
                    model = self._create_base_model(seed)
                    model.fit(current_input_train, y_train)
                    layer_models.append(model)
                    train_probs.append(model.predict_proba(current_input_train))
                    val_probs.append(model.predict_proba(current_input_val))
                
                train_output = np.mean(train_probs, axis=0)
                val_output = np.mean(val_probs, axis=0)
            
            elif self.mode == 'sequential':
                seed = self.random_state + layer_idx * 100
                model = self._create_base_model(seed)
                model.fit(current_input_train, y_train)
                layer_models.append(model)
                train_output = model.predict_proba(current_input_train)
                val_output = model.predict_proba(current_input_val)
            
            else:
                raise ValueError("mode must be 'parallel' or 'sequential'")

            val_pred = np.argmax(val_output, axis=1)
            current_val_acc = accuracy_score(y_val, val_pred)

            if self.early_stopping and layer_idx > 0 and current_val_acc < best_val_acc - 0.005:
                break
            
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_layer_count = layer_idx + 1

            self.layers_.append(layer_models)
            
            # Skip Connection
            current_input_train = np.hstack([X_train, train_output])
            current_input_val = np.hstack([X_val, val_output])

        self.layers_ = self.layers_[:best_layer_count]
        return self

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        X = np.asarray(X)
        current_input = X.copy()
        
        for layer_models in self.layers_:
            if self.mode == 'parallel':
                probs = [m.predict_proba(current_input) for m in layer_models]
                layer_output = np.mean(probs, axis=0)
            else: # sequential
                layer_output = layer_models[0].predict_proba(current_input)
                
            current_input = np.hstack([X, layer_output])
            
        return layer_output

    def predict(self, X):
        """Predict class labels for X."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ==============================================================================
# 4. DEEP RXGB REGRESSOR
# ==============================================================================
class DeepRXGBRegressor(_DeepRXGBBase, RegressorMixin):
    """Deep RXGB regressor: A multi-layered network for regression.

    This model builds a deep architecture where each layer attempts to refine 
    the predictions of the previous layer. It uses skip connections and automatic 
    early stopping based on the R-squared metric on a validation set.

    Parameters
    ----------
    n_layers : int, default=3
        The maximum number of layers (depth) in the network.

    models_per_layer : int, default=3
        The number of parallel models trained per layer (ignored in 
        sequential mode).

    base_estimator : {'rxgb', 'xgb'}, default='rxgb'
        The type of model used as the base building block.

    early_stopping : bool, default=True
        Stops training additional layers if validation R-squared drops.

    mode : {'parallel', 'sequential'}, default='parallel'
        - 'parallel': Multiple models per layer, predictions are averaged.
        - 'sequential': One model per layer, acts as a cascaded chain.

    n_jobs : int, default=None
        The number of jobs to run in parallel. ``None`` means 1, ``-1`` means 
        using all processors.

    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    layers_ : list of lists
        The trained models organized by layer.

    Examples
    --------
    >>> from rxgb import DeepRXGBRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=20, noise=15.0, random_state=42)
    
    >>> # Initialize a sequential network
    >>> reg = DeepRXGBRegressor(n_layers=4, mode='sequential', early_stopping=True)
    >>> reg.fit(X, y)
    >>> preds = reg.predict(X)
    """

    def fit(self, X, y):
        """Fit the Deep RXGB regressor."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.early_stopping and self.n_layers > 1:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, random_state=self.random_state
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = X, y

        current_input_train = X_train.copy()
        current_input_val = X_val.copy()
        
        self.layers_ = []
        best_val_r2 = -np.inf
        best_layer_count = 0

        for layer_idx in range(self.n_layers):
            layer_models = []
            
            if self.mode == 'parallel':
                train_preds = []
                val_preds = []
                for i in range(self.models_per_layer):
                    seed = self.random_state + layer_idx * 100 + i
                    model = self._create_reg_model(seed)
                    model.fit(current_input_train, y_train)
                    layer_models.append(model)
                    train_preds.append(model.predict(current_input_train))
                    val_preds.append(model.predict(current_input_val))
                
                train_output = np.mean(train_preds, axis=0).reshape(-1, 1)
                val_output = np.mean(val_preds, axis=0).reshape(-1, 1)
                
            elif self.mode == 'sequential':
                seed = self.random_state + layer_idx * 100
                model = self._create_reg_model(seed)
                model.fit(current_input_train, y_train)
                layer_models.append(model)
                train_output = model.predict(current_input_train).reshape(-1, 1)
                val_output = model.predict(current_input_val).reshape(-1, 1)

            else:
                raise ValueError("mode must be 'parallel' or 'sequential'")

            current_val_r2 = r2_score(y_val, val_output)

            if self.early_stopping and layer_idx > 0 and current_val_r2 < best_val_r2 - 0.01:
                break
            
            if current_val_r2 > best_val_r2:
                best_val_r2 = current_val_r2
                best_layer_count = layer_idx + 1

            self.layers_.append(layer_models)
            
            # Skip connection
            current_input_train = np.hstack([X_train, train_output])
            current_input_val = np.hstack([X_val, val_output])

        self.layers_ = self.layers_[:best_layer_count]
        return self

    def predict(self, X):
        """Predict target values for X."""
        X = np.asarray(X)
        current_input = X.copy()
        
        for layer_models in self.layers_:
            if self.mode == 'parallel':
                preds = [m.predict(current_input) for m in layer_models]
                layer_output = np.mean(preds, axis=0).reshape(-1, 1)
            else: # sequential
                layer_output = layer_models[0].predict(current_input).reshape(-1, 1)
                
            current_input = np.hstack([X, layer_output])
            
        return layer_output.flatten()
