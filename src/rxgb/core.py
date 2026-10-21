"""RXGB Core: Advanced Ensembles of Gradient-Boosted Learners.

This module provides four powerful, scikit-learn compatible estimators:

- :class:`RXGBClassifier` for bagging-based classification tasks.
- :class:`RXGBRegressor` for bagging-based regression tasks.
- :class:`DeepRXGBClassifier` for deep cascaded/parallel classification.
- :class:`DeepRXGBRegressor` for deep cascaded/parallel regression.

Both architectures are optimized for memory efficiency and execution speed.
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
        """Validate the input arrays X and y."""
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

    def _make_base_model(self, seed):
        """Create a single base estimator. Must be overridden."""
        raise NotImplementedError

    def fit(self, X, y):
        """Build the ensemble from the training data."""
        X, y = self._validate_input(X, y)
        n, p = X.shape
        rng = np.random.RandomState(self.random_state)

        self.models_ = []
        self.features_ = []
        
        sample_size = int(self.max_samples * n)
        feature_size = max(1, int(self.max_features * p))

        for i in range(self.n_estimators):
            idx = rng.choice(n, size=sample_size, replace=True)
            feats = rng.choice(p, size=feature_size, replace=False)
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
            return XGBClassifier(n_estimators=100, max_depth=4, 
                                 n_jobs=self.n_jobs, 
                                 random_state=seed, eval_metric='logloss')

    def _create_reg_model(self, seed):
        if self.base_estimator == 'rxgb':
            return RXGBRegressor(n_estimators=10, max_depth=4, 
                                 inner_n_estimators=30, n_jobs=self.n_jobs, 
                                 random_state=seed)
        else:
            return XGBRegressor(n_estimators=100, max_depth=4, 
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
        The total number of XGBoost models to train in the ensemble. 
        A higher value increases stability and accuracy but also increases 
        training time. Typical optimal ranges are between 20 and 50.

    max_depth : int, default=4
        The maximum depth of each XGBoost tree. Keep this value relatively 
        small (3 to 6). If the trees are too deep, the base models become too 
        strong, and the bagging process loses its effect in reducing variance.

    learning_rate : float, default=0.1
        The step size shrinkage used in each boosting round inside the XGBoost 
        models. It prevents overfitting by making the optimization process 
        more conservative.

    inner_n_estimators : int, default=50
        The number of boosting rounds (trees) built *inside* each base 
        XGBoost model. This dictates the internal complexity of each neuron 
        in the ensemble.

    max_samples : float, default=0.8
        The fraction of samples to draw from X to train each base estimator. 
        Must be between 0.0 and 1.0. Draws are done with replacement (Bootstrap).

    max_features : float, default=0.8
        The fraction of features to consider when looking for the best split 
        for each base estimator. Must be between 0.0 and 1.0. Subsampling 
        features ensures diversity among the base models.

    n_jobs : int, default=None
        The number of parallel jobs to run for training and prediction. 
        ``None`` means 1, while ``-1`` means using all available processor cores, 
        which drastically speeds up training.

    random_state : int, default=42
        Controls the randomness of the bootstrap sampling, feature 
        subsampling, and the internal random seeds of the XGBoost models. 
        Set to a fixed integer for reproducible outputs.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels found in the training data.

    models_ : list of XGBClassifier
        The collection of trained XGBoost base estimators.

    features_ : list of ndarray
        The specific feature indices used by each corresponding base estimator.

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
    >>> clf = RXGBClassifier(n_estimators=30, max_depth=5, random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> 
    >>> preds = clf.predict(X_test)
    >>> print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    """

    def _make_base_model(self, seed):
        return XGBClassifier(
            n_estimators=self.inner_n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            random_state=seed,
            eval_metric='logloss',
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
        X = np.asarray(X, dtype=np.float32)
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


# ==============================================================================
# 2. RXGB REGRESSOR
# ==============================================================================
class RXGBRegressor(_RXGBBase, RegressorMixin):
    """RXGB regressor: Bagging ensemble of shallow XGBoost regressors.

    Similar to the classifier, this model builds a bagging ensemble of shallow 
    XGBoost regressors. Each base model is trained on a random subset of data 
    and features, and their continuous outputs are averaged to produce the 
    final prediction. It is highly robust against outliers and noisy targets.

    Parameters
    ----------
    n_estimators : int, default=25
        The total number of XGBoost models in the ensemble.

    max_depth : int, default=4
        The maximum depth of each XGBoost tree. Small depths are recommended 
        to keep the base models weak and the bagging process effective.

    learning_rate : float, default=0.1
        The step size shrinkage for the XGBoost optimization.

    inner_n_estimators : int, default=50
        The number of boosting rounds inside each base XGBoost model.

    max_samples : float, default=0.8
        The fraction of samples to draw to train each base model.

    max_features : float, default=0.8
        The fraction of features to use for each base model.

    n_jobs : int, default=None
        The number of parallel jobs. ``None`` means 1, ``-1`` uses all processors.

    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    models_ : list of XGBRegressor
        The collection of trained XGBoost base estimators.

    features_ : list of ndarray
        The feature indices used by each base estimator.

    Examples
    --------
    >>> from rxgb import RXGBRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.metrics import r2_score
    >>> 
    >>> X, y = make_regression(n_samples=1000, n_features=10, noise=25.0, random_state=42)
    >>> 
    >>> reg = RXGBRegressor(n_estimators=20, max_depth=4, random_state=42)
    >>> reg.fit(X, y)
    >>> 
    >>> preds = reg.predict(X)
    >>> print(f"R2 Score: {r2_score(y, preds):.4f}")
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
        X = np.asarray(X, dtype=np.float32)
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
        X = np.asarray(X, dtype=np.float32)
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

    This estimator constructs a multi-layered architecture (similar to Deep 
    Neural Networks or Deep Forests) using XGBoost or RXGB models as "neurons". 
    Information flows from one layer to the next. To prevent information loss, 
    the original input features are concatenated with the outputs of each layer 
    (Skip Connections). It features automatic early stopping to avoid overfitting.

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
        stops adding new layers if the validation accuracy fails to improve, 
        effectively preventing overfitting and saving computation time.

    mode : {'parallel', 'sequential'}, default='parallel'
        The architecture of the network:
        - 'parallel': Multiple models per layer vote together, resembling a 
          Deep Forest.
        - 'sequential': Exactly one model per layer. The output of layer N is 
          passed directly to layer N+1, acting like a cascaded correction 
          mechanism (Iterative Stacking).

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
    >>> 
    >>> X, y = make_classification(n_samples=2000, n_features=30, n_informative=15, random_state=42)
    >>> 
    >>> # Example 1: Parallel Mode (Deep Forest style)
    >>> model_parallel = DeepRXGBClassifier(n_layers=4, models_per_layer=3, mode='parallel')
    >>> model_parallel.fit(X, y)
    >>> print(f"Network depth trained: {len(model_parallel.layers_)} layers")
    >>> 
    >>> # Example 2: Sequential Mode (Cascaded chain)
    >>> model_seq = DeepRXGBClassifier(n_layers=5, mode='sequential', early_stopping=True)
    >>> model_seq.fit(X, y)
    >>> preds = model_seq.predict(X)
    """

    def fit(self, X, y):
        """Fit the Deep RXGB classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target class labels.

        Returns
        -------
        self : DeepRXGBClassifier
            Fitted classifier.
        """
        X = np.asarray(X, dtype=np.float32)
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
        n_classes = len(self.classes_)

        for layer_idx in range(self.n_layers):
            layer_models = []
            
            if self.mode == 'parallel':
                train_probs = np.zeros((X_train.shape[0], n_classes * self.models_per_layer), dtype=np.float32)
                val_probs = np.zeros((X_val.shape[0], n_classes * self.models_per_layer), dtype=np.float32)
                
                for i in range(self.models_per_layer):
                    seed = self.random_state + layer_idx * 100 + i
                    model = self._create_base_model(seed)
                    model.fit(current_input_train, y_train)
                    layer_models.append(model)
                    
                    train_probs[:, i*n_classes:(i+1)*n_classes] = model.predict_proba(current_input_train)
                    val_probs[:, i*n_classes:(i+1)*n_classes] = model.predict_proba(current_input_val)
                
                train_output = np.mean(train_probs.reshape(-1, self.models_per_layer, n_classes), axis=1)
                val_output = np.mean(val_probs.reshape(-1, self.models_per_layer, n_classes), axis=1)
            
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
            
            current_input_train = np.hstack([X_train, train_output])
            current_input_val = np.hstack([X_val, val_output])

        self.layers_ = self.layers_[:best_layer_count]
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
            The predicted class probabilities from the final layer.
        """
        X = np.asarray(X, dtype=np.float32)
        current_input = X.copy()
        
        for layer_models in self.layers_:
            if self.mode == 'parallel':
                n_classes = len(self.classes_)
                probs = np.zeros((X.shape[0], n_classes * self.models_per_layer), dtype=np.float32)
                for i, m in enumerate(layer_models):
                    probs[:, i*n_classes:(i+1)*n_classes] = m.predict_proba(current_input)
                layer_output = np.mean(probs.reshape(-1, self.models_per_layer, n_classes), axis=1)
            else: 
                layer_output = layer_models[0].predict_proba(current_input)
                
            current_input = np.hstack([X, layer_output])
            
        return layer_output

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
        The maximum number of layers (depth) in the network. The model will 
        stop early if validation performance drops.

    models_per_layer : int, default=3
        The number of parallel models trained per layer. Outputs are averaged. 
        Ignored if ``mode='sequential'``.

    base_estimator : {'rxgb', 'xgb'}, default='rxgb'
        The type of model used as the base building block. 
        'rxgb' uses the bagging ensemble, 'xgb' uses a standard XGBoost.

    early_stopping : bool, default=True
        Stops training additional layers if validation R-squared drops, 
        preventing overfitting.

    mode : {'parallel', 'sequential'}, default='parallel'
        - 'parallel': Multiple models per layer, predictions are averaged.
        - 'sequential': One model per layer, acts as a cascaded chain where 
          each layer refines the previous layer's output.

    n_jobs : int, default=None
        The number of jobs to run in parallel. ``None`` means 1, ``-1`` means 
        using all processors.

    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    layers_ : list of lists
        The trained models organized by layer. The length of this list shows 
        the optimal depth found by early stopping.

    Examples
    --------
    >>> from rxgb import DeepRXGBRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.metrics import r2_score
    >>> 
    >>> X, y = make_regression(n_samples=1500, n_features=20, noise=15.0, random_state=42)
    >>> 
    >>> # Sequential mode (Cascaded chain)
    >>> reg = DeepRXGBRegressor(n_layers=4, mode='sequential', early_stopping=True)
    >>> reg.fit(X, y)
    >>> 
    >>> preds = reg.predict(X)
    >>> print(f"R2 Score: {r2_score(y, preds):.4f}")
    """

    def fit(self, X, y):
        """Fit the Deep RXGB regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : DeepRXGBRegressor
            Fitted regressor.
        """
        X = np.asarray(X, dtype=np.float32)
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
                train_preds = np.zeros((X_train.shape[0], self.models_per_layer), dtype=np.float32)
                val_preds = np.zeros((X_val.shape[0], self.models_per_layer), dtype=np.float32)
                
                for i in range(self.models_per_layer):
                    seed = self.random_state + layer_idx * 100 + i
                    model = self._create_reg_model(seed)
                    model.fit(current_input_train, y_train)
                    layer_models.append(model)
                    
                    train_preds[:, i] = model.predict(current_input_train)
                    val_preds[:, i] = model.predict(current_input_val)
                
                train_output = np.mean(train_preds, axis=1).reshape(-1, 1)
                val_output = np.mean(val_preds, axis=1).reshape(-1, 1)
                
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
            
            current_input_train = np.hstack([X_train, train_output])
            current_input_val = np.hstack([X_val, val_output])

        self.layers_ = self.layers_[:best_layer_count]
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
            Predicted target values from the final layer.
        """
        X = np.asarray(X, dtype=np.float32)
        current_input = X.copy()
        
        for layer_models in self.layers_:
            if self.mode == 'parallel':
                preds = np.zeros((X.shape[0], self.models_per_layer), dtype=np.float32)
                for i, m in enumerate(layer_models):
                    preds[:, i] = m.predict(current_input)
                layer_output = np.mean(preds, axis=1).reshape(-1, 1)
            else: 
                layer_output = layer_models[0].predict(current_input).reshape(-1, 1)
                
            current_input = np.hstack([X, layer_output])
            
        return layer_output.flatten()
