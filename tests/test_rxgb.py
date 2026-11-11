"""Automated tests for the RXGB library (v0.2.2)."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from rxgb import RXGBClassifier, RXGBRegressor


class TestRXGBClassifier:
    """Tests for RXGBClassifier."""

    def test_fit_predict_basic(self):
        """Test basic fit and predict functionality."""
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RXGBClassifier(n_estimators=5, max_depth=3)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        assert len(preds) == len(y_test)
        assert accuracy_score(y_test, preds) > 0.8

    def test_predict_proba(self):
        """Test that predicted probabilities sum to 1."""
        X, y = load_iris(return_X_y=True)
        model = RXGBClassifier(n_estimators=5)
        model.fit(X, y)
        proba = model.predict_proba(X[:10])
        
        # Sum of probabilities for each sample should be 1.0
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_multiclass_dynamic_strategy(self):
        """Test that the model handles 5-class data (Dynamic Strategy) correctly."""
        # Generating a difficult 5-class dataset (similar to our benchmarks)
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=15, 
            n_classes=5, random_state=42
        )
        model = RXGBClassifier(n_estimators=5, max_depth=3)
        model.fit(X, y)
        preds = model.predict(X)
        
        # Assert that it predicts multiple classes successfully
        assert len(np.unique(preds)) > 2
        assert accuracy_score(y, preds) > 0.5

    def test_string_labels(self):
        """Test that the classifier handles string labels correctly."""
        X, y = load_iris(return_X_y=True)
        y_str = np.array(["setosa", "versicolor", "virginica"])[y]
        
        model = RXGBClassifier(n_estimators=5)
        model.fit(X, y_str)
        preds = model.predict(X[:10])
        
        assert set(preds).issubset(set(y_str))

    def test_invalid_X_2d(self):
        """Test that a 1D X raises a ValueError."""
        X = np.array([1, 2, 3])  # 1D array
        y = np.array([0, 1, 0])
        
        model = RXGBClassifier(n_estimators=5)
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_reproducibility(self):
        """Test that identical random_state yields identical results."""
        X, y = load_iris(return_X_y=True)
        
        m1 = RXGBClassifier(n_estimators=5, random_state=42)
        m1.fit(X, y)
        
        m2 = RXGBClassifier(n_estimators=5, random_state=42)
        m2.fit(X, y)
        
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))

    def test_rare_class_is_present_in_every_base_learner(self):
        """Bootstrap sampling must not silently drop a minority class."""
        X = np.random.RandomState(0).normal(size=(30, 4))
        y = np.array(["common"] * 29 + ["rare"])

        model = RXGBClassifier(n_estimators=4, max_samples=0.5, random_state=7)
        model.fit(X, y)

        assert len(model.classes_) == 2
        assert all(np.array_equal(estimator.classes_, np.array([0, 1]))
                   for estimator in model.models_)
        assert model.predict_proba(X[:5]).shape == (5, 2)

    def test_rejects_single_class_target(self):
        X = np.random.RandomState(0).normal(size=(20, 3))
        y = np.zeros(20, dtype=int)
        with pytest.raises(ValueError, match="at least two distinct classes"):
            RXGBClassifier(n_estimators=2).fit(X, y)

    @pytest.mark.parametrize("kwargs", [
        {"n_estimators": 0},
        {"inner_n_estimators": 0},
        {"max_depth": 0},
        {"max_samples": 0},
        {"max_samples": 1.1},
        {"max_features": 0},
        {"max_features": 1.1},
    ])
    def test_rejects_invalid_parameters(self, kwargs):
        X, y = load_iris(return_X_y=True)
        with pytest.raises(ValueError):
            RXGBClassifier(**kwargs).fit(X, y)

    def test_rejects_bootstrap_smaller_than_number_of_classes(self):
        X = np.random.RandomState(0).normal(size=(12, 3))
        y = np.array([0, 1, 2] * 4)
        with pytest.raises(ValueError, match="too small to include every class"):
            RXGBClassifier(n_estimators=2, max_samples=0.1).fit(X, y)

    def test_predict_proba_rejects_wrong_feature_count(self):
        X, y = load_iris(return_X_y=True)
        model = RXGBClassifier(n_estimators=2).fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict_proba(X[:, :2])

    def test_float32_conversion(self):
        """Test that the model handles float64 inputs (converts to float32 internally)."""
        X, y = load_iris(return_X_y=True)
        X_float64 = X.astype(np.float64)
        
        model = RXGBClassifier(n_estimators=5)
        model.fit(X_float64, y)
        preds = model.predict(X_float64)
        
        assert len(preds) == len(y)


class TestRXGBRegressor:
    """Tests for RXGBRegressor."""

    def test_fit_predict_basic(self):
        """Test basic fit and predict functionality for regression."""
        X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RXGBRegressor(n_estimators=5, max_depth=3)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        assert r2_score(y_test, preds) > 0.5

    def test_numeric_y_only(self):
        """Test that regression with string y raises a ValueError."""
        X = np.random.rand(50, 5)
        y_str = np.array(["a", "b"] * 25)  # String target
        
        model = RXGBRegressor(n_estimators=5)
        with pytest.raises(ValueError):
            model.fit(X, y_str)
