"""Automated tests for the RXGB library."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, make_regression
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
        assert accuracy_score(y_test, preds) > 0.8  # Expect at least 80% accuracy

    def test_predict_proba(self):
        """Test that predicted probabilities sum to 1."""
        X, y = load_iris(return_X_y=True)
        model = RXGBClassifier(n_estimators=5)
        model.fit(X, y)
        proba = model.predict_proba(X[:10])
        
        # Sum of probabilities for each sample should be 1.0
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

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
        
        assert r2_score(y_test, preds) > 0.5  # Expect a reasonable R2 score

    def test_numeric_y_only(self):
        """Test that regression with string y raises a ValueError."""
        X = np.random.rand(50, 5)
        y_str = np.array(["a", "b"] * 25)  # String target
        
        model = RXGBRegressor(n_estimators=5)
        with pytest.raises(ValueError):
            model.fit(X, y_str)