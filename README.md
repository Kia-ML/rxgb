# RXGB — Random eXtreme Gradient Boosting

**An Advanced Ensemble Learning Framework for Classification and Regression**

RXGB is a Python machine learning framework that combines XGBoost-based ensemble learning with two complementary architectures: **Classic Bagging** and **Deep Cascaded Learning**. It provides scikit-learn-compatible estimators designed to explore powerful combinations of gradient-boosted trees, ensemble diversity, and layered predictive learning.

The framework includes four estimators for classification and regression, offering flexible model configurations for a wide range of supervised learning tasks.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/rxgb.svg)](https://pypi.org/project/rxgb/)

---

## Overview

RXGB provides two main ensemble architectures.

### 1. Classic RXGB — Bagging Architecture

**Estimators:** `RXGBClassifier` and `RXGBRegressor`

Classic RXGB combines multiple depth-constrained XGBoost learners using a bagging-style ensemble strategy.

Instead of relying on a single gradient-boosted model, the architecture aggregates predictions from multiple independently trained learners. This approach is designed to leverage the predictive strength of XGBoost while reducing sensitivity to individual training samples.

Key features:

- Multiple XGBoost-based base learners.
- Configurable ensemble size through `n_estimators`.
- Configurable tree depth through `max_depth`.
- Support for classification and regression.
- Scikit-learn-compatible estimator interfaces.
- Compatibility with standard evaluation and model-selection workflows.

The architecture is intended to explore the balance between predictive accuracy, model complexity, and generalization performance.

### 2. Deep RXGB — Cascaded Learning Architecture

**Estimators:** `DeepRXGBClassifier` and `DeepRXGBRegressor`

Deep RXGB extends the RXGB framework with a layered architecture in which machine learning models act as computational units within a deeper predictive system.

It supports two operating modes:

**Parallel Mode**

Multiple models are trained within each layer. Their outputs are combined to generate representations for subsequent layers. This design draws inspiration from deep forest architectures and enables the framework to combine predictions from multiple learners.

**Sequential Mode**

Models are arranged in a sequential chain, with later stages processing information derived from preceding stages. This configuration provides an alternative approach to progressively refining predictions.

Additional capabilities include:

- Configurable maximum depth of the architecture through `n_layers`.
- Multiple models per layer in parallel mode.
- Selectable base learners.
- Optional early stopping to limit unnecessary layer growth.
- Support for both classification and regression.
- Skip connections where supported by the implementation.

Deep RXGB is designed to investigate how layered ensembles can improve predictive performance while controlling model complexity.

---

## Installation

Install RXGB using pip:

```bash
pip install rxgb
```

For the latest development version, visit the project's source repository.

---

## Quick Start

### 1. Classic RXGB Classifier

The following example trains an RXGB classifier on the Iris dataset.

```python
from rxgb import RXGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
X, y = load_iris(return_X_y=True)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Initialize and train RXGB
clf = RXGBClassifier(
    n_estimators=25,
    max_depth=4,
)

clf.fit(X_train, y_train)

# Generate predictions
predictions = clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, predictions)

print(f"RXGB Accuracy: {accuracy:.4f}")
```

### 2. Classic RXGB Regressor

RXGB also supports regression tasks involving continuous target variables.

```python
from rxgb import RXGBRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate a regression dataset
X, y = make_regression(
    n_samples=2000,
    n_features=20,
    noise=10,
    random_state=42,
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Initialize and train RXGB
reg = RXGBRegressor(
    n_estimators=25,
    max_depth=4,
)

reg.fit(X_train, y_train)

# Predict target values
predictions = reg.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
```

### 3. Deep RXGB — Parallel Mode

Parallel mode allows multiple models to operate within each layer.

```python
from rxgb import DeepRXGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a classification dataset
X, y = make_classification(
    n_samples=2000,
    n_features=30,
    n_informative=15,
    random_state=42,
)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Configure Deep RXGB
model = DeepRXGBClassifier(
    n_layers=3,
    models_per_layer=3,
    mode="parallel",
    early_stopping=True,
)

# Train the model
model.fit(X_train, y_train)

# Evaluate predictions
predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

# Inspect the learned architecture, if exposed by the implementation
if hasattr(model, "layers_"):
    print(f"Learned layers: {len(model.layers_)}")
```

### 4. Deep RXGB — Sequential Mode

Sequential mode configures Deep RXGB as a cascade of predictive stages.

```python
from rxgb import DeepRXGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=2000,
    n_features=30,
    n_informative=15,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = DeepRXGBClassifier(
    n_layers=5,
    mode="sequential",
    early_stopping=True,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
```

---

## Key Parameters

The following table summarizes the principal configuration options. Availability and behavior depend on the estimator and installed version.

| Parameter | Default | Description |
|---|---|---|
| `n_estimators` | `25` | Number of base learners in the classic RXGB ensemble. |
| `max_depth` | `4` | Maximum tree depth for the configured XGBoost learners. |
| `n_layers` | `3` | Maximum number of layers in the Deep RXGB architecture. |
| `models_per_layer` | Implementation-dependent | Number of models assigned to each parallel layer. |
| `mode` | `"parallel"` | Selects the parallel or sequential architecture. |
| `base_estimator` | `"rxgb"` | Selects the configured base learner, such as RXGB or XGBoost, where supported. |
| `early_stopping` | `True` | Enables the implementation's early-stopping mechanism, where supported. |

### Adjusting Model Complexity

RXGB allows users to explore different ensemble configurations by adjusting the number and depth of base learners.

For example, a model with fewer, deeper trees can be compared with one containing more, shallower trees:

```python
# Example configuration: more shallow learners
model_a = RXGBClassifier(
    n_estimators=100,
    max_depth=4,
)

# Example configuration: fewer, deeper learners
model_b = RXGBClassifier(
    n_estimators=25,
    max_depth=8,
)
```

These configurations provide different trade-offs between model complexity, training cost, and generalization. Neither configuration is guaranteed to outperform the other; the appropriate choice should be determined through validation.

---

## Model Evaluation and Hyperparameter Tuning

RXGB estimators are designed for integration with familiar scikit-learn workflows.

Recommended evaluation practices include:

- Use separate training, validation, and test datasets when appropriate.
- Apply stratified splitting for classification tasks with imbalanced class distributions.
- Compare RXGB against established baselines, including Random Forest, standalone XGBoost, and other relevant algorithms.
- Use cross-validation to estimate performance across multiple data splits.
- Optimize hyperparameters using `GridSearchCV` or `RandomizedSearchCV` when the estimator exposes the required scikit-learn interfaces.
- Evaluate classification with metrics such as accuracy, precision, recall, F1-score, and ROC-AUC, as appropriate.
- Evaluate regression with metrics such as MAE, MSE, RMSE, and R².
- Measure training time, prediction latency, and memory consumption alongside predictive performance.

For reliable comparisons, use identical data splits, consistent preprocessing, and appropriately controlled computational budgets.

---

## Architecture at a Glance

| Feature | Classic RXGB | Deep RXGB |
|---|---|---|
| Architecture | Bagging-style ensemble | Layered ensemble |
| Base learners | XGBoost-based models | Configurable RXGB or XGBoost-based learners |
| Classification | Supported | Supported |
| Regression | Supported | Supported |
| Adjustable tree depth | Supported | Depends on base learner configuration |
| Adjustable ensemble size | Supported | Through layer and model configuration |
| Parallel layers | Not applicable | Supported in parallel mode |
| Sequential cascade | Not applicable | Supported in sequential mode |
| Early stopping | Implementation-dependent | Available where supported |

---

## Design Goals

RXGB explores several ideas in ensemble learning:

1. **Ensemble diversity:** Combining multiple learners to reduce dependence on a single fitted model.
2. **Flexible complexity:** Allowing users to adjust tree depth, ensemble size, and architecture.
3. **Layered prediction:** Exploring multi-stage learning systems built from machine learning estimators.
4. **Practical integration:** Providing familiar Python APIs for training, prediction, and evaluation.
5. **Comparative experimentation:** Supporting systematic evaluation against established machine learning algorithms.

The framework is intended for experimentation, benchmarking, and further development in ensemble-based supervised learning.

---

## Limitations and Reproducibility

Performance depends on the dataset, preprocessing, hyperparameters, validation strategy, and computational environment.

RXGB should not be assumed to outperform Random Forest, standalone XGBoost, or other algorithms on every task. Claims about predictive superiority should be supported by reproducible experiments, appropriate baselines, and statistically meaningful evaluation.

Users should consult the implementation and package documentation to verify supported parameters, estimator behavior, and compatibility with their installed versions.

---

## Citation

If you use RXGB in research, please cite the work using the following provisional BibTeX entry. Replace the placeholder journal information with the actual publication details if the work is formally published.

```bibtex
@article{nouri2026rxgb,
  title   = {RXGB: Advanced Ensembles of Gradient-Boosted Learners for Classification and Regression},
  author  = {Nouri, Kiarash},
  year    = {2026},
  note    = {Preprint or unpublished manuscript}
}
```

---

## License

RXGB is distributed under the MIT License.

See the project's `LICENSE` file for the complete license terms.

---

**RXGB — Exploring the combination of gradient boosting, ensemble diversity, and deep cascaded learning in a unified Python framework.**
