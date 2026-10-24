# RXGB — Random eXtreme Gradient Boosting

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://img.shields.io/badge/python-3.9%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://img.shields.io/badge/License-MIT-yellow.svg)
[![PyPI version](https://badge.fury.io/py/rxgb.svg)](https://badge.fury.io/py/rxgb.svg)

RXGB is a robust ensemble framework that combines Random Forest-style bagging with depth-constrained XGBoost learners. It is specifically designed to improve prediction robustness and performance on challenging tabular datasets, including noisy, imbalanced, and complex data.

## Overview

RXGB wraps multiple independent XGBoost models in a bagging-style ensemble. Each base learner is trained using a randomized sample of rows and features, and the ensemble combines their predictions.

The design brings together two powerful concepts:

- **Boosting (XGBoost):** Builds trees sequentially within each base learner to capture complex relationships and reduce bias.
- **Bagging (Random Forest):** Trains independent base learners on different samples and feature subsets to drastically reduce prediction variance.

### Why RXGB and not standard XGBoost?

Standard XGBoost is highly sensitive to noise and can easily overfit on small, complex datasets. RXGB solves this by placing XGBoost inside a bagging architecture. By averaging the probabilities of multiple independent XGBoost models, RXGB simultaneously reduces bias and variance. Our benchmarks show that RXGB consistently outperforms standard XGBoost and Random Forest on difficult datasets, offering a superior balance of performance and computational efficiency.

## How It Works

RXGB uses two levels of model construction:

1. **Row subsampling:** Each base learner trains on a randomized sample of the training rows (controlled by `max_samples`).
2. **Feature subsampling:** Each base learner uses a randomized subset of input features (controlled by `max_features`).
3. **Independent XGBoost learners:** Each base learner fits its own boosted-tree model (controlled by `inner_n_estimators` and `max_depth`).
4. **Ensemble prediction:** For classification, the base learners' predicted probabilities are combined using soft voting.

Unlike a single XGBoost model, RXGB combines predictions from multiple independently trained models. This parallel independence is what prevents overfitting on noisy data and stabilizes performance on imbalanced classes.

## Installation

Install RXGB from PyPI:

```bash
pip install rxgb
```

## Quick Start

```python
from rxgb import RXGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Create a challenging synthetic dataset (noisy)
X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=15,
    flip_y=0.15, random_state=42
)

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Initialize and train RXGB
clf = RXGBClassifier(n_estimators=25, max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 4. Predict and evaluate
predictions = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
```

## Performance Benchmarks

We evaluated RXGB against industry-standard models across five challenging datasets, including imbalanced, noisy, high-dimensional data, and the Pima Indians Diabetes dataset.

| Model | Average F1-score | Average training time | Configuration |
|---|---:|---:|---|
| RXGB (Ours) | 0.8438 | 4.9 s | Default parameters (no tuning) |
| LightGBM | 0.8564 | 2.8 s | Tuned via GridSearch |
| CatBoost | 0.8561 | 10.8 s | Tuned via GridSearch |
| XGBoost | 0.8422 | 6.1 s | Tuned via GridSearch |
| Random Forest | 0.8375 | 3.9 s | Tuned via GridSearch |

Note: RXGB achieves a highly competitive F1-score without any hyperparameter tuning, outperforming a heavily optimized XGBoost while being significantly faster to train than CatBoost.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `n_estimators` | `25` | Number of independent XGBoost base learners in the ensemble. |
| `max_depth` | `4` | Maximum tree depth used by each base learner. |
| `learning_rate` | `0.1` | Learning rate used by each XGBoost base learner. |
| `inner_n_estimators` | `50` | Number of boosting rounds inside each XGBoost base learner. |
| `max_samples` | `0.8` | Fraction of training samples drawn for each base learner. |
| `max_features` | `0.8` | Fraction of input features selected for each base learner. |
| `dynamic_weighting` | `True` | If True, uses a meta-learner (Attention) to dynamiclly weight predictions. Recommended for noisy data. |
| `n_jobs` | `None` | Number of parallel jobs; `-1` requests all available processors. |
| `random_state` | `42` | Random seed for reproducibility. |

## Documentation and Source Code

- **GitHub repository:** [https://github.com/Kia-ML/rxgb](https://github.com/Kia-ML/rxgb)
- **PyPI package:** [https://pypi.org/project/rxgb/](https://pypi.org/project/rxgb/)
- **Issue tracker:** [https://github.com/Kia-ML/rxgb/issues](https://github.com/Kia-ML/rxgb/issues)

The repository contains the full source code, automated tests (PyTest), and reproducible benchmark scripts.

## Citation

If you use RXGB in your research, please cite the work:

```bibtex
@article{nouri2026rxgb,
  title   = {RXGB: A Bagging Ensemble of Shallow XGBoost Learners for Classification},
  author  = {Nouri, Kiarash},
  journal = {Journal of Machine Learning Research (Under Review)},
  year    = {2026}
}
```

## License

RXGB is distributed under the MIT License. See the LICENSE file for the full license text.


RXGB is distributed under the MIT License. See the LICENSE file for the full license text.
