# RXGB: Random eXtreme Gradient Boosting

A bagging ensemble framework that combines Random Forest's bagging strategy
with depth-constrained XGBoost learners as base estimators.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

RXGB substitutes conventional decision trees in Random Forest with
depth-constrained XGBoost learners. This combination simultaneously
reduces **bias** and **variance**, which neither standalone Random Forest
nor standalone XGBoost can achieve alone.

## Installation

```bash
pip install rxgb