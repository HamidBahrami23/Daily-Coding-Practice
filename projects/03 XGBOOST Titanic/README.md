# Titanic Survival Prediction with XGBoost

A machine learning project that predicts passenger survival on the Titanic using XGBoost classification algorithm.

## Overview

This project implements a machine learning solution for the classic Titanic survival prediction problem. The model uses XGBoost (Extreme Gradient Boosting) to classify whether passengers survived the Titanic disaster based on various features such as age, gender, class, and more.

## Model Performance

- **Logistic Regression**: Achieved an accuracy of 0.79 (79%)
- **XGBoost**: Achieved an accuracy of 0.80-0.84 (80-85%)

## Features

- Data preprocessing and feature engineering
- Implementation of XGBoost classifier
- Baseline comparison with Logistic Regression
- Model evaluation and accuracy metrics

## Requirements

- Python 3.x
- XGBoost
- Pandas
- NumPy
- Scikit-learn

## Installation

```bash
pip install xgboost pandas numpy scikit-learn
```


## Results

The XGBoost model significantly outperforms the baseline Logistic Regression model, achieving an accuracy improvement of 1-5 percentage points. Ongoing improvements are being implemented to enhance model performance further.

## Future Improvements

- Feature engineering optimizations
- Hyperparameter tuning
- Cross-validation implementation
- Additional model evaluation metrics
- Model interpretability analysis