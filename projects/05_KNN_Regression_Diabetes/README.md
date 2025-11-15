# 🚀 05_KNN_Regression_Diabetes: Disease Progression Prediction

## 📌 Overview

This project implements a **K-Nearest Neighbors (K-NN) Regression** model for predicting **disease progression** (quantitative measure) using the **Diabetes dataset** from scikit-learn. The goal is to build, train, and evaluate a non-parametric regression model to accurately estimate diabetes progression based on various physiological features, serving as an introduction to instance-based learning algorithms in machine learning.

## 💾 Dataset

### Diabetes Dataset

  * **Description:** A classic regression dataset containing baseline measurements and one-year disease progression for diabetes patients. The dataset includes 10 features such as age, sex, body mass index (BMI), average blood pressure, and six blood serum measurements.
  * **Format:** Built-in dataset available directly through **scikit-learn**.
  * **Source:** [Diabetes Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
  * **Features:** 10 numerical features per sample (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6)
  * **Target:** Quantitative measure of disease progression one year after baseline (continuous regression target)
  * **Samples:** 442 instances

## 📂 Project Structure

```
05_KNN_Regression_Diabetes/
├── notebooks/
│   └── 01_KNN_Regression_Diabetes.ipynb  # Main data preprocessing, model building, training, and evaluation notebook
└── README.md
```

-----

## 🧠 Key Concepts & Methodology

The project follows a standard machine learning regression pipeline:

1.  **Data Loading:** Loading the dataset directly from `sklearn.datasets.load_diabetes`.
2.  **Data Preprocessing:**
      * **Train-Test Split:** Data is split into training and test sets using `train_test_split` with a random state of 42 for reproducibility.
      * **No Feature Scaling:** The K-NN algorithm works directly with the original feature values, though feature scaling can be beneficial in some cases.
3.  **K-NN Regression Model:** A non-parametric instance-based learning algorithm:
      * **Algorithm:** K-Nearest Neighbors Regressor
      * **n_neighbors:** 10 (number of neighbors to use for prediction)
      * **Weights:** Uniform (all neighbors weighted equally)
      * **Metric:** Minkowski distance (default)
4.  **Training:** The model stores all training data and makes predictions based on the average of the k nearest neighbors' target values.
5.  **Evaluation:** Model performance is assessed using **R-squared (R²) score**, which measures the proportion of variance in the target variable explained by the model.

-----

## ✅ Result Summary

The K-NN regression model demonstrates moderate performance on the Diabetes dataset with n_neighbors=10.

| Metric | Training Set | Test Set |
| :--- | :--- | :--- |
| **R-squared Score** | 0.51 | **0.46** |

The model achieves a **R-squared score of 0.46** on the test set, indicating that the model explains approximately 46% of the variance in disease progression. The visualization across different K values (1-50) helps identify the optimal number of neighbors, showing the trade-off between model complexity and generalization performance.

## 💭 Reflection

The results were not particularly good. This could indicate that the dataset may not be well-suited for K-NN regression, or that the model could be improved through better preprocessing, hyperparameter tuning, or feature engineering techniques.

-----

## 🔧 Tech Stack

| Library | Usage |
| :--- | :--- |
| **scikit-learn** | Machine learning library providing KNeighborsRegressor, dataset loading, and train-test splitting utilities. |
| **NumPy** | Fundamental array and mathematical operations for data manipulation. |
| **Matplotlib** | Data visualization for plotting model performance across different K values and analyzing the bias-variance trade-off. |

## ▶️ How to Run

1.  **Prerequisites:** Ensure you have Python and necessary libraries installed (preferably in a virtual environment).

2.  **Install Dependencies:**

    ```bash
    pip install numpy scikit-learn matplotlib
    ```

3.  **Data Setup:** The Diabetes dataset is automatically loaded when you run the notebook - no manual data setup required.

4.  **Execute:** Open and run the Jupyter Notebook:

    ```bash
    jupyter notebook notebooks/01_KNN_Regression_Diabetes.ipynb
    ```

-----

## 🚀 Future Improvements

  * **Hyperparameter Tuning:** Use **GridSearchCV** or **RandomizedSearchCV** to systematically explore optimal values for n_neighbors, weights (uniform vs. distance), and distance metrics.
  * **Feature Scaling:** Implement **StandardScaler** or **MinMaxScaler** to normalize features, which can significantly improve K-NN performance when features have different scales.
  * **Feature Engineering:** Explore feature interactions, polynomial features, or feature selection techniques to improve model performance.
  * **Cross-Validation:** Use K-fold cross-validation for more robust model evaluation and hyperparameter selection.
  * **Distance Metrics:** Experiment with different distance metrics (Manhattan, Euclidean, Minkowski with different p values) to find the best distance measure for this dataset.
  * **Weighted K-NN:** Implement distance-weighted K-NN where closer neighbors have more influence on the prediction than distant ones.
  * **Model Interpretation:** Analyze which features are most influential by examining the nearest neighbors and their feature contributions.
