# 🚀 04_Boston_Housing_NN: House Price Prediction

## 📌 Overview

This project implements a **Feedforward Neural Network (FNN)** for predicting **median house prices** in Boston neighborhoods using the **Boston Housing dataset**. The goal is to build, train, and evaluate a robust deep learning regression model to accurately estimate housing prices based on various neighborhood features, serving as a classic introduction to regression problems in deep learning.

## 💾 Dataset

### Boston Housing Dataset

  * **Description:** A classic regression dataset containing information about housing prices in Boston suburbs. The dataset includes 13 features such as crime rate, average number of rooms, accessibility to highways, and more.
  * **Format:** Built-in dataset available directly through **TensorFlow/Keras**.
  * **Source:** [Boston Housing Dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/boston_housing/load_data)
  * **Features:** 13 numerical features per sample
  * **Target:** Median value of owner-occupied homes in $1000's (continuous regression target)

## 📂 Project Structure

```
04_Boston_Housing_NN/
├── notebooks/
│   └── 01_Boston_Housing.ipynb  # Main data preprocessing, model building, and training notebook
└── README.md
```

-----

## 🧠 Key Concepts & Methodology

The project follows a standard deep learning regression pipeline:

1.  **Data Loading:** Loading the dataset directly from `tensorflow.keras.datasets.boston_housing`.
2.  **Data Preprocessing:**
      * **Normalization:** Features are normalized using **Z-score normalization** (mean subtraction and standard deviation division) based on training set statistics to ensure all features are on a similar scale.
      * **Train-Test Split:** Data is automatically split into training and test sets by the dataset loader.
      * **Validation Split:** During training, 10% of the training data is used for validation.
3.  **Neural Network Architecture:** A Sequential model with three fully connected layers:
      * **First Dense Layer** (64 neurons, ReLU activation, input shape: 13 features)
      * **Second Dense Layer** (64 neurons, ReLU activation)
      * **Output Dense Layer** (1 neuron, no activation for regression)
4.  **Training:** The model is compiled with the **RMSprop optimizer**, **Mean Squared Error (MSE)** loss function, and **Mean Absolute Error (MAE)** as an additional metric. Training is performed for 20 epochs with a batch size of 16.

-----

## ✅ Result Summary

The model demonstrates effective learning on the Boston Housing dataset after training for 20 epochs.

| Metric | Train (Last Epoch) | Validation (Last Epoch) |
| :--- | :--- | :--- |
| **Loss (MSE)** | 11.29 | **8.43** |
| **MAE** | 2.31 | **2.35** |

The model shows good **generalization ability** with validation loss decreasing consistently throughout training, indicating successful learning of the underlying patterns in the housing price data.

-----

## 🔧 Tech Stack

| Library | Usage |
| :--- | :--- |
| **TensorFlow / Keras** | Core deep learning framework for model definition, compilation, and training. Includes built-in Boston Housing dataset. |
| **NumPy** | Fundamental array and mathematical operations for data manipulation and normalization. |
| **Matplotlib** | Data visualization for plotting training and validation loss curves. |

## ▶️ How to Run

1.  **Prerequisites:** Ensure you have Python and necessary libraries installed (preferably in a virtual environment).

2.  **Install Dependencies:**

    ```bash
    pip install numpy tensorflow matplotlib
    ```

3.  **Data Setup:** The Boston Housing dataset is automatically downloaded when you run the notebook - no manual data setup required.

4.  **Execute:** Open and run the Jupyter Notebook:

    ```bash
    jupyter notebook notebooks/01_Boston_Housing.ipynb
    ```

-----

## 🚀 Future Improvements

  * **Regularization:** Introduce **Dropout** layers or **L1/L2 regularization** to further mitigate potential overfitting.
  * **Feature Engineering:** Explore feature interactions, polynomial features, or feature selection to improve model performance.
  * **Advanced Architectures:** Experiment with deeper networks, batch normalization, or different activation functions.
  * **Optimization:** Implement **Learning Rate Scheduling**, early stopping callbacks, or hyperparameter tuning (e.g., different optimizers, learning rates, batch sizes).
  * **Cross-Validation:** Use K-fold cross-validation for more robust model evaluation.
  * **Model Interpretation:** Analyze feature importance to understand which factors most influence house prices.
