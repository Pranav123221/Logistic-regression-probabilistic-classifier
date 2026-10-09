# Logistic Regression Probabilistic Classifier

**An Interactive Machine Learning Experimentation Platform**

An interactive web application for exploring Logistic Regression through configurable hyperparameters, optimization algorithms, probability estimation, and dynamic classification visualizations.

Built with Python, Scikit-learn, Streamlit, NumPy, Pandas, and Matplotlib.



---

## Table of Contents

* Overview
* Core Capabilities
* Mathematical Foundation
* Hyperparameter Configuration
* Optimization Solver Analysis
* Binary and Multiclass Classification
* Interactive Visualization
* Model Evaluation
* Technology Stack
* Installation and Execution
* Project Structure
* Learning Outcomes
* Future Enhancements

## Overview

Logistic Regression is a supervised learning algorithm used to estimate class probabilities and perform classification. Despite its name, it is a classification algorithm rather than a conventional regression model.

This project transforms the algorithm into an interactive experimentation environment where users can explore how model configuration influences the fitted classifier.

Instead of treating Logistic Regression as a black-box estimator, the application encourages experimentation with regularization strength, optimization strategies, convergence settings, class-handling approaches, and feature distributions.

The objective is to connect the mathematical foundations of Logistic Regression with practical machine learning implementation and visual interpretation.

## Core Capabilities

* Interactive model configuration and hyperparameter experimentation.
* Configurable optimization solvers and regularization settings.
* Binary and multiclass classification experiments, where supported.
* Probability estimation and class prediction.
* Feature-distribution and decision-boundary visualization.
* Analysis of model coefficients and intercepts, if exposed by the interface.
* Classification performance evaluation.
* Interactive web interface built with Streamlit.

## Mathematical Foundation

### 1. Linear Decision Function

Logistic Regression first computes a linear combination of input features:

$$
z = w_0 + w_1x_1 + w_2x_2 + \cdots + w_nx_n
$$

Where:

* \(x_i\): input features
* \(w_i\): learned feature coefficients
* \(w_0\): intercept or bias
* \(z\): linear decision score

### 2. Sigmoid Probability Function

For binary classification, the sigmoid function maps the decision score to a probability between zero and one.

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The estimated probability of the positive class is:

$$
P(y=1\mid X)=\sigma(w^TX+b)
$$

A classification threshold converts the estimated probability into a class label. The default is commonly 0.5, but the threshold can be changed when the application supports it.

### 3. Log-Loss and Regularization

The model learns coefficients by minimizing a logistic loss objective, typically with regularization.

$$
J(w)=-\frac{1}{m}\sum_{i=1}^{m}
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
+\lambda R(w)
$$

Here, \(R(w)\) represents the selected regularization term and \(\lambda\) its strength. Scikit-learn expresses regularization strength using the inverse parameter `C`.

## Hyperparameter Configuration

### Regularization Strength — `C`

`C` controls the inverse of regularization strength.

* **Small `C`:** Stronger regularization; coefficients are penalized more heavily.
* **Large `C`:** Weaker regularization; the model has more freedom to fit the training data.

Experimenting with `C` helps illustrate the trade-off between model complexity and generalization.

### Regularization Type

| Setting           | Purpose                                                                |
| ----------------- | ---------------------------------------------------------------------- |
| L1                | Encourages sparse coefficients and may eliminate less useful features. |
| L2                | Penalizes large coefficients and is a common default.                  |
| Elastic-Net       | Combines L1 and L2 regularization.                                     |
| No regularization | Removes the regularization penalty where supported.                    |

### Additional Parameters

| Parameter       | Role                                                                  |
| --------------- | --------------------------------------------------------------------- |
| `solver`        | Optimization algorithm used to fit the model.                         |
| `dual`          | Selects the dual formulation where supported.                         |
| `max_iter`      | Maximum number of optimization iterations.                            |
| `tol`           | Convergence tolerance.                                                |
| `l1_ratio`      | Controls the L1/L2 mixture for compatible Elastic-Net configurations. |
| `fit_intercept` | Determines whether an intercept is included.                          |
| `class_weight`  | Allows class weighting for imbalanced classification datasets.        |
| `random_state`  | Controls applicable sources of randomness.                            |

Parameter availability and compatibility depend on the Scikit-learn version and selected solver.

## Optimization Solver Analysis

The solver determines how the model's optimization problem is solved.

| Solver            | Technical characteristics                                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `lbfgs`           | Quasi-Newton optimization; a useful general-purpose choice.                                                               |
| `liblinear`       | Useful for smaller binary-classification problems; supports compatible L1/L2 configurations.                              |
| `newton-cg`       | Newton-based optimization using gradient and curvature information.                                                       |
| `newton-cholesky` | Hessian-based optimization; can suit datasets with many samples relative to features, but may require substantial memory. |
| `sag`             | Stochastic Average Gradient; often effective on large, appropriately scaled datasets.                                     |
| `saga`            | An extension of SAG supporting additional regularization configurations, including Elastic-Net.                           |

**Important:** Solvers are not interchangeable across every configuration. For example, Elastic-Net requires `saga`, and `dual=True` is supported only with the L2 penalty and `liblinear` in the documented implementation. Multiclass support also varies.

The application should only offer valid parameter combinations for its installed Scikit-learn version.

## Binary and Multiclass Classification

### Binary Classification

The sigmoid function estimates the probability of the positive class. A decision threshold determines the predicted label.

### Multiclass Classification

Two common approaches are:

* **One-vs-Rest (OvR):** Trains a binary classifier for each class against the remaining classes.
* **Multinomial Logistic Regression:** Models class probabilities jointly using the softmax function.

$$
P(y=k\mid X)=
\frac{e^{z_k}}{\sum_j e^{z_j}}
$$

The supported approach depends on the solver and Scikit-learn version. In current versions, multiclass behavior is generally handled automatically rather than through the older `multi_class` parameter.

## Interactive Visualization

### 1. Feature Distribution

Explore how input features are distributed and how their values relate to the target classes.

### 2. Decision Boundary and Feature Space

For two selected features, a two-dimensional decision-boundary plot can illustrate how the classifier separates the classes.

Changing supported model parameters and retraining the estimator can update the predicted regions and boundary.

### 3. Probability Analysis

Inspect estimated class probabilities to understand the model's confidence in its predictions. A probability estimate is not automatically a guarantee of correctness or calibration.

### 4. Coefficient and Intercept Analysis

Where implemented, inspect learned coefficients and the intercept to understand how input features contribute to the linear decision function.

### 5. Hyperparameter Experiments

Compare model behavior under different settings for regularization, optimization, convergence, and classification strategy.

*Visualizations should reflect the actual controls and charts available in the deployed application.*

## Model Evaluation

Depending on the implemented evaluation panel, classification performance can be examined using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* ROC curve and ROC-AUC
* Log-loss

Metrics should be calculated on appropriate held-out data. For imbalanced datasets, accuracy alone may not adequately describe model performance.

## Technology Stack

| Technology     | Application                         |
| -------------- | ----------------------------------- |
| Python         | Core implementation                 |
| Scikit-learn   | Model training and evaluation       |
| NumPy          | Numerical operations                |
| Pandas         | Data processing                     |
| Matplotlib     | Data and model visualization        |
| Streamlit      | Interactive web application         |
| Git and GitHub | Version control and project hosting |

## Installation and Execution

### Clone the Repository

```bash
git clone https://github.com/Pranav123221/logistic-regression-probabilistic-classifier.git
cd logistic-regression-probabilistic-classifier
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### Install Dependencies

If the project contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise:

```bash
pip install streamlit scikit-learn numpy pandas matplotlib
```

### Launch the Application

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit in your browser.

## Project Structure

A possible project structure is:

```text
logistic-regression-probabilistic-classifier/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   └── screenshots/
└── .gitignore
```

Adapt this structure to match the files actually present in your repository.

## Learning Outcomes

This project provides hands-on experience with:

* Probabilistic classification and sigmoid-based inference.
* Regularized optimization and model complexity.
* Solver selection and convergence behavior.
* Hyperparameter experimentation.
* Binary and multiclass classification strategies.
* Feature-space interpretation and decision boundaries.
* Interactive machine learning application development.



## Author

**Pranav Sharma**



This project is part of a hands-on Machine Learning learning journey focused on understanding algorithms through implementation, experimentation, and visualization.

## License

MIT 
