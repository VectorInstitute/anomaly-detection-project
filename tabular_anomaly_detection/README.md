# BAF Demo

Welcome to Bank Account Fraud (BAF) demo where we provide two reference implementations to do a comprehensive analysis of this dataset and explore fraud detection in online bank account openings. In `BAF-demo-1.ipynb`, we will delve into a detailed description of the [BAF dataset](https://arxiv.org/pdf/2211.13358.pdf), which centers around detecting fraudulent applications in a large consumer bank and explore machine learning-based methods for fraud detection. In `BAF-demo-2.ipynb`, we will further dive into deep learning-based methodologies applied to the same dataset.

## Dataset Overview

The BAF dataset focuses on detecting fraudulent online bank account opening applications in a large consumer bank. Fraudsters attempt to impersonate someone or create fictional individuals to gain access to banking services and carry out illicit activities. This dataset was accepted at NeurIPS 2022.

The dataset consists of individual applications, with each row representing an application made on an online platform. The label indicating fraud or legitimacy is stored in the "is_fraud" column, where a positive instance represents a fraudulent attempt and a negative instance represents a legitimate application. The dataset spans eight months from February to September, with varying fraud prevalence figures and distribution of applications.

## Data Generation and Privacy Preservation

To produce the dataset, CTGAN models are trained with some original features. These features are the top thirty most important features selected from the top-performing LightGBM models, considering expressiveness, interpretability, and redundancy. Differential privacy is enforced by perturbing each column in the original dataset using a Laplacian noise mechanism. Additional obfuscation measures are applied to certain applicant data, such as age and income categorization, to enhance privacy.

The CTGAN models are trained on the perturbed dataset with the selected features, and the dataset is augmented with a column representing the month of application to incorporate temporal information. A total of 70 trained CTGAN models are created, and the generative models are evaluated based on predictive performance metrics and statistical similarity between the real and generated data.

## Performance and Fairness Metrics

Performance metrics focus on defining a relevant threshold and metric for fraud detection, specifically targeting a 5% false positive rate (FPR) and measuring the true positive rate (TPR) at that point. This metric strikes a balance between detecting fraud and minimizing customer attrition. we also calculate the Area Under the Receiver Operating Characteristic Curve (AUROC) for the provided data as another metric of performance.

Fairness metrics aim to ensure that the probability of a legitimate application being wrongly classified as fraudulent is independent of the sensitive attribute value. The fairness ratio is calculated by comparing the FPRs of different groups, emphasizing predictive equality.

## Dataset Variants

The BAF dataset includes six dataset variants, each with pre-determined and controllable bias patterns. These variants introduce disparities in group sizes, prevalence, and separability to stress test predictive performance and fairness.

# Dataset Preparation

The `BAFDataset` class provides a convenient way to load, preprocess, and split the BAF dataset for use in machine learning experiments. The class contains three primary functions that work together to load the data, split it into train and test sets, and preprocess the categorical features via one-hot encoding.

- `load_baf(file_path)`: This function loads the specified subset of the BAF dataset as a pandas DataFrame. The available subsets are 'Base', 'Variant I', 'Variant II', 'Variant III', 'Variant IV', and 'Variant V'.
- `train_test_split(df, month)`: This function splits the BAF dataset into train and test sets based on the specified month.
- `one_hot_encode_categorical(X_train, X_test)`: This function preprocesses the categorical features in the BAF dataset by one-hot encoding them.

The `BAFDataset` class streamlines the process of preparing the BAF dataset for machine learning experiments, handling loading, train-test splitting, and preprocessing tasks in a clear and organized manner.

# Models for Anomaly Detection

## Logistic Regression (Supervised)

Logistic regression is a popular statistical model used for binary classification tasks. It estimates the probability that an observation belongs to a certain class based on a set of input features.

## Random Forest (Supervised)

Random forest is a powerful ensemble learning algorithm used for both classification and regression tasks. It combines multiple decision trees to make predictions.

## XGBoost (Supervised)

XGBoost (Extreme Gradient Boosting) is an advanced implementation of gradient boosting, known for its exceptional performance and efficiency.

## CatBoost (Supervised)

CatBoost is a gradient boosting framework developed by Yandex that is particularly well-suited for working with categorical features.

## Light GBM (Supervised)

Light GBM (Light Gradient Boosting Machine) is a gradient boosting framework known for its high performance and efficiency for machine learning tasks.

## TabNet (Supervised and Semi-supervised)

TabNet presents an innovative approach that bridges the gap between Decision Trees (DTs) and Deep Neural Networks (DNNs).

## Autoencoder (AE) (Unsupervised)

The autoencoder is a neural network designed to learn a compressed representation of input data and reconstruct the original data from this representation.

## Isolation Forest (IF) (Unsupervised)

The Isolation Forest algorithm is used for spotting anomalies within datasets. It constructs a forest of isolation trees where data points are partitioned based on random feature choices and values.

## ICL (Unsupervised)

ICL focuses on out-of-class sample detection in tabular data, aiming to capture the structure of single training class samples by learning mappings that maximize mutual information between each sample and the masked-out portion.

# Model Performance and Fairness Results

Here are the results for various models and their performance metrics on `Base` variant of BAF dataset:

| Model               | TPR     | FPR     | AUROC  | Fairness Ratio | Training Time | Inference Time |
|---------------------|---------|---------|--------|----------------|---------------|-----------------|
| Logistic Regression | 0.2074  | 0.0497  | 0.6802 | 0.3455         | 18.675150     | 0.080438        |
| Random Forest       | 0.4288  | 0.0415  | 0.8398 | 0.3007         | 283.078872    | 3.608654        |
| XGBoost             | 0.5049  | 0.0498  | 0.8787 | 0.3233         | 307.644614    | 0.549125        |
| CatBoost            | 0.5101  | 0.0499  | 0.8836 | 0.2992         | 198.038225    | 0.373308        |
| Light GBM           | 0.5302  | 0.0500  | 0.8898 | 0.3017         | 13.898896     | 1.010416        |
| TabNet              | 0.5069  | 0.0500  | 0.8822 | 0.3459         | 666.011628    | 6.612230        |
| AE    | 0.0629  | 0.0498  | 0.6143 | 0.3692         | 50.658261     | 0.110126        |
| IF | 0.0806  | 0.0499 | 0.5816 | 0.2516         | 0.915280      | 1.485850        |
| ICL | 0.0865 | 0.0499 | 0.6172 | 0.9953         | 1137.885359   | 21.296762       |
