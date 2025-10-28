# Detecting Credit Card Fraud with Machine Learning

---

## Project Goal
This project, developed for the **MSDS 5506 Data Mining Project**, focuses on building a machine learning classifier to detect **credit card fraud**.

---

## Data Source and Characteristics

### Data Acquisition
The dataset used for this analysis is the **Credit Card Fraud Detection** dataset.
* **Source:** KaggleHub dataset `mlg-ulb/creditcardfraud`.
* **File:** `creditcard.csv`.

### Data Structure
* **Total Entries:** 284,807 transactions.
* **Missing Values:** The dataset contains **no null values**.
* **Features:** The dataset includes 31 columns:
    * **Time** and **Amount:** Transaction time (seconds since first transaction) and transaction amount. These are not scaled.
    * **V1 to V28:** Principal Component Analysis (PCA) components, which are the anonymized and scaled features resulting from a dimensionality reduction technique.
    * **Class:** The target variable (`0` for Non-Fraudulent, `1` for Fraudulent).

### Class Imbalance
The dataset exhibits a severe class imbalance, which is a major challenge for this classification problem:
* **Non-Fraudulent (`Class = 0`):** 284,315 transactions (99.827%).
* **Fraudulent (`Class = 1`):** 492 transactions (0.173%).

---

## Methodology

### Preprocessing Strategy
To address the severe class imbalance, the methodology employs a hybrid sampling technique:
* **Sampling Method:** The `SMOTETomek` method from the `imblearn` library is used. This combines **oversampling** (SMOTE) with **undersampling** (Tomek links) to generate synthetic samples of the minority class while cleaning the boundaries between classes.

### Modeling
[needs updating]

---

## Requirements
The project relies on standard data science and machine learning libraries:
* `pandas`
* `numpy`
* `kagglehub` (for data download)
* `sklearn` (for model selection)
* `imblearn` (specifically `SMOTETomek` for handling imbalance)
* `seaborn` and `matplotlib` (for visualization)
