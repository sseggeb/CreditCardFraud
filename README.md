# Detecting Credit Card Fraud with Machine Learning
## MSDS 5506 Data Mining Project

This project applies various machine learning models to a severely imbalanced dataset to detect fraudulent credit card transactions, with a focus on optimizing for **Recall**.

## I. Data Understanding

### Data Source
The dataset, `creditcard.csv`, was imported from Kaggle using `kagglehub`.

### Key Characteristics
* **Total Entries:** 284,807 transactions.
* **Missing Data:** No null values were found.
* **Features:** The dataset includes time (seconds elapsed since the first transaction), amount, and 28 principal components (`V1` to `V28`) obtained via PCA for confidentiality. The `V` features are already scaled.
* **Target Variable (`Class`):**
    * **Non-Fraud (0):** 284,315 transactions (99.83% of the data).
    * **Fraud (1):** 492 transactions (0.17% of the data).
    * (Severe Class Imbalance) * **Feature Insights:**
    * The distribution of transaction `Time` for fraudulent transactions appears more uniform compared to non-fraudulent ones.
    * Fraudulent transactions tend to involve smaller `Amount` values. 
## II. Data Preprocessing

The primary preprocessing challenge was the **severe class imbalance**.

### Handling Imbalance
The notebook explored techniques to re-balance the training data:
1.  **SMOTETomek (Oversampling & Undersampling):** A combination of synthetic minority oversampling (SMOTE) and Edited Nearest Neighbors (ENN) undersampling.
    * The resampled training set achieved a 1:1 ratio: `226602` non-fraud (0) and `226602` fraud (1).
2.  **RandomUnderSampler (Undersampling):** A simple random selection to match the minority class count.
    * The resampled training set achieved a 1:1 ratio: `378` non-fraud (0) and `378` fraud (1).

The data was then split into training and testing sets using `train_test_split`.

## III. Data Modeling

The modeling phase focused on algorithms suitable for optimizing Recall, including Logistic Regression, Ensemble Methods, and Neural Networks. The models were trained on the SMOTETomek resampled data.

### 1. Logistic Regression (Baseline)

A baseline Logistic Regression model was trained using the SMOTETomek resampled data.

| Metric | Score |
| :--- | :--- |
| **Recall** | 0.9053 |
| **Precision** | 0.0512 |
| **F1-Score** | 0.0969 |

**Confusion Matrix (Test Set):**
| | Actual 0 (Non-Fraud) | Actual 1 (Fraud) |
| :--- | :--- | :--- |
| **Predicted 0** | 55057 (True Negatives) | 1594 (False Positives) |
| **Predicted 1** | 9 (False Negatives) | 86 (True Positives) |

*Insight:* The model achieved high **Recall (90.53%)** but suffered from very low Precision, flagging a large number of valid transactions as fraudulent.

### 2. Random Forest Classifier

A Random Forest Classifier with `class_weight='balanced'` was trained to improve performance, particularly precision, using the SMOTETomek resampled data.

| Metric | Score |
| :--- | :--- |
| **Recall** | 0.8316 |
| **Precision** | 0.9405 |
| **F1-Score** | 0.8827 |

*Insight:* The Random Forest model achieved a much better balance between Recall and Precision, resulting in the highest F1-Score of all models.

**Confusion Matrix (Test Set):**
| | Actual 0 (Non-Fraud) | Actual 1 (Fraud) |
| :--- | :--- | :--- |
| **Predicted 0** | 56646 (True Negatives) | 5 (False Positives) |
| **Predicted 1** | 16 (False Negatives) | 79 (True Positives) |

### 3. XGBoost Classifier

The XGBoost Classifier was trained and configured with a `scale_pos_weight` of **2.00** to account for class imbalance during training on the SMOTETomek resampled data.

| Metric | Score |
| :--- | :--- |
| **Recall** | 0.8421 |
| **Precision** | 0.7407 |
| **F1-Score** | 0.7882 |

**Confusion Matrix (Test Set):**
| | Actual 0 (Non-Fraud) | Actual 1 (Fraud) |
| :--- | :--- | :--- |
| **Predicted 0** | 56623 (True Negatives) | 28 (False Positives) |
| **Predicted 1** | 15 (False Negatives) | 80 (True Positives) |

### 4. Precision-Recall AUC and Threshold Tuning

To select the best model and optimize for a specific goal (e.g., a minimum recall), the notebook compared models using the Precision-Recall Area Under the Curve (PR AUC) and demonstrated decision threshold adjustment.

| Model | PR AUC Score |
| :--- | :--- |
| **Random Forest** | 0.8862 |
| **XGBoost** | 0.8672 |
| **Logistic Regression** | 0.7844 |

* **Model Selection:** The **Random Forest Classifier** was the best overall performer, achieving the highest PR AUC and a strong balance of Recall and Precision.

#### Decision Threshold Adjustment Example (XGBoost)
The threshold was tuned to achieve a target Recall of **0.86** on the XGBoost model.

* **Optimal Threshold:** 0.6726
* **Tuned Recall:** 0.8632
* **Tuned Precision:** 0.8542
* **Tuned F1-Score:** 0.8586

**Tuned Confusion Matrix:**
| | Actual 0 (Non-Fraud) | Actual 1 (Fraud) |
| :--- | :--- | :--- |
| **Predicted 0** | 56637 (True Negatives) | 14 (False Positives) |
| **Predicted 1** | 13 (False Negatives) | 82 (True Positives) |

*Insight:* By adjusting the decision threshold, the XGBoost model was tuned to prioritize higher recall (86.32%) while maintaining a significantly improved precision (85.42%) compared to its default threshold performance (74.07%).
