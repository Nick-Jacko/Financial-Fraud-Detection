# Financial-Fraud-Detection Project Description 

# Question 1: Can a model predict whether a transaction is fraudulent?

## Goal
Use machine learning to detect fraudulent financial transactions in real-time and help financial institutions mitigate financial losses.

##  Dataset
- **Source**: [Kaggle - Financial Fraud Detection Dataset](https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset)
- **Rows**: 6,362,620
- **Columns**: 11 (plus engineered features)
- **Train/Test Split**: 80/20 → 5,090,096 training & 1,272,524 testing rows

### Target Variable
- `isFraud`: 1 if transaction is fraudulent, 0 otherwise

### Sample Features

| Feature                     | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `step`                      | Time step (1–744 hours)                                                     |
| `type`                      | Type of transaction                                                         |
| `amount`                    | Amount of transaction                                                       |
| `oldbalanceOrg`             | Balance before transaction (sender)                                        |
| `newbalanceOrig`            | Balance after transaction (sender)                                         |
| `oldbalanceDest`            | Balance before transaction (receiver)                                      |
| `newbalanceDest`            | Balance after transaction (receiver)                                       |
| `amount_oldbalanceOrg_ratio` | % of sender's money moved                                                 |
| `isCompleteTransfer`        | Flag for accounts zeroed after transaction                                 |
| `errorBalance`              | Discrepancy between expected and actual balances                           |
| `type_fraud_rate`           | Historical fraud rate by transaction type                                  |

## Models Used

| Model               | Description                                              |
|---------------------|----------------------------------------------------------|
| Logistic Regression | Baseline binary classifier                               |
| Random Forest       | Handles high-dimensional data, good against overfitting  |
| XGBoost             | Boosted decision trees for high-performance classification|

## Hyperparameters

### Logistic Regression
- `class_weight='balanced'`
- `solver='liblinear'`
- `max_iter=1000`

### Random Forest
- `n_estimators=100`
- `class_weight='balanced'`

### XGBoost
- `eval_metric='logloss'`
- `scale_pos_weight=50`

## Results

| Model               | Precision | Recall   | F1-Score | AUC   |
|---------------------|-----------|----------|----------|--------|
| Logistic Regression | 0.0307    | 0.8716   | 0.059    | -      |
| Random Forest       | >0.99     | >0.99    | >0.99    | 0.999  |
| XGBoost             | >0.99     | >0.99    | >0.99    | 0.999  |

## Confusion Matrix Summary

| Model         | False Positives | False Negatives |
|---------------|------------------|------------------|
| Random Forest | 6                | 4                |
| XGBoost       | 8                | 4                |

## Conclusion
Random Forest is the preferred model due to high accuracy, fast training time, and strong performance with imbalanced data. Feature engineering and class weighting significantly enhanced performance.
