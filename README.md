# Customer Churn Prediction - ML Assignment 2
---

## Problem Statement

Customer churn refers to the situation where customers stop using a company's services. Predicting churn is important for businesses as retaining existing customers is more cost-effective than acquiring new ones.


---

## Dataset Description 

- **Dataset Name**: Customer Churn Dataset  
- **Source**: Provided as part of the assignment  
- **Total Records**: 3,150 customer records  
- **Training Set**: 2,520 records (80%)
- **Test Set**: 630 records (20%)
- **Class Distribution**: 
  - Non-churners (Class 0): 2,655 customers (84.29%)
  - Churners (Class 1): 495 customers (15.71%)
- **Imbalance Ratio**: 5.36:1 (non-churners to churners)
- **Target Variable**: `Churn`  
  - `1` → Customer churned  
  - `0` → Customer did not churn  

### Feature Description:
- **Numerical Features**: Call Failure, Complains, Subscription Length, Charge Amount, Seconds of Use, Frequency of use, Frequency of SMS, Distinct Called Numbers, Age, Customer Value  
- **Categorical Features**: Age Group, Tariff Plan, Status  
- **Total Features**: 13 predictive features
- **Target Feature**: Churn (Binary Classification)

### Preprocessing Steps:
- Missing values were removed
- Categorical features were encoded using Label Encoding
- Numerical features were scaled using StandardScaler
- Dataset split into 80% training and 20% testing using stratified sampling
- Class imbalance handled using balanced class weights for applicable models

---

## Models Used 

The following six machine learning classification models were implemented and evaluated using the same preprocessing pipeline:

1. **Logistic Regression** - Linear model with custom class weights (3:1)
2. **Decision Tree** - Tree-based model with custom class weights (3:1)
3. **k-Nearest Neighbors (kNN)** - Distance-based classifier (k=5)
4. **Naive Bayes (Gaussian)** - Probabilistic classifier
5. **Random Forest (Ensemble)** - Ensemble of 200 decision trees with balanced weights
6. **XGBoost (Ensemble)** - Gradient boosting with scale_pos_weight for imbalance

### Evaluation Metrics Used:
- Accuracy  
- AUC (ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 85.71% | 93.27% | 52.67% | 87.34% | 65.71% | 60.31% |
| Decision Tree | 86.31% | 93.55% | 53.68% | 92.41% | 67.91% | 63.54% |
| kNN | 95.24% | 96.46% | 87.67% | 81.01% | 84.21% | 81.50% |
| Naive Bayes | 75.40% | 90.01% | 38.10% | 91.14% | 53.73% | 47.77% |
| Random Forest (Ensemble) | 94.25% | 97.93% | 76.60% | 91.14% | 83.24% | 80.23% |
| XGBoost (Ensemble) | 95.63% | 98.89% | 88.00% | 83.54% | 85.71% | 83.18% |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Achieved 85.71% accuracy with excellent recall (87.34%) after applying custom class weights (3:1). However, precision is moderate (52.67%), resulting in more false positives. Good AUC (93.27%) indicates strong ranking ability. Suitable as a baseline model with interpretable coefficients. |
| Decision Tree | Improved performance with 86.31% accuracy and very high recall (92.41%) - the highest among all models. Custom class weights (3:1) significantly boosted minority class detection, though precision remains moderate (53.68%). The model is interpretable and provides decision rules, but may overfit without proper pruning. |
| kNN | Excellent overall performance with 95.24% accuracy and strong balance between precision (87.67%) and recall (81.01%). The high AUC (96.46%) and MCC (81.50%) indicate robust classification. Distance-based approach works well for this feature space without requiring class weight adjustments. |
| Naive Bayes | Lowest accuracy (75.40%) but tied for highest recall (91.14%), catching most churners at the cost of many false positives (38.10% precision). The independence assumption doesn't hold well for this dataset with correlated features. Useful only when minimizing false negatives is critical. |
| Random Forest (Ensemble) | Outstanding performance with 94.25% accuracy and exceptional AUC (97.93%). Balanced class weights enabled high recall (91.14%) - tied with Decision Tree and Naive Bayes - while maintaining good precision (76.60%). The ensemble approach handles feature interactions well and provides robust predictions with feature importance insights. |
| XGBoost (Ensemble) | **Best overall model** with 95.63% accuracy, highest AUC (98.89%), and best balance of precision (88.00%) and recall (83.54%). The MCC of 83.18% confirms excellent performance on this imbalanced dataset. Using scale_pos_weight (5.36) effectively captures complex patterns while handling class imbalance. |

---

## Project Structure

```
2025ab05263_ML_Assignment_2/
├── Customer+Churn.csv          # Training dataset
├── Customer+Churn_test.csv     # Test dataset
├── train_models.py             # Model training script
├── test_models.py              # Model testing script
├── app.py                      # Streamlit web application
├── model/                      # Saved trained models
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl             # Fitted StandardScaler
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

