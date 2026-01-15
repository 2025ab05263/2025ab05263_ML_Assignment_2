# Customer Churn Prediction - ML Assignment 2

# Customer Churn Prediction – Machine Learning Assignment

---

## a. Problem Statement

Customer churn refers to the situation where customers stop using a company’s services. Predicting churn is important for businesses as retaining existing customers is more cost-effective than acquiring new ones.

The objective of this project is to build, evaluate, and compare multiple machine learning classification models to predict whether a customer will churn or not based on historical customer data. An interactive Streamlit web application is developed to demonstrate predictions and compare model performance.

---

## b. Dataset Description 

- **Dataset Name**: Customer Churn Dataset  
- **Source**: Provided as part of the assignment  
- **Total Records**: 3,150 customer records  
- **Target Variable**: `Churn`  
  - `1` → Customer churned  
  - `0` → Customer did not churn  

### Feature Description:
- **Numerical Features**: Call failures, subscription length, charge amount, usage statistics  
- **Categorical Features**: Tariff plan, customer status, age group  
- **Target Feature**: Churn (Binary Classification)

### Preprocessing Steps:
- Missing values were removed
- Categorical features were encoded using Label Encoding
- Numerical features were scaled using StandardScaler
- Dataset split into 80% training and 20% testing using stratified sampling

---

## c. Models Used 

The following six machine learning classification models were implemented and evaluated using the same preprocessing pipeline:

1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

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
| Logistic Regression | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| Decision Tree | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| kNN | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| Naive Bayes | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| Random Forest (Ensemble) | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| XGBoost (Ensemble) | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |



---

## d. Observations on Model Performance

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression |  |
| Decision Tree |  |
| kNN |  |
| Naive Bayes |  |
| Random Forest (Ensemble) |  |
| XGBoost (Ensemble) |  |

---

## Conclusion



---
