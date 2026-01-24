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
- **Total Records**: 2,520 customer records  
- **Training Set**: 2,016 records (80%)
- **Test Set**: 504 records (20%)
- **Class Distribution**: 84.3% non-churners (2,124) vs 15.7% churners (396)
- **Imbalance Ratio**: 5.36:1
- **Target Variable**: `Churn`  
  - `1` → Customer churned  
  - `0` → Customer did not churn  

### Feature Description:
- **Numerical Features**: Call failures, subscription length, charge amount, seconds of use, frequency of use, frequency of SMS, distinct called numbers, age, customer value  
- **Categorical Features**: Tariff plan, customer status, age group  
- **Total Features**: 13 predictive features
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
| Logistic Regression | 88.29% | 92.86% | 75.00% | 37.97% | 50.42% | 47.91% |
| Decision Tree | 92.66% | 93.89% | 90.38% | 59.49% | 71.76% | 69.70% |
| kNN | 95.24% | 96.46% | 87.67% | 81.01% | 84.21% | 81.50% |
| Naive Bayes | 75.40% | 90.01% | 38.10% | 91.14% | 53.73% | 47.77% |
| Random Forest (Ensemble) | 95.04% | 98.49% | 89.71% | 77.22% | 82.99% | 80.42% |
| XGBoost (Ensemble) | 95.63% | 98.89% | 88.00% | 83.54% | 85.71% | 83.18% |



---

## d. Observations on Model Performance

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Achieved 88.29% accuracy with good AUC (92.86%), but suffers from low recall (37.97%), meaning it misses many churners. The linear decision boundary is insufficient for this complex dataset. Best used as a baseline model. |
| Decision Tree | Shows improved performance with 92.66% accuracy and excellent precision (90.38%). However, recall is still moderate (59.49%), indicating it misses some churners. The model provides good interpretability but may overfit without proper pruning. |
| kNN | Excellent overall performance with 95.24% accuracy and strong balance between precision (87.67%) and recall (81.01%). The high AUC (96.46%) and MCC (81.50%) indicate robust classification. Distance-based approach works well for this feature space. |
| Naive Bayes | Lowest accuracy (75.40%) but highest recall (91.14%), catching most churners at the cost of many false positives (38.10% precision). The independence assumption doesn't hold well for this dataset. Useful when minimizing false negatives is critical. |
| Random Forest (Ensemble) | Outstanding performance with 95.04% accuracy and exceptional AUC (98.49%). High precision (89.71%) but slightly lower recall (77.22%) compared to kNN. The ensemble approach handles feature interactions well and provides robust predictions. |
| XGBoost (Ensemble) | **Best overall model** with 95.63% accuracy, highest AUC (98.89%), and best balance of precision (88.00%) and recall (83.54%). The MCC of 83.18% confirms excellent performance on this imbalanced dataset. Gradient boosting effectively captures complex patterns. |

---

## Conclusion

This project successfully implemented and compared six machine learning models for customer churn prediction. The dataset contained 2,520 customer records with a class imbalance ratio of 5.36:1 (non-churners to churners), which presented a significant challenge for model training.

**Key Findings:**

1. **Best Performing Model**: **XGBoost** emerged as the top performer with 95.63% accuracy, 98.89% AUC, and the best balance between precision (88.00%) and recall (83.54%). Its MCC of 83.18% confirms excellent performance on this imbalanced dataset.

2. **Ensemble Methods Dominate**: Both ensemble models (Random Forest and XGBoost) significantly outperformed traditional algorithms, demonstrating the value of ensemble learning for complex classification tasks.

3. **Trade-offs Between Metrics**: 
   - Naive Bayes achieved the highest recall (91.14%) but lowest precision (38.10%), making it suitable only when minimizing false negatives is paramount.
   - Logistic Regression showed poor recall (37.97%), missing too many churners to be practical.
   - kNN and Random Forest provided excellent alternatives to XGBoost with >95% accuracy.

4. **Class Imbalance Handling**: Despite the 5.36:1 imbalance, the top three models (XGBoost, kNN, Random Forest) maintained strong recall (>77%), indicating effective handling of minority class prediction through stratified sampling and appropriate model selection.

**Recommendations:**
- Deploy **XGBoost** for production use due to its superior overall performance
- Consider ensemble voting between XGBoost, Random Forest, and kNN for even more robust predictions
- The Streamlit application provides an intuitive interface for real-time churn predictions and model comparison

**Business Impact**: With 95.63% accuracy and 83.54% recall, the XGBoost model can identify approximately 84% of potential churners, enabling proactive retention strategies and significant cost savings compared to customer acquisition.

---
