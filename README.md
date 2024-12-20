# **Credit Score Prediction System**

## **Project Overview**
The **Credit Score Prediction System** is a machine learning solution designed to predict the credit scores of individuals based on their financial, behavioral, and demographic attributes. This project utilizes a **Random Forest Classifier** to make accurate predictions and deploys the model using a **Flask API** for real-time interaction. Financial institutions can leverage this system to assess creditworthiness and make data-driven lending decisions.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Deployment](#deployment)
---

## **Features**
- Predicts credit scores based on customer data using a **Random Forest Classifier**.
- RESTful **Flask API** for real-time predictions.
- Comprehensive **Exploratory Data Analysis (EDA)** to understand and clean the data.
- Handles missing values, outliers, and feature engineering for better model performance.
- Supports deployment-ready architecture for integration into production systems.

---

## **Technologies Used**
- **Programming Language:** Python
- **Libraries:** scikit-learn, pandas, numpy, Flask, matplotlib, seaborn
- **Version Control:** Git
- **Deployment:** Flask API
- **Data Visualization:** Matplotlib, Seaborn

---

## **Data Preprocessing**

### Key Steps:
1. **Data Cleaning:**
   - Handled missing values using median imputation and KNN imputation.
   - Removed or capped outliers to reduce skewness.
   - Converted `Credit_History_Age` from textual to numerical format.
2. **Feature Engineering:**
   - Encoded categorical features (`Credit_Mix`, `Payment_Behaviour`) using **One-Hot Encoding**.
   - Created new features, such as `Debt_to_Income_Ratio` and `Savings_to_Income_Ratio`.
3. **Scaling:**
   - Standardized numerical features for model compatibility.

### Insights from EDA:
- **Age:** Majority of individuals are in their 20s to 40s.
- **Income:** Most customers earn under $40,000 annually, with outliers above $140,000.
- **Credit Utilization:** Balanced usage, with most values between 20%-50%.
- **Multicollinearity:** High correlation between `Annual_Income` and `Monthly_Inhand_Salary`.

---

## **Model Training**

### Steps:
1. **Model Selection:** Chose **Random Forest Classifier** for its robustness and ability to handle complex datasets.
2. **Evaluation Metrics:**
   - Accuracy: Achieved **85%** on test data.
   - Precision, Recall, F1-Score: Evaluated using a classification report.
3. **Hyperparameter Tuning:**
   - Optimized `n_estimators`, `max_depth`, and `min_samples_split` using **RandomizedSearchCV**.
4. **Final Model:** Saved as `best_random_forest_model.joblib`.

---

## **Deployment**

The trained model is deployed using a **Flask API**, enabling real-time predictions.

### API Endpoint:
- **POST /predict**
  - **Input:** JSON payload with customer attributes.
  - **Output:** Predicted credit score.

### Sample Request:
```json
{
  "Age": 30,
  "Annual_Income": 50000,
  "Monthly_Inhand_Salary": 4000,
  "Credit_Utilization": 30,
  "Outstanding_Debt": 2000,
  "Num_of_Loan": 1,
  "Credit_Mix": "Good",
  "Payment_Behaviour": "Regular",
  "Type_of_Loan": "Personal"
}
```

### Sample Response:
```json
{
  "success": true,
  "prediction": 750
}
```
