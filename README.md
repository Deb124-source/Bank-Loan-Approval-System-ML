## Bank-Loan-Approval-ML

This project predicts whether a bank loan will be approved or not using a Machine Learning model trained on customer demographic and financial details. It helps banks automate loan screening, saving time and improving consistency in decision-making.

## Objective

To build a predictive model that classifies loan applications as Approved (Y) or Rejected (N) based on features like income, education, credit history, and employment.

## Model Description

Algorithm Used: Logistic Regression 
Why: Offers transparency, high recall, and interpretability—ideal for financial decision systems.

## Performance

 ✓ F1-Score: 99.62% (balanced performance)
  
 ✓ Accuracy: 99.53% (overall correctness)
 
 ✓ Precision: 99.25% (low false approvals)
  
 ✓ Recall: 100.00% (captures most good applicants)


## Exploratory Data Analysis (EDA): 
Visualized loan approvals by education, income, and property area. Found credit history as the strongest predictor of approval.

## Data Preprocessing

Handled missing values and encoded categorical features. Created new engineered features: Total_Income, Income_to_Loan_Ratio.

Model Training & Evaluation

Trained multiple models and selected Logistic Regression for best F1-score.

## Prediction:

The model predicts “Y” (approved) or “N” (rejected) for new loan applications.

## Chosen Metric

F1-Score (99.62%) — chosen because it balances precision (avoiding bad loans) and recall (not missing good applicants). This ensures the bank’s decision system is both safe and inclusive.

## Application

Banks can integrate this model to: Automatically pre-screen applicants Reduce manual workload by 60–70% Make consistent, data-driven loan decisions Improve customer experience through faster processing

## Tech Stack
Language: Python
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn 
Environment: Google Colab
