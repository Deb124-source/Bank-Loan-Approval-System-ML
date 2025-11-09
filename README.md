# 🏦 Bank Loan Approval System

## 📘 Overview
The **Bank Loan Approval System** is a machine learning project that predicts whether a loan application should be **approved or rejected** based on customer details such as income, credit history, and loan amount.  
It uses multiple classification algorithms to evaluate performance and determine the most accurate model.

---

## 🎯 Objectives
- Analyze bank loan applicant data.  
- Preprocess missing and categorical values.  
- Split the data into training and testing sets.  
- Train and compare multiple ML models.  
- Evaluate model performance using accuracy and approval rate.  

---

## 🧠 Machine Learning Models Used
The notebook initializes and evaluates several algorithms:
- **Logistic Regression**  
- **Decision Tree Classifier**  
- **Random Forest Classifier**  
- **K-Nearest Neighbors (KNN)**  

Each model is trained and tested using an **80-20 data split**, ensuring balanced class distribution through stratified sampling.

---

## ⚙️ Key Steps
1. **Data Loading:** Import dataset into a pandas DataFrame.  
2. **Exploratory Data Analysis (EDA):** Understand data distribution and relationships.  
3. **Data Cleaning:** Handle missing values and outliers.  
4. **Feature Encoding & Scaling:** Convert categorical values and normalize data.  
5. **Data Splitting:** Train-test split using `train_test_split()` with `stratify=y`.  
6. **Model Training:** Fit models to the training data.  
7. **Evaluation:** Compare models using accuracy, precision, recall, and F1-score.  

---

## 🧾 Example Code Snippet
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

 ## 📊Evaluation Metrices

After training, each model’s performance is evaluated using metrics like accuracy, precision, recall, and F1-score.

| Model                    | Accuracy   | Precision  | Recall     | F1-Score   |
| ------------------------ | ---------- | ---------- | ---------- | ---------- |
| **Random Forest**        | **0.9953** | **0.9925** | **1.0000** | **0.9962** |
| Decision Tree Classifier | 0.9930     | 0.9925     | 0.9962     | 0.9944     |
| Logistic Regression      | 0.8033     | 0.7942     | 0.9228     | 0.8537     |
| K-Nearest Neighbors      | 0.5597     | 0.6224     | 0.7420     | 0.6770     |

Best Model : Random Forest (F1 Score- 99.62%)

## 📉 Sample Outputs
✅ Data Split Summary
3.1 Data Split:
--------------------------------------------------
✓ Training set: 80.0%
✓ Test set: 20.0%
✓ Number of features: 10
✓ Train approval rate: 70.25%
✓ Test approval rate: 69.85%

✅ Model Training Output
Logistic Regression trained successfully.
Decision Tree trained successfully.
Random Forest trained successfully.
K-Nearest Neighbors trained successfully.

✅ Example Evaluation Output
Random Forest Accuracy: 84.25%
Precision: 83.40%
Recall: 82.10%
F1-score: 82.75%

## 📈 Visualizations

The notebook may include these plots:

Correlation Heatmap – To identify relationships between numerical features.

Feature Importance Plot – To visualize which attributes most influence predictions.

Confusion Matrix – To analyze correct vs incorrect classifications.

Model Comparison Bar Chart – To display accuracy across all trained models.

## 💡 Insights

Proper data cleaning (especially handling NaN values in the target column) is critical.

Random Forest and Decision Tree models often outperform Logistic Regression in this task.

Balancing classes using stratified sampling improves overall fairness.

Feature scaling helps algorithms like KNN perform better.

## 🧰 Technologies Used

Python

Jupyter Notebook

scikit-learn

pandas

NumPy

Matplotlib / Seaborn

## 🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/Bank_Loan_Approval_System.git


Install dependencies:

pip install -r requirements.txt


Open the notebook:

jupyter notebook Bank_Loan_Approval_System.ipynb


Run all cells to reproduce the results.

## 📈 Future Enhancements

Add feature importance visualization and SHAP explainability.

Experiment with advanced models like XGBoost, LightGBM, or CatBoost.

Deploy the best-performing model using Streamlit or Flask.

Integrate a user interface for loan officers to input applicant data and get instant predictions.

##🧑‍💻 Author

Debdut Nandy
📍 CSE (AI & ML), Brainware University

## 🏁 Conclusion

This project demonstrates how machine learning can automate loan approval prediction with high accuracy and transparency.
It provides a practical understanding of data preprocessing, model training, and evaluation within a real-world financial context.
