# 🤖 Customer Churn Prediction Model

<p align="center">
  A machine learning project to predict customer churn for a telecommunications company, demonstrating a complete end-to-end ML pipeline.
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blueviolet" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Scikit--learn-orange" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-blue" alt="XGBoost">
  <img src="https://img.shields.io/badge/Pandas-lightgrey" alt="Pandas">
</p>

---

## 📁 Project Structure

The project is organized into a clean, easy-to-navigate directory.

customer-churn-prediction/
├── data/              # 📂 Data sources (raw and processed)

├── notebooks/         # 📓 Jupyter notebooks for EDA and modeling

├── src/               # ⚙ Source code for preprocessing and prediction

├── models/            # 💾 Trained models and preprocessing objects

├── requirements.txt   # 📋 Python dependencies

└── README.md          # 📜 Project overview

---

## 🚀 Quick Start

Follow these simple steps to get the project up and running on your local machine.

1.  *Clone the repository*
    bash
    git clone <your-repo-url>
    cd customer-churn-prediction
    

2.  *Install dependencies*
    bash
    pip install -r requirements.txt
    

3.  *Download the data*
    Download the WA_Fn-UseC_-Telco-Customer-Churn.csv dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the **data/raw/** directory.

4.  *Run the Jupyter notebooks*
    Execute the notebooks in order:
    -   01_eda_and_preprocessing.ipynb ✨
    -   02_model_training_evaluation.ipynb ✨

---

## 📊 Results & Key Findings

The final model demonstrated strong performance on the test set.

| Metric | Score |
| :--- | :--- |
| *Accuracy* | ~0.80 |
| *Precision (Churn)* | ~0.65 |
| *Recall (Churn)* | ~0.75 |
| *F1-Score (Churn)* | ~0.70 |
| *ROC-AUC* | ~0.85 |

*💡 Key Findings:* The most important features for predicting churn were **tenure**, **MonthlyCharges**, **Contract type**, and **InternetService type**.

---

## 🛠 Tech Stack & Methods

* *Language*: Python 3.8+
* *Libraries*: Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Seaborn, Jupyter
* *Methods*: Random Forest, Logistic Regression, SMOTE, GridSearchCV, Feature Importance analysis
* *Concepts*:
    * *Data Cleaning & Preprocessing*: Handling missing values, categorical encoding, and feature scaling.
    * *Imbalanced Data Handling: Using **SMOTE* (Synthetic Minority Over-sampling Technique) to address the class imbalance in the churn data.
    * *Model Selection & Hyperparameter Tuning: Evaluating multiple algorithms and fine-tuning them with **GridSearchCV* for optimal performance.

---

## 👨‍💻 Author

<p align="center">
  <a href="https://www.linkedin.com/in/akshit-kotiyal-80b402257/" target="_blank">
    <img src="https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
  </a>
  <a href="https://github.com/Akshit1103" target="_blank">
    <img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Badge"/>
  </a>
</p>