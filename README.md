# 🚀 Employee Attrition Predictor

A Machine Learning web application that predicts whether an employee is likely to leave the company based on HR-related attributes. The application includes an interactive dashboard for data exploration and a prediction interface powered by a trained machine learning model.

🔗 **Live Demo:** https://attrition-predictor-4d8sibkvy4aegtyg39mgut.streamlit.app

---

## 📌 Project Overview

Employee attrition is a major challenge for organizations. This project uses machine learning to analyze employee information and predict the likelihood of attrition. It also provides interactive visualizations to understand the dataset and key business insights.

The application is built using **Python**, **Scikit-learn**, and **Streamlit**, and is deployed on **Streamlit Community Cloud**.

---

## ✨ Features

- 📊 Interactive HR Analytics Dashboard
- 🤖 Employee Attrition Prediction
- 📈 Data Visualization
- 📋 Model Performance Comparison
- ⚡ Fast and User-Friendly Streamlit Interface
- 🌐 Live Web Deployment

---

## 📂 Dataset

**Dataset:** IBM HR Analytics Employee Attrition Dataset

- Total Employees: **1,470**
- Target Variable: **Attrition**
- Numerical and Categorical Features
- Real-world HR analytics dataset

---

##  Machine Learning Models

The following models were trained and compared:

- Logistic Regression (Best Model)
- Random Forest
- Gradient Boosting

### Best Model Performance

| Metric | Score |
|---------|--------|
| F1 Score | 0.44 |
| ROC-AUC | 0.81 |

---

## 🖥️ Application Pages

### 🏠 Home
Provides project overview, technologies used, and model information.

### 📊 Dataset Dashboard
Displays interactive charts and insights from the IBM HR Attrition dataset.

### 🤖 Predict Attrition
Allows users to enter employee information and predict whether the employee is likely to leave the organization.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib
- Git
- GitHub

---

## 📁 Project Structure

```text
attrition-predictor/
│
├── app.py
├── train_model.py
├── best_model.pkl
├── scaler.pkl
├── feature_names.pkl
├── HR_Employee_Attrition.csv
├── model_results.json
├── requirements.txt
├── render.yaml
├── README.md
└── images/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/utkarshpatilds/attrition-predictor.git
```

Move into the project directory

```bash
cd attrition-predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📷 Screenshots

### 🏠 Home Page

> Add screenshot here

```
images/home.png
```

### 📊 Dataset Dashboard

> Add screenshot here

```
images/dashboard.png
```

### 🤖 Prediction Page

> Add screenshot here

```
images/prediction.png
```

### 📈 Prediction Result

> Add screenshot here

```
images/result.png
```

---

## 🎯 Future Improvements

- SHAP Explainable AI
- Employee Risk Score
- Download Prediction Report
- Batch Prediction using CSV Upload
- Cloud Database Integration
- Authentication System

---

## 👨‍💻 Author

**Utkarsh Patil**

Final Year B.Tech (Computer Science & Engineering - Data Science)

### GitHub

https://github.com/utkarshpatilds

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
