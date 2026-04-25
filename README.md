# 📊 AI-Powered Sales Predictor

A machine learning web application that predicts product sales based on marketing budget allocation across multiple platforms (YouTube, Facebook, Newspaper).

---

## 🚀 Project Overview

This project uses **Linear Regression** to model the relationship between marketing spend and sales. It provides an interactive web interface where users can adjust budgets and instantly see predicted sales outcomes.

The goal is not just prediction, but understanding how different marketing channels influence sales performance.

---

## 🎯 Features

* 📺 Adjust YouTube, Facebook, and Newspaper budgets using sliders
* 📈 Instant sales prediction using a trained ML model
* 📊 Model performance metrics (R² Score & MAE)
* 📌 Feature importance visualization
* 🌐 Clean and interactive UI built with Streamlit

---

## 🧠 Machine Learning Approach

* Model: **Linear Regression**
* Input Features:

  * YouTube Budget
  * Facebook Budget
  * Newspaper Budget
* Target Variable:

  * Sales

The model learns a relationship of the form:

> Sales = f(YouTube + Facebook + Newspaper)

---

## 📊 Model Evaluation

To ensure reliability, the model is evaluated using:

* **R² Score** → Measures how well the model explains variance
* **MAE (Mean Absolute Error)** → Average prediction error

---

## 📁 Project Structure

```
sales-predictor/
│── app.py
│── Marketing_Data.csv
│── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone <your-repo-link>
cd sales-predictor
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
streamlit run app.py
```

---

## 🌐 Deployment

This app can be deployed easily using **Streamlit Community Cloud**:

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy `app.py`

---

## ⚠️ Limitations

* Assumes a linear relationship between marketing spend and sales
* Does not account for external factors (seasonality, competition, etc.)
* Performance depends on data quality

---

## 💡 Future Improvements

* Add advanced models (Ridge, Lasso, Random Forest)
* Include visualization of actual vs predicted values
* Improve feature engineering
* Deploy with custom domain

---

## 👨‍💻 Author

Built as a machine learning project to explore:

* Model building
* Evaluation
* Deployment
* Real-world usability

---

## 📌 Final Thought

This project goes beyond simple prediction by emphasizing **interpretability and evaluation**, which are critical in real-world machine learning applications.
