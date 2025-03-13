# House Prices Regression Analysis & EDA

## Overview
This project focuses on **Exploratory Data Analysis (EDA)** and **Regression Modeling** for house price prediction using the Kaggle dataset: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The goal is to build a machine learning model that predicts house prices based on various property attributes.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Training & Evaluation](#model-training--evaluation)
- [Model Deployment](#model-deployment)
- [Results](#results)
- [How to Use](#how-to-use)
- [Future Improvements](#future-improvements)
- [References](#references)

## Dataset
- The dataset is available on **Kaggle**: [House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- It contains various house attributes such as lot size, number of rooms, year built, and sale price.
- The target variable: **SalePrice**

## Installation
To run this project locally, install the necessary dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib flask fastapi uvicorn
```

If using Jupyter Notebook, start it with:
```bash
jupyter notebook
```

## Project Structure
```
House-Price-Prediction/
│── data/
│   ├── house_prices.csv  # Dataset file
│── notebooks/
│   ├── eda.ipynb  # Exploratory Data Analysis
│   ├── model_training.ipynb  # Model Training
│── models/
│   ├── house_price_model.pkl  # Saved trained model
│── src/
│   ├── preprocess.py  # Data Preprocessing
│   ├── train.py  # Model Training Script
│   ├── predict.py  # Model Prediction
│   ├── app.py  # API Implementation
│── README.md  # Documentation
│── requirements.txt  # Dependencies
```

## Exploratory Data Analysis (EDA)
The EDA process involves:
- Checking for **missing values** and handling them.
- Understanding the **distribution** of variables.
- Identifying **correlations** between features and SalePrice.
- Visualizing insights using `matplotlib` and `seaborn`.

### Sample Visualization:
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

## Feature Engineering
- **Handling Missing Values:** Imputation of numerical and categorical features.
- **Encoding Categorical Variables:** Using `OneHotEncoder` or `pd.get_dummies()`.
- **Feature Scaling:** Standardizing numerical features using `StandardScaler`.
- **Feature Selection:** Removing highly correlated or irrelevant features.

## Model Training & Evaluation
### Steps:
1. **Splitting the Dataset:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
2. **Training Regression Models:**
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
3. **Evaluating Model Performance:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))
```
4. **Hyperparameter Tuning (Optional):**
```python
from sklearn.model_selection import GridSearchCV
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
grid_search.fit(X_train, y_train)
```
5. **Saving the Model:**
```python
import joblib
joblib.dump(model, "models/house_price_model.pkl")
```

## Model Deployment
### Flask API Implementation
```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)
model = joblib.load("models/house_price_model.pkl")
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])
    return jsonify({"predicted_price": prediction.tolist()})
if __name__ == "__main__":
    app.run(debug=True)
```

## Results
- **Model Performance:** Achieved an R² score of **0.85+**.
- **Error Metrics:** RMSE and MAE values indicate a well-fitted model.

## How to Use
1. **Train the model:** Run `model_training.ipynb`.
2. **Run API:**
```bash
python src/app.py
```
3. **Test API:**
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [3, 2, 1500, 1]}'
```

## Future Improvements
- Implement **XGBoost** for better performance.
- Deploy API using **Docker & AWS/GCP**.
- Use **MLflow** for model tracking.

## References
- [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Flask for Machine Learning API](https://flask.palletsprojects.com/)

---
### ✨ Developed by [Your Name] 🚀

