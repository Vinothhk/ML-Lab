import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load dataset from library
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['TARGET'] = data.target

# Preprocessing (handle missing values, feature scaling, etc.)
df.dropna(inplace=True)  # Drop missing values
X = df.iloc[:, :-1].values  # Select features (all columns except last)
y = df.iloc[:, -1].values   # Select target variable (last column)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Polynomial Regression Model (Degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Model Evaluation
def evaluate_model(y_test, y_pred, model_name):
    """Evaluate model performance using MSE and R².
    Args:
        y_test (array): true target values
        y_pred (array): predicted target values
        model_name (string): model name
    Returns:
        None.
        Prints the evaluation metrics
    """
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Performance:")
    print(f"MSE: {mse:.4f}, R²: {r2:.4f}\n")

evaluate_model(y_test, y_pred_linear, "Linear Regression")
evaluate_model(y_test, y_pred_poly, "Polynomial Regression")

# Overfitting Analysis
# train_pred_poly = poly_model.predict(X_train_poly)
# evaluate_model(y_train, train_pred_poly, "Polynomial Regression (Train Data)")

# Plot results
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), y_pred_linear, color='red', label='Linear Predicted')
plt.legend()
plt.title("Actual vs Linear Regression Predicted Values")
plt.show()

plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), y_pred_poly, color='green', label='Polynomial Predicted', alpha=0.7)
plt.legend()
plt.title("Actual vs Polynomial Regression Predicted Values")
plt.show()


# Visualizing Errors
errors_linear = y_test - y_pred_linear
errors_poly = y_test - y_pred_poly

plt.figure(figsize=(10, 5))
plt.hist(errors_linear, bins=50, alpha=0.3, label='Linear Regression Errors', color='red')
plt.hist(errors_poly, bins=50, alpha=0.3, label='Polynomial Regression Errors', color='green')
plt.legend()
plt.title("Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show()