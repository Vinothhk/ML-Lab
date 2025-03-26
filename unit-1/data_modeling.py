import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import norm,uniform
import math
import statistics as st
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



"""Normal Distribution"""
mean = 0.0; stdev = 1.0  # Standard Nomral Distribution
normal_data = np.random.normal(mean,stdev, size=1000)
plt.hist(normal_data, bins=60, density=True)
plt.show()
plt.title('Normal Distribution')

"""Uniform Distribution"""
uniform_data = np.random.uniform(low=0, high=5, size=1000) 
plt.hist(uniform_data, bins=60, density=True)
plt.title('Uniform Distribution')
plt.show()



""" Probability Density Functions (PDFs) """
gaussian_data = np.random.normal(loc=5, scale=2, size=1000)  # Mean=5, Std=2
uniform_data = np.random.uniform(low=0, high=10, size=1000)  # Range [0,10]

# Compute PDF values
x_gaussian = np.linspace(-2, 12, 1000)
pdf_gaussian = norm.pdf(x_gaussian, loc=5, scale=2)

x_uniform = np.linspace(0, 10, 1000)
pdf_uniform = uniform.pdf(x_uniform, loc=0, scale=10)

# Plot PDFs
plt.figure(figsize=(10, 5))
# plt.plot(x_gaussian, pdf_gaussian, 'r-', label="Gaussian PDF")
# plt.plot(x_uniform, pdf_uniform, 'b-', label="Uniform PDF")
plt.hist(gaussian_data, bins=30, density=True, alpha=0.5, color='red', label="Gaussian Histogram")
plt.hist(uniform_data, bins=30, density=True, alpha=0.5, color='blue', label="Uniform Histogram")
plt.title("Probability Density Functions (PDFs)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()



"""Class Conditional Probability"""
np.random.seed(42)
X_class0 = np.random.normal(4, 1, 300)  # Class 0: Mean=4, Std=1
X_class1 = np.random.normal(7, 1, 300)  # Class 1: Mean=7, Std=1

# Compute class priors P(C)
prior_C0 = len(X_class0) / (len(X_class0) + len(X_class1))
prior_C1 = len(X_class1) / (len(X_class0) + len(X_class1))

# Estimate Gaussian parameters for P(x|C)
mean_C0, std_C0 = np.mean(X_class0), np.std(X_class0)
mean_C1, std_C1 = np.mean(X_class1), np.std(X_class1)

# Compute probability densities P(x|C) using Gaussian distribution
x_range = np.linspace(0, 11, 1000)
pdf_C0 = norm.pdf(x_range, mean_C0, std_C0)  # P(x|C0)
pdf_C1 = norm.pdf(x_range, mean_C1, std_C1)  # P(x|C1)

# Visualize Class Conditional Probabilities
plt.figure(figsize=(8, 5))
plt.plot(x_range, pdf_C0, 'r-', label="P(x|Class 0) (Gaussian)")
plt.plot(x_range, pdf_C1, 'b-', label="P(x|Class 1) (Gaussian)")
plt.title("Class Conditional Probability Estimation")
plt.xlabel("Feature Value (x)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()

# Print estimated probabilities at a sample point (e.g., x=5)
x_sample = 5
P_x_given_C0 = norm.pdf(x_sample, mean_C0, std_C0)
P_x_given_C1 = norm.pdf(x_sample, mean_C1, std_C1)
print(f"P(x=5 | Class 0) = {P_x_given_C0:.4f}")
print(f"P(x=5 | Class 1) = {P_x_given_C1:.4f}")



"""Naiive Bayes Classifier"""
X, y = make_classification(
    n_features=6,
    n_classes=2,
    n_samples=800,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.title("Data Distribution")
plt.xlabel("Feature")   
plt.ylabel("Class")
plt.show()
# Task 4: Implement Naïve Bayes classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Naïve Bayes Classifier Accuracy: {accuracy:.2f}")
