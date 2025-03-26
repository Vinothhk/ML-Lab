import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.svm import SVC

""" Vector Operations """

# Define vectors
v1 = np.array([3, 4])
v2 = np.array([1, 2])

# Dot product
def perform_operation(v1, v2, operation):
    """Perform operation on two vectors.

    Args:
        v1 (array_like): vector 1
        v2 (array_like): vector 2
        operation (String): tells the operation that needs to be performed.

    Returns:
        ndarray: result of the operation
    """
    if operation == "dot":
        return np.dot(v1, v2)
    elif operation == "cross":
        return np.cross(v1, v2)
    elif operation == "norm":
        return np.linalg.norm(v1)

# Dot product
dot_product = perform_operation(v1, v2, "dot")

# Cross product (dimension must be 2 or 3, else it will throw an error)
v1_3D = np.array([3, 4, 0])
v2_3D = np.array([1, 2, 0])
cross_product = perform_operation(v1_3D, v2_3D, "cross")

# Norms (Magnitude of vectors)
norm_v1 = perform_operation(v1,None,"norm")
norm_v2 = perform_operation(v2,None, "norm")

print(f"Dot Product: {dot_product}")
print(f"Cross Product: {cross_product}")
print(f"Norms: ||v1||={norm_v1}, ||v2||={norm_v2}")



""" Plot 2D Separable Dataset """

# Generate dataset
X, y = make_classification(n_samples=500, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0)

# Plot dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linearly Separable Dataset")
plt.show()

""" Classification Boundary using SVM """
svm_model = SVC(kernel='linear')
svm_model.fit(X, y)

xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), 
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Classification Boundary")
plt.show()