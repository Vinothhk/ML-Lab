import numpy as np
import math
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class NeuralNetwork():
    """
    A simple feedforward neural network with customizable hidden layers and activation functions.

    Attributes:
        input_size (int): Number of input features.
        hidden_size (list): List of hidden layer sizes.
        output_size (int): Number of output neurons.
        learning_rate (float): Learning rate for weight updates.
        weights (list): List of weight matrices for each layer.
        biases (list): List of bias vectors for each layer.
        activation_function (function): Chosen activation function.
        activation_derivative (function): Derivative of the activation function.
    """
        
    def __init__(self, input_size, hidden_size, output_size, activation : str ='sigmoid', learning_rate : float = 0.1):
        """
        Initializes the neural network with random weights and biases.

        Args:
            input_size (int): Number of input features.
            hidden_size (list): List of hidden layer sizes.
            output_size (int): Number of output neurons.
            activation (str): Activation function ('sigmoid' or 'relu'). Default is 'sigmoid'.
            learning_rate (float): Learning rate for training. Default is 0.1.
        """        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate = learning_rate
        
        # Initialize weights and biases for multiple hidden layers
        self.weights = []
        self.biases = []
        
        total_layer_sizes = [input_size] + hidden_size + [output_size]
        for i in range(len(total_layer_sizes) - 1):
            self.weights.append(np.random.randn(total_layer_sizes[i], total_layer_sizes[i+1]))
            self.biases.append(np.zeros((1, total_layer_sizes[i+1])))

        print(f'Intial Weights: {self.weights}')
        print(f'Weights len: {len(self.weights)}')
        print(f'Initial Bias: {self.biases}')
        print(f'Bias Len: {len(self.biases)}')
        
        # Set activation function
        if activation == 'sigmoid':
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'relu':
            self.activation_function = self.relu
            self.activation_derivative = self.relu_derivative

        else:
            raise ValueError("Unsupported activation function")
        
        print('==== Network initiation Done ====')
        
    def forward(self, X):        
        """
        Performs forward propagation through the network.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            list: Activations for each layer.
        """
        activations = [X]
        for i in range(len(self.weights)):
            X = self.activation_function(np.dot(X, self.weights[i]) + self.biases[i])
            # activations.append(X)
            activations.append(np.array(X))  # Convert to NumPy array explicitly
        return activations
    
    def backward(self, X, y, activations):
        """
        Performs backward propagation and updates weights and biases.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target labels.
            activations (list): Activations from forward propagation.
        """
        errors = [y - activations[-1]]
        deltas = [errors[-1] * self.activation_derivative(activations[-1])]
        
        for i in range(len(self.weights) - 1, 0, -1):
            errors.append(np.dot(deltas[-1], self.weights[i].T))
            deltas.append(errors[-1] * self.activation_derivative(activations[i]))
        
        deltas.reverse()
        
        for i in range(len(self.weights)):
            self.weights[i] += np.dot(activations[i].T, deltas[i]) * self.learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate
    
    def train(self, X, y, epochs=1000):
        """
        Trains the neural network using forward and backward propagation.

        Args:
            X (numpy.ndarray): Training input data.
            y (numpy.ndarray): Target labels.
            epochs (int): Number of training iterations. Default is 1000.
        """
        for epoch in range(epochs):
            activations = self.forward(X)
            self.backward(X, y, activations)
            if epoch % 5 == 0:
                loss = np.mean(np.square(y - activations[-1]))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """
        Generates predictions for given input data.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted values (binary classification).
        """
        return self.forward(X)[-1] > 0.5

    def relu_derivative(self, x):
        """
        Computes the derivative of the ReLU activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Derivative of ReLU.
        """
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self,x):
        """
        Applies the sigmoid activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output after applying sigmoid.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1 - x)

    def relu(self,x):
        """
        Applies the ReLU activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output after applying ReLU.
        """
        return np.maximum(0, x)
    
    def evaluate(self, X, y):
        """
        Evaluates the accuracy of the neural network.

        Args:
            X (numpy.ndarray): Test input data.
            y (numpy.ndarray): True labels.

        Returns:
            float: Accuracy score.
        """                
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Accuracy: {accuracy:.4f}")

    # def predict(self, X):
    #     hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
    #     hidden_output = self.activation_function(hidden_input)
    #     final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
    #     final_output = self.activation_function(final_input)
    #     return final_output > 0.5


X, y = make_classification(n_samples=500, n_features=5, n_classes=2,n_clusters_per_class = 2, random_state=42)
y = y.reshape(-1, 1)  # Ensure y has the correct shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the neural network
nn = NeuralNetwork(input_size=5, hidden_size=[4, 3], output_size=1, activation='sigmoid', learning_rate=0.1)
nn.train(X_train, y_train, epochs=100)
y_pred = nn.predict(X_test)
# Evaluate the model
nn.evaluate(X_test, y_test)

plt.scatter(range(len(y_test)),y_test)
plt.scatter(range(len(y_pred)), y_pred)
plt.legend()
plt.title("Actual vs Predicted Values")
plt.show()