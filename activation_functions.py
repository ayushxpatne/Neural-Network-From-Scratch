import numpy as np

class ActivationFunction:
    def sigmoid(x):
        return 1 / (1 + np.e ** (-x))
    
    def sigmoid_derivative(sigmoid_term):
        return sigmoid_term * (1 - sigmoid_term)

    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return (x > 0).astype(float)