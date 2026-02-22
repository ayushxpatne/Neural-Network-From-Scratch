import numpy as np

class ActivationFunctionType():
    def __init__(self) -> None:
        self.relu = 'relu'
        self.sigmoid = 'sigmoid'


class ActivationFunction():
    def __init__(self) -> None:
        self.type = ActivationFunctionType()

    @staticmethod
    def get_initializer(activation_type):
        """Returns the appropriate weight initializer for the given activation."""
        initializers = {
            'relu': ActivationFunction._he_init,
            'sigmoid': ActivationFunction._xavier_init,
            'tanh': ActivationFunction._xavier_init,
            'softplus': ActivationFunction._xavier_init,  # add new ones here
        }
        return initializers.get(activation_type, ActivationFunction._xavier_init)  # xavier as safe default

    @staticmethod
    def _he_init(fan_in, fan_out):
        """Best for ReLU and variants (LeakyReLU, ELU, etc.)"""
        return np.random.randn(fan_in, fan_out) #* np.sqrt(2 / fan_in)

    @staticmethod
    def _xavier_init(fan_in, fan_out):
        """Best for sigmoid, tanh, softplus - keeps variance stable across layers."""
        return np.random.randn(fan_in, fan_out) #* np.sqrt(1 / fan_in)
    

    def sigmoid(x):
        """
        Computes the Logistic Sigmoid function.
        
        Maps any real value into a range between 0 and 1.
        """
        return 1 / (1 + np.e ** (-x))
    
    def sigmoid_derivative(sigmoid_output):
        """
        Computes the derivative of the Sigmoid function.
        
        Args:
            sigmoid_output: The final output value returned by the sigmoid 
                            function (i.e., y = sigmoid(x)). Using the 
                            pre-calculated output is more computationally 
                            efficient than re-calculating from the input x.
        """
        # Formula: f'(x) = f(x) * (1 - f(x))
        return sigmoid_output * (1 - sigmoid_output)

    def relu(x):
        """
        Computes the Rectified Linear Unit (ReLU).
        
        Returns x if x > 0, otherwise returns 0.
        """
        return np.maximum(0, x)

    def relu_derivative(x):
        """
        Computes the derivative of the ReLU function.
        Args:
            x: The original input (pre-activation value).
        
        Returns 1.0 for input values > 0 and 0.0 otherwise.
        """
        return (x > 0).astype(float)