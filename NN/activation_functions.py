from enum import Enum
import numpy as np


class ActivationFunctionType(Enum):
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    SOFTMAX = 'softmax'
    SOFTPLUS = 'softplus'


class ActivationFunction():
    type = ActivationFunctionType

    @staticmethod
    def get_initializer(activation_type):
        """
        Returns the appropriate weight initializer for the given activation.
        
        The returned initialization function expects two parameters: 
        (fan_in, fan_out).
        """
        initializers = {
            'relu': ActivationFunction._he_init,
            'sigmoid': ActivationFunction._xavier_init,
            'tanh': ActivationFunction._xavier_init,
            'softmax': ActivationFunction._xavier_init,
            'softplus': ActivationFunction._xavier_init,
        }
        return initializers.get(activation_type, ActivationFunction._xavier_init)

    @staticmethod
    def get(activation_type):
        """
        Returns the appropriate function for the given activation.
        
        Note: All returned activation functions expect the pre-activation 
        input value (often denoted as 'x' or 'z').
        """
        functions = {
            'relu': ActivationFunction.relu,
            'sigmoid': ActivationFunction.sigmoid,
            'tanh': ActivationFunction.tanh,
            'softmax': ActivationFunction.softmax,
            'softplus': ActivationFunction.softplus,
        }
        if activation_type not in functions:
            return -1
        return functions.get(activation_type)

    @staticmethod
    def get_derivative(activation_type):
        """
        Returns the appropriate function derivative for the given activation.
        
        IMPORTANT: The returned derivative functions require different inputs 
        depending on the activation type for computational efficiency.
        
        Requires pre-activation input (z or x):
            - relu
            - softplus
            
        Requires post-activation output (a):
            - sigmoid
            - tanh
            - softmax
        """
        derivatives = {
            'relu': ActivationFunction.relu_derivative,
            'sigmoid': ActivationFunction.sigmoid_derivative,
            'tanh': ActivationFunction.tanh_derivative,
            'softmax': ActivationFunction.softmax_derivative,
            'softplus': ActivationFunction.softplus_derivative,
        }
        if activation_type not in derivatives:
            return -1
        return derivatives.get(activation_type)

    @staticmethod
    def _he_init(fan_in, fan_out):
        """Best for ReLU and variants (LeakyReLU, ELU, etc.)"""
        return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)

    @staticmethod
    def _xavier_init(fan_in, fan_out):
        """Best for sigmoid, tanh, softplus - keeps variance stable across layers."""
        return np.random.randn(fan_in, fan_out) * np.sqrt(1 / fan_in)

    @staticmethod
    def sigmoid(x):
        """
        Computes the Logistic Sigmoid function.

        Maps any real value into a range between 0 and 1.
        """
        return 1 / (1 + np.e ** (-x))

    @staticmethod
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

    @staticmethod
    def relu(x):
        """
        Computes the Rectified Linear Unit (ReLU).

        Returns x if x > 0, otherwise returns 0.
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(z):
        """
        Computes the derivative of the ReLU function.

        Args:
            z: The original input (pre-activation value).

        Returns 1.0 for input values > 0 and 0.0 otherwise.
        """
        return (z > 0).astype(float)

    @staticmethod
    def tanh(x):
        """
        Computes the Hyperbolic Tangent (tanh).

        Maps input values to a range between -1 and 1.
        """
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(a):
        """
        Computes the derivative of the tanh function.

        Args:
            a: The output of the tanh function (post-activation value).

        Returns 1 - a^2.
        """
        return 1 - np.power(a, 2)

    @staticmethod
    def softmax(x):
        """
        Computes the Softmax function.

        Maps a vector of real values to a probability distribution (sums to 1).
        Subtracts max(x) per sample for numerical stability.

        Args:
            x: Input array of shape (n,) or (batch, n).
        """
        # Subtract max along last axis for numerical stability
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    @staticmethod
    def softmax_derivative(softmax_output):
        """
        Computes the Jacobian of the Softmax function for a single sample.

        Args:
            softmax_output: Output of softmax, shape (n,).

        Returns:
            Jacobian matrix of shape (n, n).
            J[i][j] = s_i * (delta_ij - s_j)
        
        Note: In practice with cross-entropy loss, the combined gradient
        simplifies to (softmax_output - y), so the full Jacobian is rarely
        needed directly.
        """
        s = softmax_output.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    @staticmethod
    def softplus(x):
        """
        Computes the Softplus function.

        A smooth approximation of ReLU: log(1 + e^x).
        Always positive, differentiable everywhere.

        Uses log1p(exp(x)) with numerical stability clipping to avoid overflow.
        """
        # Clip to avoid overflow in exp for large x; log1p(exp(x)) ≈ x for large x
        return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -500, 30))))

    @staticmethod
    def softplus_derivative(x):
        """
        Computes the derivative of the Softplus function.

        The derivative of softplus is simply the sigmoid: 1 / (1 + e^-x).

        Args:
            x: The original input (pre-activation value).
        """
        return ActivationFunction.sigmoid(x)