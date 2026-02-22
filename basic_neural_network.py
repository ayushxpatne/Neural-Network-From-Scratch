import numpy as np
from activation_functions import ActivationFunction
from back_propogation_helpers import NeuralNetworkHelpers
from loss_functions import BinaryCrossEntropy
from nn_utility import NeuralNetworkUtility

xor_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor_output = [0, 1, 1, 0]


class NeuralNetwork:
    def __init__(
        self,
        x_arr: list,
        y_arr,
        epochs=10,
        learning_rate=0.1,
        neurons_per_layer=2,
        hidden_layers=2,
    ) -> None:

        self.x_arr = x_arr
        self.y_arr = y_arr

        self.x_dim = x_arr[0].__len__()
        self.y_dim = 1 if type(y_arr[0]) == int else y_arr[0].__len__()

        self.epochs = epochs
        self.learning_rate = learning_rate

        self.neurons_per_layer = neurons_per_layer
        self.hidden_layers = hidden_layers

        self.input_weights = np.random.rand(self.x_dim, self.neurons_per_layer)

        self.hidden_layer_weights = [
            np.random.rand(self.neurons_per_layer, self.neurons_per_layer)
            for i in range(self.hidden_layers)
        ]
        self.hidden_layer_biases = [
            np.random.rand(1, self.neurons_per_layer)
            for i in range(self.hidden_layers + 1)
        ]

        self.output_weights = np.random.rand(self.neurons_per_layer, self.y_dim)
        self.output_bias = np.random.rand(1, self.y_dim)

        self.utility = NeuralNetworkUtility()
        self.helpers = NeuralNetworkHelpers()

    def train(self):
        if self.x_dim < 1 or self.y_dim < 1:
            return -2
        if self.x_arr.__len__() != self.y_arr.__len__():
            return -1

        for e in range(self.epochs):
            # print(f"\n--- EPOCH {e} ---")
            for i in range(len(self.x_arr)):
                x = self.x_arr[i]
                y = self.y_arr[i]

                y_pred = self.forward_pass(x)
                self.back_propogation(x, y_pred, y)

    def test(self, x):
        return self.forward_pass(x)

    def forward_pass(self, x):
        #       Z                       |   NAME         | NEXT STEP
        # Z0 = X @ Win + B0             | Hidden Layer 1 | A0
        # Z1 = A0 @ W0 + B1             | Hidden Layer 2 | A1
        # .                             |                |
        # .                             |                |
        # Zk-1 = A_k-2 @ W_k-2 + Bk-1   | Hidden Layer k | A_k-1  (here k-1 == no. of hidden layers (h))
        # Zk = A_k-1 @ Wout + Bout      | Output Layer   | A_k    (Sigmoid) <=> y_pred

        self.z_values = []
        total_layers = (
            self.hidden_layers + 1
        )  # If there are h hidden layers, then there will be h+1 z values. (The extra one is output)

        self.activation_values = []

        for k in range(total_layers):
            if k == 0:  # ie Z0 => Hidden Layer 1
                z = x @ self.input_weights + self.hidden_layer_biases[k]
                a = ActivationFunction.relu(z)

            elif k == total_layers - 1:  # ie Zk => Output Layer
                z = (
                    self.activation_values[k - 1] @ self.output_weights
                    + self.output_bias
                )
                a = ActivationFunction.sigmoid(z)

            else:
                z = (
                    self.activation_values[k - 1] @ self.hidden_layer_weights[k]
                    + self.hidden_layer_biases[k]
                )
                a = ActivationFunction.relu(z)
                # self.hidden_layer_weights consists of only hidden layer weights, the list starts from W0 (Win is a seperate attribute)
                # hence the length of the list is self.hidden_layer - 1, making k here equivalent to k-1.

            self.z_values.append(z)
            self.activation_values.append(a)

        return self.activation_values[-1]

    def back_propogation(self, x, y_pred, y):
        self.helpers.deltas = {}

        delta = BinaryCrossEntropy(
            y_pred=y_pred, y_true=y
        ).d_Loss_d_y_pred() * ActivationFunction.sigmoid_derivative(y_pred)

        # FOR A LAYER m, and hidden_layer = k the gradient will be
        # dL/DWm = delta * delta_k-1 * delta_k-2 * ... * delta_m+1 * d_Zm+1/d_Wm
        # 
        # FOR SOME delta_i, where i range from [0, k-1]
        # delta_i = d_z_i+1/d_A_i * dA_i/d_Z_i

        # deltas = self.helpers.get_deltas(self)
        for i in range(-1, self.hidden_layers + 1):
            # print('-'*50)
            # if i == -1:
            #     print('[back_propogation] For INPUT WEIGHTS')
            # elif i == self.hidden_layers :
            #     print('[back_propogation] For OUTPUT WEIGHTS')
            # else:
            #     print(f'[back_propogation] For HIDDEN LAYER {i}')
            # print('-'*50)
            dZ_dW_prev = self.helpers.dZ_dW_prev(self, x_input=x, layer=i)
            delta_chain = self.helpers.delta_chain(self, i)
            # print(f'[back_propogation]Shape dZ_dW_prev {i}: {dZ_dW_prev.shape}')
            # print(f'[back_propogation]Shape delta_chain {i}: {delta_chain.shape} ')

            w =  delta * (dZ_dW_prev @ delta_chain)
            if i > -1:
                b = delta * delta_chain * 1
            else:
                b = 1
            # print('')
            if i == -1:
                # print(f'[back_propogation] INPUT WEIGHT GRADIENT SHAPE: {w.shape} ')
                self.input_weights -= w * self.learning_rate

            elif i == self.hidden_layers:
                # print(f'[back_propogation] OUTPUT WEIGHT GRADIENT SHAPE: {w.shape} ')
                # print(f'[back_propogation] OUTPUT BIAS GRADIENT SHAPE: {b.shape} ')
                self.output_weights -= w * self.learning_rate
                self.output_bias -= b * self.learning_rate
         
            else:
                # print(f'[back_propogation]HIDDEN LAYER WEIGHT GRADIENT SHAPE {i}: {w.shape} ')
                # print(f'[back_propogation]HIDDEN LAYER BIAS GRADIENT SHAPE {i}: {b.shape} ')
                self.hidden_layer_weights[i] -= w * self.learning_rate
                self.hidden_layer_biases[i] -= b * self.learning_rate


if __name__ == "__main__":
    NN = NeuralNetwork(
        x_arr=xor_input,
        y_arr=xor_output,
        learning_rate=0.1,
        neurons_per_layer=4,
        hidden_layers=3,
        epochs=10000,
    )

    print("\n--- Pre Train ---")
    # NN.utility.get_weight_logs(NN)

    print("\n--- Pre Train Test ---")
    for x, y in zip(xor_input, xor_output):
        pred = NN.test(x)
        print(f"Input: {x}, Expected: {y}, Got: {round(float(pred[0][0]), 4)}")

    print(
        f"\n--- Start Train for {NN.epochs} epochs at learning rate {NN.learning_rate}---"
    )
    # NN.utility.get_weight_shape_logs(NN)
    NN.train()

    print("\n--- Post Train ---")
    # NN.utility.get_weight_logs(NN)
    # NN.utility.get_weight_shape_logs(NN)
    # NN.utility.get_gradient_logs(NN)


    print("\n--- Test ---")
    for x, y in zip(xor_input, xor_output):
        pred = NN.test(x)
        print(f"Input: {x}, Expected: {y}, Got: {round(float(pred[0][0]), 4)}")
