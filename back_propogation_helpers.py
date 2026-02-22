import numpy as np
from activation_functions import ActivationFunction

# from basic_neural_network import NeuralNetwork


class NeuralNetworkHelpers:

    def __init__(self) -> None:
        self.deltas = {}

    def get_deltas(self, NN, type=ActivationFunction().type.relu):
        """
        type default is relu

        returns dictionary of deltas
        """
        # k = No. of Hidden Layers (self.hidden_layers)
        k = NN.hidden_layers

        # i range from [0, k-1]
        for i in range(0, k):

            key = f"delta_{i}"
            if key in self.deltas:
                continue

            dAprevDZprev = ActivationFunction.relu_derivative(NN.activation_values[i])
            # print(f"[get_deltas]Shape dA{i}/dZ{i} : {dAprevDZprev.shape}")
            # print(f"[get_deltas]dA{i}/dZ{i} : {dAprevDZprev}")

            # dZ/dA_prev = W_prev
            # When i = k - 1, W_prev = W_out
            if i == k - 1:
                dZdAprev = NN.output_weights.T  # (n, y_dim)
            else:
                dZdAprev = NN.hidden_layer_weights[i].T  # (n,n)

            # print(f"[get_deltas]Shape dZ{i + 1}/dA{i} : {dZdAprev.shape}")
            # print(f"[get_deltas] dZ{i + 1}/dA{i} : {dZdAprev}")

            if i == k - 1:
                self.deltas[key] = dZdAprev * dAprevDZprev
            else:

                self.deltas[key] = dAprevDZprev * dZdAprev
            # print(f"[get_deltas]Shape {key}: { self.deltas[key].shape}\n")
            # print(f"[get_deltas]Shape {key}: { self.deltas[key].shape}\n")
            # print(self.deltas)

        return self.deltas

    def delta_chain(self, NN, layer):
        """
        Here, layer = the layer of which's gradient we are finding.
            -1 = Input Layer
             k = Output Layer

        FOR A LAYER m, and hidden_layer = k the gradient will be
        dL/DWm = delta * delta_k-1 * delta_k-2 * ... * delta_k-m * ... * delta_m+1 * d_Zm+1/d_Wm

        At m = k, No Delta Chain from deltas in self.deltas
        Hence k - m == -1 => return 1

        Ex:
        For Input Layer (-1):
        dL/dW_input = delta * delta_k-1 * ... * delta_0 * dZ0/d_W_input
        For Output Layer (k):
        dl/dW_output = delta * de
        """

        self.get_deltas(NN)

        k = NN.hidden_layers
        if layer == k:  # (ie  Output layer)
            return np.array([[1]])  # no delta chain from deltas in self.delta

        temp = np.array([])

        if layer == k - 1:
            temp = self.deltas[f"delta_{layer}"]

        # Range = [Index of Last Hidden Layer ie k-1, Current Layer's Index)        # [ = including, ) = excluding
        # Ex: For W_input gradient: The chain starts from delta_k-1 till delta_0,
        # Hence we put k-1 as start, and layer as end
        for i in range(
            k - 1, layer, -1
        ):  # This returns empty when layer = k-1, for rest it works fine.
            delta_i = self.deltas[f"delta_{i}"]

            if temp.size == 0:
                temp = delta_i
                # print(
                #     f"Initialization: Start chain with delta_{i}. Shape: {temp.shape}"
                # )

            else:
                prev_shape = temp.shape
                # Matrix multiplication: (Batch x N) @ (N x M) -> (Batch x M)
                # print(f"[delta_chain]Prev shape {prev_shape}")
                temp = temp @ delta_i
                # print(
                #     f"[delta_chain]Multiplied by delta_{i} ({delta_i.shape}).New Shape: {prev_shape} * {delta_i.shape} -> {temp.shape}"
                # )

        # if temp.size > 0:
            # print(
            #     f"[delta_chain]Chain Complete for Layer {layer}. Final Chain Shape: {temp.shape}\n"
            # )

        return temp

    def dZ_dW_prev(self, NN, x_input, layer): 
        """
        Here, layer = the layer of which's gradient we are finding.
            -1 = Input Layer
             k = Output Layer

        For some layer m, the last derivative to get the final step of the chain is d_Zm+1/d_W_m.
        Zm+1 = Am @ Wm +Bm+1

        if m+1 = 0, then: Am = X_input, Wm = W_input
        if m+1 = k, then: Am = A_k-1   ,Wm = W_out
        where k = number of hidden layers (self.hidden_layers).

        the self.hidden_layer_weights index range = [0, k-2],

        so for example if k = 2,
        then self.hidden_layers index = [0],
        and m + 1 = k
            => m+ 1= 2 => m = 1,
        but hidden_layers[1] does not exist => W_out

        Hence, d_Zm+1/d_W_m = A_m
        For input weights (W_in), ie d_Z0/d_W_in = X_input


        returns d_Zm+1/d_Wm
        """
        if layer == -1:
            return np.array(x_input).reshape(-1,1)
        if layer == NN.hidden_layers:
            return NN.activation_values[layer - 1].T

        return NN.activation_values[layer].T
