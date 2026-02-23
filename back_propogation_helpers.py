import numpy as np
from activation_functions import ActivationFunction

# from basic_neural_network import NeuralNetwork


class NeuralNetworkHelpers:

    def __init__(self) -> None:
        self.deltas = {}

    def get_deltas(self, NN, type=ActivationFunction.type.RELU):
        """
        This Function will calculate all the Deltas from DELTA 0 till DELTA_K-1 (inclusive)
        default Type is ReLU
        """
        # The Loop In which this function will be called, ie in Backpropogation loop the values will range from [-1, k-1]
        # Where, -1 => Input Weights & k-1 => Output Weights
        # For -1 <-> dL/dWin <-> Input Gradient, We need to get deltas from DELTA_k-1 till m+1
        # This Function will calculate all the Deltas from DELTA 0 till DELTA_K-1 (inclusive)

        k = NN.hidden_layers
        # Remember, range(inclusive, exlusive), ie Upper Limit/Stop is excluded
        for i in range(0, k):

            key = f"delta_{i}"
            if key in self.deltas:
                continue

            # For some Delta m : DELTA_m = dZ[m+1]/dA[m] * dA[m]/dZ[m]
            #                   => dZ[m+1]/dA[m] == W[m]
            #                 and, dA[m] / dZ[m] == activation_function_derivative(Zm)

            # i             |              Delta Eqn                  | Output
            # DELTA 0       |        dZ[1]/dA[0] * dA[0]/dZ[0]        | W0 * dA(Z0) (HiddenLayerWeigh[0])
            # DELTA 1       |        dZ[2]/dA[1] * dA[1]/dZ[1]        | W1 * dA(Z1) (H(HiddenLayerWeigh[0])
            # .             |                    .                    |          .
            # .             |                    .                    |          .
            # .             |                    .                    |          .
            # DELTA k-1     |      dZ[k]/dA[k-1] * dA[k-1]/dZ[k-1]    | W[k-1] * dA(Z[k-1]) <-> W_out * dA[z[k-1]]

            derivative = ActivationFunction.get_derivative(type)
            dAmDZm = derivative(NN.z_values[i])
            # print(f"[get_deltas]Shape dA{i}/dZ{i} : {dAmDZm.shape}")
            # print(f"[get_deltas]dA{i}/dZ{i} : {dAmDZm}")

            # dZ/dA_prev = W_prev
            # When i = k - 1, W_prev = W_out
            if i == k - 1:
                dZnextAm = NN.output_weights.T  # (n, y_dim)
            else:
                dZnextAm = NN.hidden_layer_weights[i].T  # (n,n)

            # print(f"[get_deltas]Shape dZ{i + 1}/dA{i} : {dZnextAm.shape}")
            # print(f"[get_deltas] dZ{i + 1}/dA{i} : {dZnextAm}")

            if i == k - 1:
                self.deltas[key] = dZnextAm * dAmDZm
            else:

                self.deltas[key] = dZnextAm * dAmDZm
            # print(f"[get_deltas]Shape {key}: { self.deltas[key].shape}\n")
            # print(f"[get_deltas]Shape {key}: { self.deltas[key].shape}\n")
            # print(self.deltas)

        # return self.deltas

    def delta_chain(self, NN, m):
        """
        Here, layer = the layer of which's gradient we are finding.
            -1 = Input Layer
             k-1 = Output Layer

        FOR A LAYER m, and hidden_layer = k the gradient will be
        dL/DWm = delta * delta_k-1 * delta_k-2 * ... * delta_k-m * ... * delta_m+1 * d_Zm+1/d_Wm


        Ex:
        For Input Layer (-1):
        dL/dW_input = delta * delta_k-1 * ... * delta_0 * dZ0/d_W_input
        For Output Layer (k):
        dl/dW_output = delta * de
        """

        # The Loop In which this function will be called, ie in Backpropogation loop the values will range from [-1, k-1]
        # Where, -1 => Input Weights & k-1 => Output Weights
        # For -1 <-> dL/dWin <-> Input Gradient, We need to get deltas from DELTA_k-1 till m+1
        # Init the deltas dictinary.
        self.get_deltas(NN)

        # LAYER/WEIGHT    | CHAIN TO RETURN
        # -1              | DELTA_K-1 * ... * DELTA 0
        # 0               | DELTA_K-1 * ... * DELTA 1
        # 1               | DELTA_K-1 * ... * DELTA 2
        # .               |  .
        # .               |  .
        # .               |  .
        # m               | DELTA_K-1 * ... * DELTA m+1
        # k-2 (Last Layer)| DELTA_K-1
        # k-1 (Output Wt) | None

        k = NN.hidden_layers
        if m == k - 1:  # (ie  Output layer)
            # No Delta Chain from deltas in self.delta
            # dL/dW_out = DELTA * dZ[k]/W[k-1], where DELTA = DL/DA[k] * DA[k]/DZ[k]
            return np.array([[1]])

        temp = np.array([])

        # In the Loop Below where we multiply deltas, at m = k, the range becomes (k-1, k)
        # And for range(start, stop), start is inclusive & stop is exclusive, the loop essentially
        # does not run at m = k
        # HENCE, we need to explicitly return the temp for this.
        # dL/dWk = DELTA *
        if m == k:
            temp = self.deltas[f"delta_{m}"]

        # Range = [Index of Last Hidden Layer ie k-1, Current Layer's Index)        # [ = including, ) = excluding
        # Ex: For W_input gradient: The chain starts from delta_k-1 till delta_0,
        # Hence we put k-1 as start, and layer as end
        for i in range(k-1, m, -1):
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
                    # f"[delta_chain]Multiplied by delta_{i} ({delta_i.shape}).New Shape: {prev_shape} * {delta_i.shape} -> {temp.shape}"
                # )

        # if temp.size > 0:
        # print(
        #     f"[delta_chain]Chain Complete for Layer {layer}. Final Chain Shape: {temp.shape}\n"
        # )

        return temp

    def dZ_next_dW(self, NN, x, m):
        """
        Here, m = the weight/layer of which's gradient we are finding.
            -1 = Input Layer
             k-1 = Output Layer

        For some layer m's weight, the derivative to get the last step of the chain is d_Zm+1/d_W_m.
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
        # The Loop In which this function will be called, ie in Backpropogation loop the values will range from [-1, k-1]
        # Where, -1 => Input Weights & k-1 => Output Weights
        # For -1, We need to return x
        if m == -1:
            return np.array(x).reshape(-1, 1)
        # For the 0th layer and onwards we need to return the activation value of that layer.
        return NN.activation_values[m].T
