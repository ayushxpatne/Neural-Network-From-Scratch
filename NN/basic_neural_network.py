import numpy as np
from .activation_functions import ActivationFunction
from .neural_network_helpers import NeuralNetworkHelpers
from .loss_functions import BinaryCrossEntropy
from .nn_utility import NeuralNetworkUtility


class NeuralNetwork:
    def __init__(
        self,
        x_arr: list,  # Input
        y_arr,  # Labels
        epochs=10,
        learning_rate=0.1,
        neurons_per_layer=2,
        hidden_layers=2,  # k
        layer_activation=ActivationFunction.type.RELU,
        output_activation=ActivationFunction.type.SIGMOID,
    ) -> None:
        self.layer_activation = layer_activation
        self.output_activation = output_activation

        init = ActivationFunction.get_initializer(layer_activation)
        # output_init = ActivationFunction.get_initializer(output_activation)

        self.x_arr = x_arr
        self.y_arr = y_arr

        self.x_dim = x_arr[0].__len__()  # Shape of Input X_i
        y_sample = np.array(y_arr[0])
        self.y_dim = (
            1 if y_sample.ndim == 0 else y_sample.shape[0]
        )  # Shape of Output/Label

        self.epochs = epochs
        self.learning_rate = learning_rate

        self.neurons_per_layer = neurons_per_layer  # n
        self.hidden_layers = hidden_layers  # k

        self.input_weights = init(self.x_dim, self.neurons_per_layer)  # Win, W_-1
        self.hidden_layer_weights = [  # Hidden Layer weights: [W0, W1 ... Wk-2]
            init(self.neurons_per_layer, self.neurons_per_layer)
            for _ in range(self.hidden_layers - 1)
        ]
        self.output_weights = init(self.neurons_per_layer, self.y_dim)  # Wout, Wk

        # Biases to zero - works well for all activations
        self.hidden_layer_biases = [  # Hidden Layer Biases: [B0, B1 ... Bk-1]
            np.zeros((1, self.neurons_per_layer)) for _ in range(self.hidden_layers)
        ]
        self.output_bias = np.zeros((1, self.y_dim))  # Bout, Bk

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
        # print('FORWARD PASS')
        #       Z                       |   NAME         | NEXT STEP
        # Z0 = X @ Win + B0             | Hidden Layer 1 | A0
        # Z1 = A0 @ W0 + B1             | Hidden Layer 2 | A1
        # .                             |                |
        # .                             |                |
        # Zk-1 = A_k-2 @ W_k-2 + Bk-1   | Hidden Layer k | A_k-1  (here k-1 == no. of hidden layers (h))
        # Zk = A_k-1 @ Wout + Bout      | Output Layer   | A_k    (Sigmoid) <=> y_pred

        # In Forward Pass, We go from left to right, hence utilise the Weights & Activation Values of Previous Layer to get Z value of current layer.
        # NOTE:
        # We use the Bias of current layer and NOT previous layer.
        # Since Hidden Biases Count =  Hidden Layer Count = k-1, To calculate Z for some hidden layer m, Zm = Am-1 @ Wm-1 @ Bm,
        # When m = 0, we use Input Weights, Win, instead of Am-1.
        self.z_values = []  # [Z0, Z1 ... Zk]
        self.activation_values = []  # [A0, A1, ... Ak]

        layer_fn = ActivationFunction.get(self.layer_activation)
        output_fn = ActivationFunction.get(self.output_activation)

        # We need the loop to go from 0 till k, to get Z values and Activation Values.
        # Hence the range is (0, k+1), (the upper limit is exclusive in range function)
        # Where, at i = 0 => Z0 and i = k => Zk
        total_layers = self.hidden_layers + 1
        for i in range(total_layers):
            if i == 0:  # Z0 => Hidden Layer 1
                # print(f"CALCULATING Z{k}")
                # Z0 = X @ Win + B0
                z = x @ self.input_weights + self.hidden_layer_biases[i]
                a = layer_fn(z)
            elif i == total_layers - 1:  # Zk => Output Layer
                # print(f"CALCULATING OUTPUT Z ie Zk or z{i}")
                # Zk = Ak-1 # Wk-1 + Bk
                # -> The Hidden Layer weights range is [0 .. k-2], as Input weights and Output weights are defined seperately.
                # -> Therefore at i = k, Wk-1 will basically imply W_out.
                # -> Similary, Hidden Layer biases range is [0 .. k-1], at i = k. B_k will imply B_out.
                # Zk = Ak-1 @ Wout + Bout
                z = (
                    self.activation_values[i - 1] @ self.output_weights
                    + self.output_bias
                )
                a = output_fn(z)
            else:
                # print(f"CALCULATING z{k}")
                # Zm = Am-1 @ Wm-1 + Bm for some layer m
                # -> This segment is run when 0 < i < k.
                # -> The Hidden Layer weights range is [0 .. k-2], as Input weights and Output weights are defined seperately.
                # -> Ex: i = 1 (remember at i=0 => Z0 => X @ W_in + B0)
                #        Zi = A[i-1] @ W[i-1] + B[i] => Z1 = A0 @ W0 + B1
                #    Similarly,
                #        i = k-1 (ie for Last Hidden Layer) (remember, i = k => Output Layer)
                #        Zk-1 = A[k-1-1] @ W[k-1-1] + B[k-1] => Zk-1 = A[k-2] @ W[k-2] + B[k-1]

                z = (
                    self.activation_values[i - 1] @ self.hidden_layer_weights[i - 1]
                    + self.hidden_layer_biases[i]
                )
                # NOTE: The Final Output Activation, also referred as y_pred, is also stored in the self.activation_values.
                #       This is consistent with the how the attributes & loops are defined.
                #
                #       Ex:
                #       For Hidden Layer 0, ie the First Hidden Layer, There is no activation value from previous layer, since the layer prior
                #       is the Input & Input Weights. The Loop in Forward Pass addresses this.
                #       From Hidden Layer 1, ie the Second Hidden Layer onwards TILL the output layer, we have Activation Values & Z of Previous Layer.
                #       Hence, for Z1, we use A0 to use the Activation Value of Previous Layer
                #       and for Zk ie the Output Layer, we use A[k-1] to get Activation value of Last Hidden Layer, ie Hidden Layer k-1
                #       and A[k] signifies the final value, ie y_pred.

                a = layer_fn(z)

            self.z_values.append(z)
            self.activation_values.append(a)

        return self.activation_values[-1]

    def back_propogation(self, x, y_pred, y):
        # In Back Propogation, We move from right to left, ie, from Output till Tnput
        # The Chain May Look Like:
        # NOTE: THE HIDDEN LAYER WEIGHT LIST RANGES FROM [0 ... k-2]
        #       The Loop Below is designed to accomodate around this Hidden Layer Weight List
        #       Implying, -1 = Input Weights, [k-1] => Output Weights.
        #
        #       This can also be seen in the Forward Pass Loop
        #       ```
        #           # Zk = Ak-1 # Wk-1 + Bk
        #           # -> The Hidden Layer weights range is [0 .. k-2], as Input weights and Output weights are defined seperately.
        #           # -> Therefore at i = k, Wk-1 will basically imply W_out.
        #       ```

        #
        # WEIGHT                                |  CHAIN
        # W_k-1 (W_out)                         |  dLoss/dW_k-1 = dLoss/dA[k] * dA[k]/dZ[k] * dZ[k]/dW[k-1]
        # W_k-2 (Last Hidden Layer)             |  dLoss/dW_k-2 = DELTA * {{dZ[k]/dA[k-1] * dA[k-1]/dZ[k-1]}} * dZ[k-1]/dW[k-2]
        # W_k-3 (2nd Last Hidden Layer)         |  dLoss/dW_k-3 = DELTA * DELTA_K-1 * {{dZ[k-1]/dA[k-2] * dA[k-2/dZ[k-2]}} * dZ[k-2]/dW[k-3]
        # .                                     |
        # .                                     |
        # .                                     |
        # W_k-l (l-1th Hidden Layer from Last)  |  dLoss/dW_k-l = DELTA * DELTA_K-1 * DELTA_K-2 ... DELTA_K-L-1 * dZ[k-l+1]/dW[k-l]
        # .                                     |
        # .                                     |
        # .                                     |
        # W_m (mth Hidden Layer from front)     |  dLoss/dW_m = DELTA * DELTA_K-1 ... * DELTA_m+1 * dZ[m+1]/dW[m]
        # .                                     |
        # .                                     |
        # W_0 (First Hidden Layer)              |  dLoss/dW_0 = DELTA * DELTA_K-1 ... * DELTA_1 * dZ[1]/dW[0]
        # W_-1 (Input Weights, W_in)            |  dLoss/dW_-1 = DELTA * DELTA_K-1 ...* DELTA_0 * dZ[0]/dW[-1]

        # *       DELTA TERMS
        # *
        # *      DELTA : dLoss/dA[k] * dA[k]/dZ[k]
        # *    DELTA_K : dZ[k]/dA[k-1] * dA[k-1]/dZ[k-1]
        # *  DELTA_K-2 : dZ[k-1]/dA[k-2] * dA[k-2/dZ[k-2]
        # *    DELTA_m : dZ[m+1]/dA[m] * dA[m]/dZ[m]

        self.helpers.deltas = {}
        # We need to calculate DELTAS from -1 till k-1 (inclusive) => range of loop = (-1, k)
        # and DELTA = BinaryCrossEntropy.dLoss_dypred_sigmoid(y_pred, y)
        delta = BinaryCrossEntropy.dLoss_dypred_sigmoid(y_pred, y)

        for i in range(-1, self.hidden_layers):  # -1, 0, 1 ... k-1, | Wk-1 = W_out
            # print('-'*50)
            # if i == -1:
            #     print('[back_propogation] For INPUT WEIGHTS')
            # elif i == self.hidden_layers :
            #     print('[back_propogation] For OUTPUT WEIGHTS')
            # else:
            #     print(f'[back_propogation] For HIDDEN LAYER {i}')
            # print('-'*50)
            dZ_next_dW = self.helpers.dZ_next_dW(self, x, i)
            delta_chain = self.helpers.delta_chain(self, i)
            # print(f'[back_propogation]Shape dZ_next_dW {i}: {dZ_next_dW.shape}')
            # print(f'[back_propogation]Shape delta_chain {i}: {delta_chain.shape} ')

            w = dZ_next_dW @ (delta * delta_chain)
            b = delta * delta_chain * 1

            if i == -1:
                # print(f'[back_propogation] INPUT WEIGHT GRADIENT SHAPE: {w.shape} ')
                # print(f'[back_propogation] INPUT WEIGHT GRADIENT: {w} ')
                self.input_weights -= w * self.learning_rate

            elif i == self.hidden_layers - 1:
                # print(f'[back_propogation] OUTPUT WEIGHT GRADIENT SHAPE: {w.shape} ')
                # print(f'[back_propogation] OUTPUT BIAS GRADIENT SHAPE: {b.shape} ')
                # print(f'[back_propogation] OUTPUT WEIGHT GRADIENT : {w} ')
                # print(f'[back_propogation] OUTPUT BIAS GRADIENT : {b} ')
                self.output_weights -= w * self.learning_rate
                self.output_bias -= b * self.learning_rate

            else:
                # print(f'[back_propogation]HIDDEN LAYER WEIGHT GRADIENT SHAPE {i}: {w.shape} ')
                # print(f'[back_propogation]HIDDEN LAYER BIAS GRADIENT SHAPE {i}: {b.shape} ')
                # print(f'[back_propogation]HIDDEN LAYER WEIGHT GRADIENT {i}: {w} ')
                # print(f'[back_propogation]HIDDEN LAYER BIAS GRADIENT {i}: {b} ')
                self.hidden_layer_weights[i] -= w * self.learning_rate
                self.hidden_layer_biases[i] -= b * self.learning_rate



