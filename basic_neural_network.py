import numpy as np
from activation_functions import ActivationFunction
from loss_functions import BinaryCrossEntropy

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
            for i in range(self.hidden_layers - 1)
        ]

        self.hidden_layer_biases = [
            np.random.rand(1, self.neurons_per_layer) for i in range(self.hidden_layers)
        ]

        self.output_weights = np.random.rand(self.neurons_per_layer, self.y_dim)

        self.output_bias = np.random.rand(1, self.y_dim)

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

        # INPUT TO FIRST HIDDEN LAYER:
        x_l = x @ self.input_weights
        self.activation_values = []
        self.z_values = []

        # HIDDEN LAYER TO LAST LAYER
        for current_layer in range(self.hidden_layers):
            # print(self.activation_values)
            # LAST LAYER TO OUTPUT
            if current_layer == self.hidden_layers - 1:
                l = (
                    self.activation_values[current_layer - 1] @ self.output_weights
                    + self.output_bias
                )
                a = ActivationFunction.sigmoid(l)
                self.activation_values.append(a)
                return a

            if current_layer == 0:
                l = x_l + self.hidden_layer_biases[current_layer]
                
            else:
                l = (
                    self.activation_values[current_layer - 1]
                    @ self.hidden_layer_weights[current_layer]
                    + self.hidden_layer_biases[current_layer]
                )

            self.z_values.append(l)
            a = ActivationFunction.relu(l)
            self.activation_values.append(a)

    def back_propogation(self, x, y_pred, y):
        delta = BinaryCrossEntropy(
            y_pred=y_pred, y_true=y
        ).derivative_bce_wrt_y_pred() * ActivationFunction.sigmoid_derivative(
            y_pred
        )  # (1,1)

        deltas = {}

        for k in range(
            self.hidden_layers - 1, 0, -1
        ):  # Hidden Layer 0 ie delta 0 involves input weights which is currently leading to incompatible element-wise multiplcation, hence has to be done seperately.
            eqn = f"dz_{k+1}/da_{k}*da_{k}/dz_{k}"
            key = f"delta_{k}"
            # print(f"{key} : {eqn}")
            temp = self.derivative_Z_wrt_A_prev(k) * self.derivative_A_wrt_Z(k - 1)
            # print(self.derivative_Z_wrt_A_prev(k).shape)
            # print(self.derivative_A_wrt_Z(k - 1).shape)

            deltas[key] = temp

        # print(self.get_deltas_matmul(0, deltas).shape)
        # print(self.derivative_A_wrt_Z(-1).shape)
        # print(self.derivative_Z_wrt_A_prev(0).shape)
        # print(self.derivative_Z_wrt_W(0).shape)

        # 1. Calculate the 'error signal' (delta) for the input layer first
        # This represents dL/dZ for the first hidden layer.
        # It should result in a shape of (1, 4)
        layer_0_delta = (
            delta  # (1, 1)
            * (
                self.derivative_A_wrt_Z(-1) @ self.derivative_Z_wrt_A_prev(0)
            )  # (1, 4) if matrix math is right
            @ self.get_deltas_matmul(0, deltas)  # (4, 4)
        )

        input_wt_gradient = self.derivative_Z_wrt_W(0).T @ layer_0_delta
        output_wt_gradient = delta * self.derivative_Z_wrt_W(-1).T

        print(f"Output Gradient Shape : {output_wt_gradient.shape}")
        print(f"Input Gradient Shape : {input_wt_gradient.shape}")

        hidden_wt_gradient = []

        for h in range(
            1, self.hidden_layers
        ):  # starting from 1 as layer 0 is already calculated above
            layer_temp_delta = delta * self.get_deltas_matmul(h, deltas)  # (1, 1)
            print(f"layer_temp_delta {h} Shape : {layer_temp_delta.shape}")
            print(
                f"self.derivative_Z_wrt_W(h) {h} Shape : {self.derivative_Z_wrt_W(h).shape}"
            )
            hidden_wt_gradient_ = self.derivative_Z_wrt_W(h) * layer_temp_delta

            print(f"Hidden Layer {h} Gradient Shape : {hidden_wt_gradient_.shape}")

            hidden_wt_gradient.append(hidden_wt_gradient_)

    def get_deltas_matmul(self, layer: int, deltas: dict):
        temp = np.array([])
        for i in range(self.hidden_layers - 1 - layer, 0, -1):
            if temp.__len__() == 0:
                key = f"delta_{i}"
                temp = deltas[key]
            else:
                temp = temp @ deltas[key]

        return temp

    def derivative_A_wrt_Z(self, current_layer: int):

        if current_layer < 0:
            return ActivationFunction.relu_derivative(self.z_values[0])

        return ActivationFunction.relu_derivative(self.z_values[current_layer])
        # if type == "sigmoid":
        #     ActivationFunction.sigmoid_derivative(self.activation_values[current_layer])
        #     pass

    def derivative_Z_wrt_A_prev(self, current_layer: int):
        # if (
        #     current_layer == self.hidden_layers - 1
        # ):  # ie the output layer then use output weights
        #     return self.output_weights  # (n, y_dim)
        # elif current_layer == 0:
        #     return self.input_weights
        if current_layer == self.hidden_layers - 1:
            return self.output_weights

        return self.hidden_layer_weights[current_layer]

    def derivative_Z_wrt_W(self, current_layer: int):
        # Z = x * W + B => dZ/dW = x
        if current_layer == 0:
            return np.array(self.x_arr)

        return self.activation_values[current_layer + 1]  # (1,n)

        pass

    def get_gradient_logs(
        self,
        dLoss_dOutput_weights,
        dLoss_dOutput_bias,
        dLoss_dLayer_1_weights,
        dLoss_dLayer_1_bias,
    ):
        print("\n--- Gradients ---")
        # Pretty-print gradient matrices
        print("Gradient w.r.t. layer 1 weights (dLoss_dLayer_1_weights):")
        print(np.array2string(dLoss_dLayer_1_weights, precision=4, separator=", "))

        print("Gradient w.r.t. output weights (dLoss_dOutput_weights):")
        print(np.array2string(dLoss_dOutput_weights, precision=4, separator=", "))

        print("Gradient w.r.t. layer 1 bias (dLoss_dLayer_1_bias):")
        print(np.array2string(dLoss_dLayer_1_bias, precision=4, separator=", "))

        print("Gradient w.r.t. output bias (dLoss_dOutput_bias):")
        print(np.array2string(dLoss_dOutput_bias, precision=4, separator=", "))

    def get_weight_logs(self, title="Model Weights & Biases"):
        print(f"\n{'='*20} {title} {'='*20}")

        # Input Layer
        print(f"\n[INPUT LAYER]")
        print(f"Weights (Shape: {self.input_weights.shape}):")
        print(np.array2string(self.input_weights, precision=4, separator=", "))

        # Hidden Layers
        print(f"\n[HIDDEN LAYERS]")
        for i, weights in enumerate(self.hidden_layer_weights):
            print(f"Hidden Layer {i} Weights (Shape: {weights.shape}):")
            print(np.array2string(weights, precision=4, separator=", "))

            # Matching biases (assuming you have a bias for every hidden layer)
            if i < len(self.hidden_layer_biases):
                print(
                    f"Hidden Layer {i} Bias (Shape: {self.hidden_layer_biases[i].shape}):"
                )
                print(
                    np.array2string(
                        self.hidden_layer_biases[i], precision=4, separator=", "
                    )
                )
            print("-" * 15)

        # Output Layer
        print(f"\n[OUTPUT LAYER]")
        print(f"Weights (Shape: {self.output_weights.shape}):")
        print(np.array2string(self.output_weights, precision=4, separator=", "))
        print(f"Bias (Shape: {self.output_bias.shape}):")
        print(np.array2string(self.output_bias, precision=4, separator=", "))
        print(f"{'='*50}\n")


if __name__ == "__main__":
    NN = NeuralNetwork(
        x_arr=xor_input[:1],
        y_arr=xor_output[:1],
        learning_rate=0.1,
        neurons_per_layer=4,
        hidden_layers=3,
        epochs=1,
    )

    # print("\n--- Pre Train ---")
    NN.get_weight_logs()

    # print("\n--- Pre Train Test ---")
    # for x, y in zip(xor_input, xor_output):
    #     pred = NN.test(x)
    #     print(f"Input: {x}, Expected: {y}, Got: {round(float(pred[0][0]), 4)}")

    # print(
    #     f"\n--- Start Train for {NN.epochs} epochs at learning rate {NN.learning_rate}---"
    # )
    NN.train()

    print("\n--- Post Train ---")
    NN.get_weight_logs()

    # print("\n--- Test ---")
    # for x, y in zip(xor_input, xor_output):
    #     pred = NN.test(x)
    #     print(f"Input: {x}, Expected: {y}, Got: {round(float(pred[0][0]), 4)}")
