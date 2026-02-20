import numpy as np
from activation_functions import ActivationFunction

xor_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor_output = [0, 1, 1, 0]


class NeuralNetwork:
    def __init__(self, epochs=10, learning_rate=0.1) -> None:
        self.epoch = epochs
        self.learning_rate = learning_rate
        self.layer1_weights = np.random.rand(2, 2)  # (2,2)
        self.output_weights = np.random.rand(2, 1)  # (2,1)
        self.layer1_bias = np.random.rand(1, 2)  # (1,2)
        self.output_bias = np.random.rand(1, 1)  # (1,1)

    def train(self, x_arr, y_arr):
        if x_arr.__len__() != y_arr.__len__():
            return -1

        for e in range(self.epoch):
            # print(f"\n--- EPOCH {e} ---")
            for i in range(len(x_arr)):
                x = x_arr[i]
                y = y_arr[i]

                y_pred = self.forward_pass(x)
                self.back_propogation(x, y_pred, y)

    def test(self, x):
        return self.forward_pass(x)

    def forward_pass(self, x):
        # * LAYER 1
        self.layer_1 = (
            x @ self.layer1_weights + self.layer1_bias
        )  # (1,2) @ (2,2) + (1,2) => (1,2) +(1,2) => (1,2)

        # *  LAYER 1 THROUGH ACTIVATION FUNCTION
        self.layer_1_af = ActivationFunction.relu(self.layer_1)  # (1,2)

        # * OUTPUT LAYER
        self.output_layer = (
            self.layer_1_af @ self.output_weights + self.output_bias
        )  # (1,2) @ (2,1) + (1,1) => (1,1) + (1,1)

        # * OUTPUT LAYER THROUGH ACTIVATION FUNCTION
        self.output_layer_af = ActivationFunction.sigmoid(self.output_layer)  # (1,1)

        return self.output_layer_af  # (1,1)

    def back_propogation(self, x, y_pred, y_true):
        # LOSS FUNCTION : BINARY CROSS ENTROPY : -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

        # * COMMON DERIVATIVE STEPS OF THE CHAIN RULE FOR ALL GRADIENTS
        # DERIVATIVE OF LOSS wrt Y PRED
        dLoss_dY_pred = (y_pred - y_true) / (y_pred * (1 - y_pred))
        # DERIVATIVE OF Y PRED wrt OUTPUT LAYER THROUGH ACTIVATION FUNCTION
        dY_pred_dOutput_layer_af = y_pred * (1 - y_pred)

        # * OUTPUT LAYER : INTERMEDIATE STEPS FOR CHAIN RULE
        # DERIVATIVE OF OUTPUT LAYER THROUGH ACTIVATION FUNCTION wrt OUTPUT LAYER WEIGHTS
        dOutput_layer_af_dOutput_layer_weights = self.layer_1_af  # (1,2)
        # DERIVATIVE OF OUTPUT LAYER THROUGH ACTIVATION FUNCTION wrt OUTPUT LAYER BIAS
        dOutput_layer_af_dOutput_layer_bias = 1

        # * HIDDEN LAYER : INTERMEDIATE STEPS FOR CHAIN RULE
        # DERIVATIVE OF OUTPUT LAYER THROUGH ACTIVATION FUNCTION wrt OUTPUT LAYER
        dOutput_layer_af_dOutput_layer = ActivationFunction.sigmoid_derivative(
            sigmoid_term=self.output_layer_af
        )  # (1,1)
        # DERIVATIVE OF OUTPUT LAYER wrt HIDDEN LAYER THROUGH ACTIVATION FUNCTION
        dOutput_layer_dLayer_1_af = self.output_weights  # (2,1)
        # DERIVATIVE OF LAYER 1 THROUGH ACTIVATION FUNCTION wrt LAYER 1
        dLayer_1_af_dLayer_1 = ActivationFunction.relu_derivative(self.layer_1)  # (1,2)
        # DERIVATIVE OF LAYER 1 wrt LAYER 1 WEIGHTS
        dLayer_1_wrt_dLayer_1_weights = np.array(x).reshape(
            -1, 1
        )  # (1,2) #Considering the input X in this case is a 1D array, .T does no change- ie it doesnt transpose it to a column,
        # so we have to use the reshape it.

        # DERIVATIVE OF LAYER 1 wrt LAYER 1 BIASES
        dLayer_1_wrt_dLayer_1_biases = 1

        delta = dLoss_dY_pred * dY_pred_dOutput_layer_af  # (1,1)

        # * DERIVATIVE OF LOSS wrt OUTPUT LAYER WEIGHT
        dLoss_dOutput_weights = (
            dOutput_layer_af_dOutput_layer_weights.T
        ) * delta  # (1,2) * (1,1) , transpose dOutput_layer_af_dOutput_layer_weights to get (2,1) *(1,1) => (2, 1)

        # * DERIVATIVE OF LOSS wrt OUTPUT LAYER BIAS
        dLoss_dOutput_bias = (
            dOutput_layer_af_dOutput_layer_bias
        ) * delta  # (1,1) *(1,1) => (1, 1)

        # * DERIVATIVE OF LOSS wrt LAYER 1 or HIDDEN LAYER WEIGHT
        dLoss_dLayer_1_weights = dLayer_1_wrt_dLayer_1_weights @ (
            (
                (
                    dOutput_layer_af_dOutput_layer
                    * delta  # (1,1) * (1,1) -> called output error signal
                )
                @ dOutput_layer_dLayer_1_af.T  # (2,1) -> transpose it to get (1,2) so that we can make matrix multiplocation with (1,1)
                # => (1,1) @ (1,2) => (1,2)
            )
            * dLayer_1_af_dLayer_1  # (1,2) * (1,2) - element wise multiplication here instead of matmul
        )  # @ dLayer_1_wrt_dLayer_1_weights.T  # (1,2) ->this output as 1,1 matrix, so we shift it to the front instead.

        dLoss_dLayer_1_bias = (
            (
                (
                    dOutput_layer_af_dOutput_layer
                    * delta  # (1,1) * (1,1) -> called error signal
                )
                @ dOutput_layer_dLayer_1_af.T  # (2,1) -> transpose it to get (1,2) so that we can make matrix multiplocation with (1,1) with delta
                # => (1,1) @ (1,2) => (1,2)
            )
            * dLayer_1_af_dLayer_1  # (1,2) * (1,2) - element wise multiplication here instead of matmul
        ) * dLayer_1_wrt_dLayer_1_biases  # (1,2)

        # print("\n--- Before Backpropagation ---")
        # self.get_weight_logs()
        # self.get_gradient_logs(
        #     dLoss_dOutput_weights,
        #     dLoss_dOutput_bias,
        #     dLoss_dLayer_1_weights,
        #     dLoss_dLayer_1_bias,
        # )

        self.layer1_weights -= self.learning_rate * dLoss_dLayer_1_weights
        self.layer1_bias -= self.learning_rate * dLoss_dLayer_1_bias
        self.output_weights -= self.learning_rate * dLoss_dOutput_weights
        self.output_bias -= self.learning_rate * dLoss_dOutput_bias

        # print("\n--- After Backpropagation ---")
        # self.get_weight_logs()

        return (
            dLoss_dOutput_weights,
            dLoss_dOutput_bias,
            dLoss_dLayer_1_weights,
            dLoss_dLayer_1_bias,
        )

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

    def get_weight_logs(self):
        print("Layer 1 weights (self.layer1_weights):")
        print(np.array2string(self.layer1_weights, precision=4, separator=", "))

        print("Output weights (self.output_weights):")
        print(np.array2string(self.output_weights, precision=4, separator=", "))

        print("Layer 1 bias (self.layer1_bias):")
        print(np.array2string(self.layer1_bias, precision=4, separator=", "))

        print("Output bias (self.output_bias):")
        print(np.array2string(self.output_bias, precision=4, separator=", "))


if __name__ == "__main__":
    NN = NeuralNetwork(learning_rate=0.1, epochs=10000)
    print("\n--- Pre Train ---")
    NN.get_weight_logs()
    print("\n--- Pre Train Test ---")
    for x, y in zip(xor_input, xor_output):
        pred = NN.test(x)
        print(f"Input: {x}, Expected: {y}, Got: {round(float(pred[0][0]), 4)}")
    print(f"\n--- Start Train for {NN.epoch} epochs at learning rate {NN.learning_rate}---")
    NN.train(xor_input, xor_output)
    print("\n--- Post Train ---")
    NN.get_weight_logs()
    print("\n--- Test ---")
    for x, y in zip(xor_input, xor_output):
        pred = NN.test(x)
        print(f"Input: {x}, Expected: {y}, Got: {round(float(pred[0][0]), 4)}")

