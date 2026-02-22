def scracth()    :
    for k in range(self.hidden_layers, -1, -1):
        # eqn = f"dz_{k+1}/da_{k}*da_{k}/dz_{k}"
        key = f"delta_{k}"
        # print(f"{key} : {eqn}")

        # dZ_k/d_A_k-1 = W_k-1
        d_Zk_d_Ak_prev = self.hidden_layer_weights[k - 1]
        # dA_k-1/dZ_k-1 = dA(Z_k-1)
        d_Ak_prev_d_Zk_prev = ActivationFunction.relu_derivative(self.z_values[k])
        print(d_Zk_d_Ak_prev.shape)  # (n,n)
        print(d_Ak_prev_d_Zk_prev.shape)  # (1, n)
        print()
        temp = d_Ak_prev_d_Zk_prev * d_Zk_d_Ak_prev
        print(f"{key} : {temp.shape}")
        deltas[key] = temp

    # Zk = Ak-1 * Wk-1 + Bk => dZk/dWk-1 = Ak-1  (FOR Z0 : Ak-1 == x)
    output_wt_gradient = delta * y_pred
    # input_wt_gradient = (
    #     delta
    #     * self.helpers.delta_matmul(self, 0, deltas)
    #     * np.array(x).reshape(-1, 1)
    # )

    # print(input_wt_gradient.shape)

    # Zk = Ak-1 * Wk-1 + Bk => dZ/dB = 1
    output_bias_gradient = delta * 1

    # dL/dWin   |  delta * delta h @ delta h-1 @ ... delta 0 * dZ0/dWin  | h == k - 1
    # dL/W0     |  delta * delta h @ delta h-1 @ ... delta 1 * dZ1/dW0   |
    # dL/W1     |  delta * delta h @ delta h-1 @ ... delta 2 * dZ2/dW1   |
    #   .       |                                                        |
    #   .       |                                                        |
    #   .       |                                                        |
    # dL/Wh-1   |  delta * delta h * dZh/dWh-1                           |
    # dL/Wh     |  delta * dZ2/dWh                                       |  Wh <=> Wk-1 <=> Wout

    for k in range(0, self.hidden_layers + 2):
        print(k)
        if k == 0:
            print(f"X = {np.array(x).reshape(-1, 1).shape}")
            print(
                f"self.helpers.delta_matmul(self, k, deltas) : {self.helpers.delta_matmul(self, k, deltas).shape}"
            )
            temp = delta * (
                self.helpers.delta_matmul(self, k, deltas)
                * np.array(x).reshape(-1, 1)
            )

        else:
            print(f"A{k-1} = {self.activation_values[k - 1].T.shape}")
            print(
                f"self.helpers.delta_matmul(self, k, deltas) : {self.helpers.delta_matmul(self, k, deltas).shape}"
            )
            temp = delta * (
                self.activation_values[k - 1].T
                @ self.helpers.delta_matmul(self, k, deltas)
            )

        print(f"Temp: {temp.shape}")

    pass


# def back_propogation(self, x, y_pred, y):
#     delta = BinaryCrossEntropy(
#         y_pred=y_pred, y_true=y
#     ).derivative_bce_wrt_y_pred() * ActivationFunction.sigmoid_derivative(
#         y_pred
#     )  # (1,1)

#     deltas = {}

#     for k in range(
#         self.hidden_layers - 1, 0, -1
#     ):  # Hidden Layer 0 ie delta 0 involves input weights which is currently leading to incompatible element-wise multiplcation, hence has to be done seperately.
#         eqn = f"dz_{k+1}/da_{k}*da_{k}/dz_{k}"
#         key = f"delta_{k}"
#         # print(f"{key} : {eqn}")
#         temp = self.derivative_Z_wrt_A_prev(k) * self.derivative_A_wrt_Z(k - 1)
#         # print(self.derivative_Z_wrt_A_prev(k).shape)
#         # print(self.derivative_A_wrt_Z(k - 1).shape)

#         deltas[key] = temp

#     # print(self.get_deltas_matmul(0, deltas).shape)
#     # print(self.derivative_A_wrt_Z(-1).shape)
#     # print(self.derivative_Z_wrt_A_prev(0).shape)
#     # print(self.derivative_Z_wrt_W(0).shape)

#     # 1. Calculate the 'error signal' (delta) for the input layer first
#     # This represents dL/dZ for the first hidden layer.
#     # It should result in a shape of (1, 4)
#     layer_0_delta = (
#         delta  # (1, 1)
#         * (
#             self.derivative_A_wrt_Z(-1) @ self.derivative_Z_wrt_A_prev(0)
#         )  # (1, 4) if matrix math is right
#         @ self.get_deltas_matmul(0, deltas)  # (4, 4)
#     )

#     input_wt_gradient = self.derivative_Z_wrt_W(0).T @ layer_0_delta
#     output_wt_gradient = delta * self.derivative_Z_wrt_W(-1).T

#     print(f"Output Gradient Shape : {output_wt_gradient.shape}")
#     print(f"Input Gradient Shape : {input_wt_gradient.shape}")

#     hidden_wt_gradient = []

#     for h in range(
#         1, self.hidden_layers
#     ):  # starting from 1 as layer 0 is already calculated above
#         layer_temp_delta = delta * self.get_deltas_matmul(h, deltas)  # (1, 1)
#         print(f"layer_temp_delta {h} Shape : {layer_temp_delta.shape}")
#         print(
#             f"self.derivative_Z_wrt_W(h) {h} Shape : {self.derivative_Z_wrt_W(h).shape}"
#         )
#         hidden_wt_gradient_ = self.derivative_Z_wrt_W(h) * layer_temp_delta

#         print(f"Hidden Layer {h} Gradient Shape : {hidden_wt_gradient_.shape}")

#         hidden_wt_gradient.append(hidden_wt_gradient_)

# def delta_matmul(self, layer: int, deltas: dict):
#     temp = np.array([])
#     for i in range(self.hidden_layers - 1 - layer, 0, -1):
#         if temp.__len__() == 0:
#             key = f"delta_{i}"
#             temp = deltas[key]
#         else:
#             temp = temp @ deltas[key]

#     return temp

# def derivative_A_wrt_Z(self, current_layer: int):

#     if current_layer < 0:
#         return ActivationFunction.relu_derivative(self.z_values[0])

#     return ActivationFunction.relu_derivative(self.z_values[current_layer])
#     # if type == "sigmoid":
#     #     ActivationFunction.sigmoid_derivative(self.activation_values[current_layer])
#     #     pass

# def derivative_Z_wrt_A_prev(self, current_layer: int):
#     # if (
#     #     current_layer == self.hidden_layers - 1
#     # ):  # ie the output layer then use output weights
#     #     return self.output_weights  # (n, y_dim)
#     # elif current_layer == 0:
#     #     return self.input_weights
#     if current_layer == self.hidden_layers - 1:
#         return self.output_weights

#     return self.hidden_layer_weights[current_layer]

# def derivative_Z_wrt_W(self, current_layer: int):
#     # Z = x * W + B => dZ/dW = x
#     if current_layer == 0:
#         return np.array(self.x_arr)

#     return self.activation_values[current_layer + 1]  # (1,n)

#     pass