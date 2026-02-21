def back_propogation(self, x, y_pred, y):
    delta = BinaryCrossEntropy(
        y_pred=y_pred, y_true=y
    ).derivative_bce_wrt_y_pred() * ActivationFunction.sigmoid_derivative(
        y_pred
    )  # (1,1)

    # 0 => input layer, ie input weight and hidden layer bias 0, and [dz/da * da/dz-1] count = h
    # 1 => first hidden layer, ie hidden layer weight 0 and hidden layer bias 1 and [dz/da * da/dz-1] count = h- 1
    # 2 => second hidden layer, ie hidden layer wright 1 and hidden layer bias 2 and [dz/da * da/dz-1] count = h - 2
    # h - 1 => last hidden layer, ie hidden layer weight h - 2 and hidden layer bias h - 1 and [dz/da * da/dz-1] count = h - (h-1)
    # h => output layer, ie output weight and output bias = h - h

    # FOR OUTPUT LAYER:
    output_wt_gradient = delta * self.derivative_Z_wrt_W(-1).T
    print(f"Output shape grad : {output_wt_gradient.shape}")

    gradients = {}
    gradient_step = {}

    for i in range(self.hidden_layers): # 0 till h-1
        current_layer_wt = np.array([])
        for j in range(self.hidden_layers - 1 - i): # 0 till h-1-i
            temp = self.derivative_Z_wrt_A_prev(j) @ self.derivative_A_wrt_Z(j).T
            # print(f"[Layer {i}, Chain Pos {j}] dZ/dA_prev @ dA/dZ shape: {temp.shape}")
            if current_layer_wt.__len__() == 0:
                current_layer_wt = temp.T
            else:
                current_layer_wt = current_layer_wt @ temp

        if current_layer_wt.__len__() != 0:
            gradient_step[f"Layer_{i}"] = current_layer_wt

            print(
                f"Gradient Step Shape for Layer {i} : {current_layer_wt.shape}\ndZdW Shape : {self.derivative_Z_wrt_W(i).shape} "
            )

            gradients[f"Layer_{i}"] = delta * (
                self.derivative_Z_wrt_W(i) @ gradient_step[f"Layer_{i}"]
            )

    for key, value in gradients.items():
        print(key)
        print(value)
        print("\n")


    # for i in range(self.hidden_layers):
    #         chain_depth = self.hidden_layers - 1 - i
    #         # if on say 1st hidden layer (index in list would be 0), and the self.hidden_layer = 2,
    #         # then dl/dw0 = delta * (dz2/da1 * da1/dz1) * dz1/dw0
    #         # ie the chain depth = 2 - 1 - 0 = 1 => (dZ/da_prev * da_prev/dz_prev) will appear once
    #         steps_chain_calc = np.array([])
    #         for j in range(chain_depth):
    #             # for middle layers:
    #             # dZ/da_prev shape = (n,n) (since dz/da_prev = W_prev)
    #             # da_prev/dz_prev shape = (n,n) (intuition: a's shape is dependent of W's shape )

    #             key = f'dz_{j}/da_{j-1}*da_{j-1}/dz_{j-1}'

    #             # if step is already calculated- use that step to get the chain's middle part
    #             if key in steps:
    #                 print(f'KEY {key} FOUND in steps')
    #                 step = steps[key] 
    #             else:
    #                 step = self.derivative_A_wrt_Z(j - 1) * self.derivative_Z_wrt_A_prev(j) 
    #                 # print(f'[step] FOR LAYER {i}, DEPTH {j}: {step.shape}')
    #                 # if j-1 < 0 then get input weights is managed by the function. 
    #                 # in that case, the shape will be: (x_dim, n)
    #                 # if we do "derivative_Z_wrt_A_prev(j)  @ derivative_A_wrt_Z(j - 1)"
    #                 # then for middle layers this is (n,n) @ (n,n) there is no effect 
    #                 # but for first layer (ie Hidden Layer 0), where we get input weights for derivative_A_wrt_Z(j - 1): 
    #                 # the matmul (n,n) @ (x_din, n) is incompatible 
    #                 # We need the output of this to be the shape of Wj-1, because the last matmul in the step_chain will be dZj-1/dWj-1, 
    #                 # So in case where we get input weights (x_dim, n) as outcome of derivative_A_wrt_Z(j - 1),
    #                 # we do: self.derivative_A_wrt_Z(j - 1) @ self.derivative_Z_wrt_A_prev(j)  =>  (x_dim, n) @ (n,n) => (x_dim, n)
    #             if steps_chain_calc.__len__() == 0:
    #                 steps_chain_calc = step.T
    #                 # print(f'[steps_chain_calc] FOR LAYER {i}, DEPTH {j}: {steps_chain_calc.shape}')
    #             else:
    #                 steps_chain_calc =  steps_chain_calc @ step
    #                 # print(f'[steps_chain_calc] FOR LAYER {i}, DEPTH {j}: {steps_chain_calc.shape}')
 
    #         if steps_chain_calc.__len__() != 0:
    #             print(f'[steps_chain_calc]FOR LAYER {i}: {steps_chain_calc.shape}')
    #             print(f'[self.derivative_Z_wrt_W(j)]FOR LAYER {i}: {self.derivative_Z_wrt_W(j).shape}')
    #             complete_chain_calc = delta * (steps_chain_calc @ self.derivative_Z_wrt_W(j))
    #             print(f'FOR LAYER {i}: {complete_chain_calc.shape}')
    #             middle_layer_gradients.append(complete_chain_calc)
            
    #     input_wt_gradient



# def back_propogation(self, x, y_pred, y_true):
#     # LOSS FUNCTION : BINARY CROSS ENTROPY : -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

#     # * COMMON DERIVATIVE STEPS OF THE CHAIN RULE FOR ALL GRADIENTS
#     # DERIVATIVE OF LOSS wrt Y PRED
#     dLoss_dY_pred = (y_pred - y_true) / (y_pred * (1 - y_pred))
#     # DERIVATIVE OF Y PRED wrt OUTPUT LAYER THROUGH ACTIVATION FUNCTION
#     dY_pred_dOutput_layer_af = y_pred * (1 - y_pred)

#     # * OUTPUT LAYER : INTERMEDIATE STEPS FOR CHAIN RULE
#     # DERIVATIVE OF OUTPUT LAYER THROUGH ACTIVATION FUNCTION wrt OUTPUT LAYER WEIGHTS
#     dOutput_layer_af_dOutput_layer_weights = self.layer_1_af  # (1,2)
#     # DERIVATIVE OF OUTPUT LAYER THROUGH ACTIVATION FUNCTION wrt OUTPUT LAYER BIAS
#     dOutput_layer_af_dOutput_layer_bias = 1

#     # * HIDDEN LAYER : INTERMEDIATE STEPS FOR CHAIN RULE
#     # DERIVATIVE OF OUTPUT LAYER THROUGH ACTIVATION FUNCTION wrt OUTPUT LAYER
#     dOutput_layer_af_dOutput_layer = ActivationFunction.sigmoid_derivative(
#         sigmoid_term=self.output_layer_af
#     )  # (1,1)
#     # DERIVATIVE OF OUTPUT LAYER wrt HIDDEN LAYER THROUGH ACTIVATION FUNCTION
#     dOutput_layer_dLayer_1_af = self.output_weights  # (2,1)
#     # DERIVATIVE OF LAYER 1 THROUGH ACTIVATION FUNCTION wrt LAYER 1
#     dLayer_1_af_dLayer_1 = ActivationFunction.relu_derivative(self.layer_1)  # (1,2)
#     # DERIVATIVE OF LAYER 1 wrt LAYER 1 WEIGHTS
#     dLayer_1_wrt_dLayer_1_weights = np.array(x).reshape(
#         -1, 1
#     )  # (1,2) #Considering the input X in this case is a 1D array, .T does no change- ie it doesnt transpose it to a column,
#     # so we have to use the reshape it.

#     # DERIVATIVE OF LAYER 1 wrt LAYER 1 BIASES
#     dLayer_1_wrt_dLayer_1_biases = 1

#     delta = dLoss_dY_pred * dY_pred_dOutput_layer_af  # (1,1)

#     # * DERIVATIVE OF LOSS wrt OUTPUT LAYER WEIGHT
#     dLoss_dOutput_weights = (
#         dOutput_layer_af_dOutput_layer_weights.T
#     ) * delta  # (1,2) * (1,1) , transpose dOutput_layer_af_dOutput_layer_weights to get (2,1) *(1,1) => (2, 1)

#     # * DERIVATIVE OF LOSS wrt OUTPUT LAYER BIAS
#     dLoss_dOutput_bias = (
#         dOutput_layer_af_dOutput_layer_bias
#     ) * delta  # (1,1) *(1,1) => (1, 1)

#     # * DERIVATIVE OF LOSS wrt LAYER 1 or HIDDEN LAYER WEIGHT
#     dLoss_dLayer_1_weights = dLayer_1_wrt_dLayer_1_weights @ (
#         (
#             (
#                 dOutput_layer_af_dOutput_layer
#                 * delta  # (1,1) * (1,1) -> called output error signal
#             )
#             @ dOutput_layer_dLayer_1_af.T  # (2,1) -> transpose it to get (1,2) so that we can make matrix multiplocation with (1,1)
#             # => (1,1) @ (1,2) => (1,2)
#         )
#         * dLayer_1_af_dLayer_1  # (1,2) * (1,2) - element wise multiplication here instead of matmul
#     )  # @ dLayer_1_wrt_dLayer_1_weights.T  # (1,2) ->this output as 1,1 matrix, so we shift it to the front instead.

#     dLoss_dLayer_1_bias = (
#         (
#             (
#                 dOutput_layer_af_dOutput_layer
#                 * delta  # (1,1) * (1,1) -> called error signal
#             )
#             @ dOutput_layer_dLayer_1_af.T  # (2,1) -> transpose it to get (1,2) so that we can make matrix multiplocation with (1,1) with delta
#             # => (1,1) @ (1,2) => (1,2)
#         )
#         * dLayer_1_af_dLayer_1  # (1,2) * (1,2) - element wise multiplication here instead of matmul
#     ) * dLayer_1_wrt_dLayer_1_biases  # (1,2)

#     # print("\n--- Before Backpropagation ---")
#     # self.get_weight_logs()
#     # self.get_gradient_logs(
#     #     dLoss_dOutput_weights,
#     #     dLoss_dOutput_bias,
#     #     dLoss_dLayer_1_weights,
#     #     dLoss_dLayer_1_bias,
#     # )

#     self.layer1_weights -= self.learning_rate * dLoss_dLayer_1_weights
#     self.layer1_bias -= self.learning_rate * dLoss_dLayer_1_bias
#     self.output_weights -= self.learning_rate * dLoss_dOutput_weights
#     self.output_bias -= self.learning_rate * dLoss_dOutput_bias

#     # print("\n--- After Backpropagation ---")
#     # self.get_weight_logs()

#     return (
#         dLoss_dOutput_weights,
#         dLoss_dOutput_bias,
#         dLoss_dLayer_1_weights,
#         dLoss_dLayer_1_bias,
#     )