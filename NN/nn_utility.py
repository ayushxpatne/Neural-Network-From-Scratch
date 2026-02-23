import numpy as np


class NeuralNetworkUtility:

    def get_gradient_logs(self, nn, title="Current Gradients"):

        print(f"\n{'='*20} {title} {'='*20}")

        # 1. Input Layer Gradients
        if hasattr(nn, "input_weights_grad"):
            print(f"\n[INPUT LAYER]")
            print(
                f"Gradient w.r.t. Input Weights (Shape: {nn.i.shape}):"
            )
            print(np.array2string(nn.input_weights_grad, precision=4, separator=", "))
        else:
            print("\n[INPUT LAYER] No gradients found.")

        # 2. Hidden Layer Gradients (Weights and Biases)
        print(f"\n[HIDDEN LAYERS]")
        # We loop through the weights list as the master index
        hidden_w_grads = getattr(nn, "hidden_layer_weights_grad", [])
        hidden_b_grads = getattr(nn, "hidden_layer_biases_grad", [])

        for i in range(max(len(hidden_w_grads), len(hidden_b_grads))):
            if i < len(hidden_w_grads):
                print(
                    f"Hidden Layer {i} Weights Gradient (Shape: {hidden_w_grads[i].shape}):"
                )
                print(np.array2string(hidden_w_grads[i], precision=4, separator=", "))

            if i < len(hidden_b_grads):
                print(
                    f"Hidden Layer {i} Bias Gradient (Shape: {hidden_b_grads[i].shape}):"
                )
                print(np.array2string(hidden_b_grads[i], precision=4, separator=", "))

            print("-" * 15)

        # 3. Output Layer Gradients
        print(f"\n[OUTPUT LAYER]")
        if hasattr(nn, "output_weights_grad"):
            print(
                f"Gradient w.r.t. Output Weights (Shape: {nn.output_weights_grad.shape}):"
            )
            print(np.array2string(nn.output_weights_grad, precision=4, separator=", "))

        if hasattr(nn, "output_bias_grad"):
            print(f"Gradient w.r.t. Output Bias (Shape: {nn.output_bias_grad.shape}):")
            print(np.array2string(nn.output_bias_grad, precision=4, separator=", "))

        print(f"{'='*50}\n")

    def get_weight_logs(self, nn, title="Model Weights & Biases"):
        print(f"\n{'='*20} {title} {'='*20}")

        # Input Layer
        print(f"\n[INPUT LAYER]")
        print(f"Weights (Shape: {nn.input_weights.shape}):")
        print(np.array2string(nn.input_weights, precision=4, separator=", "))

        # Hidden Layers
        print(f"\n[HIDDEN LAYERS]")
        for i, weights in enumerate(nn.hidden_layer_weights):
            print(f"Hidden Layer {i} Weights (Shape: {weights.shape}):")
            print(np.array2string(weights, precision=4, separator=", "))

            # Matching biases (assuming you have a bias for every hidden layer)
            if i < len(nn.hidden_layer_biases):
                print(
                    f"Hidden Layer {i} Bias (Shape: {nn.hidden_layer_biases[i].shape}):"
                )
                print(
                    np.array2string(
                        nn.hidden_layer_biases[i], precision=4, separator=", "
                    )
                )
            print("-" * 15)
            print("\n")

        # Output Layer
        print(f"\n[OUTPUT LAYER]")
        print(f"Weights (Shape: {nn.output_weights.shape}):")
        print(np.array2string(nn.output_weights, precision=4, separator=", "))
        print(f"Bias (Shape: {nn.output_bias.shape}):")
        print(np.array2string(nn.output_bias, precision=4, separator=", "))
        print(f"{'='*50}\n")

    def get_weight_shape_logs(self, nn, title="Model Weights & Biases"):
        print(f"\n{'='*20} {title} {'='*20}")

        # Input Layer
        print(f"\n[INPUT LAYER]")
        print(f"Weights (Shape: {nn.input_weights.shape}):")
        # print(np.array2string(nn.input_weights, precision=4, separator=", "))

        # Hidden Layers
        print(f"\n[HIDDEN LAYERS]")
        for i, weights in enumerate(nn.hidden_layer_weights):
            print(f"Hidden Layer {i} Weights (Shape: {weights.shape}):")
            # print(np.array2string(weights, precision=4, separator=", "))

            # Matching biases (assuming you have a bias for every hidden layer)
            if i < len(nn.hidden_layer_biases):
                print(
                    f"Hidden Layer {i} Bias (Shape: {nn.hidden_layer_biases[i].shape}):"
                )
                # print(
                #     np.array2string(
                #         nn.hidden_layer_biases[i], precision=4, separator=", "
                #     )
                # )
            print("-" * 15)

        # Output Layer
        print(f"\n[OUTPUT LAYER]")
        print(f"Weights (Shape: {nn.output_weights.shape}):")
        # print(np.array2string(nn.output_weights, precision=4, separator=", "))
        print(f"Bias (Shape: {nn.output_bias.shape}):")
        # print(np.array2string(nn.output_bias, precision=4, separator=", "))
        print(f"{'='*50}\n")

    def get_activation_logs(self, nn, title="Layer Activations (Forward Pass)"):
        print(f"\n{'='*20} {title} {'='*20}")

        # The number of activation sets should match the number of z_values
        total_steps = len(nn.z_values)

        for i in range(total_steps):
            # Determine Layer Label
            if i == 0:
                layer_name = "Input -> Hidden Layer 1"
            elif i == total_steps - 1:
                layer_name = "Hidden -> Output Layer"
            else:
                layer_name = f"Hidden Layer {i} -> Hidden Layer {i+1}"

            print(f"\n[{layer_name}]")

            # Log Z values (Weighted Sum)
            z_val = nn.z_values[i]
            print(f"  Z (Pre-Activation)  | Shape: {z_val.shape}")
            print(np.array2string(z_val, precision=4, separator=", "))

            # Log A values (After Activation Function)
            if i < len(nn.activation_values):
                a_val = nn.activation_values[i]
                # Identify which activation was likely used
                act_type = "Sigmoid" if i == total_steps - 1 else "ReLU"
                print(f"  A ({act_type} Output) | Shape: {a_val.shape}")
                print(np.array2string(a_val, precision=4, separator=", "))

            print("-" * 15)

        print(f"{'='*50}\n")
