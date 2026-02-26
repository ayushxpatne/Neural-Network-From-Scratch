import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NN.basic_neural_network import NeuralNetwork

xor_input = [[0, 1], [0, 0], [1, 0], [1, 1]]
xor_output = [1, 0, 1, 0]

def evaluate(nn, X, y, label=""):
    correct = 0
    for xi, yi in zip(X, y):
        pred = nn.test(xi)
        predicted_class = 1 if float(pred[0][0]) >= 0.5 else 0
        if predicted_class == yi:
            correct += 1
    acc = correct / len(y) * 100
    print(f"{label} Accuracy: {correct}/{len(y)} = {acc:.2f}%")
    return acc

NN = NeuralNetwork(
    x_arr=xor_input,
    y_arr=xor_output,
    learning_rate=0.1,
    neurons_per_layer=2,
    hidden_layers=1,
    epochs=1000,
)
print(
    f"\nNeural Network Details:\nNeurons Per Hidden Layer: {NN.neurons_per_layer}\nNo. Of Hidden Layers = {NN.hidden_layers}"
)

print("\n--- Pre Train ---")
# NN.utility.get_weight_logs(NN)

print("\n--- Pre Train Test ---")
for x, y in zip(xor_input, xor_output):
    pred = NN.test(x)
    print(f"Input: {x}, Expected: {y}, Got: {round(float(pred[0][0]), 8)}")

print(
    f"\n--- Start Train for {NN.epochs} epochs at learning rate {NN.learning_rate}---"
)
# NN.utility.get_weight_shape_logs(NN)
NN.train()

print("\n--- Post Train ---")
# NN.utility.get_weight_logs(NN)
# NN.utility.get_activation_logs(NN)
NN.utility.get_weight_shape_logs(NN)
# NN.utility.get_gradient_logs(NN)

print("\n--- Test ---")
for x, y in zip(xor_input, xor_output):
    pred = NN.test(x)
    print(f"Input: {x}, Expected: {y}, Got: {round(float(pred[0][0]), 8)}")

evaluate(NN, xor_input, xor_output)