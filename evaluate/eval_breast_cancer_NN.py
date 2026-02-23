from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NN.basic_neural_network import NeuralNetwork


# Load & preprocess
data = load_breast_cancer()
X, y = data.data, data.target

# Scale features to mean=0, std=1 — critical for gradient-based training
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


NN = NeuralNetwork(
    x_arr=X_train,
    y_arr=y_train,
    learning_rate=0.01,  # lower than XOR — 30 features means bigger gradients
    neurons_per_layer=16,
    hidden_layers=2,
    epochs=50,  # start low, see if loss is moving before committing to 1000
)

print(
    f"\nNeural Network Details:\nNeurons Per Hidden Layer: {NN.neurons_per_layer}\nNo. Of Hidden Layers = {NN.hidden_layers}"
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")


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


print("\n--- Pre Train ---")
evaluate(NN, X_test, y_test, "Pre-train Test")

print(f"\n--- Training for {NN.epochs} epochs ---")
NN.train()

print("\n--- Post Train ---")
evaluate(NN, X_train, y_train, "Train")
evaluate(NN, X_test, y_test, "Test")

print("\n----SEED----")
for seed in [0, 1, 7, 13, 42, 99]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    NN = NeuralNetwork(
        x_arr=X_train,
        y_arr=y_train,
        learning_rate=0.01,
        neurons_per_layer=16,
        hidden_layers=2,
        epochs=50,
    )
    NN.train()
    evaluate(NN, X_test, y_test, f"Seed {seed}")

# OUTPUT
# Neural Network Details:
# Neurons Per Hidden Layer: 16
# No. Of Hidden Layers = 2
# Training samples: 455, Test samples: 114

# --- Pre Train ---
# Pre-train Test Accuracy: 60/114 = 52.63%

# --- Training for 50 epochs ---

# --- Post Train ---
# Train Accuracy: 447/455 = 98.24%
# Test Accuracy: 108/114 = 94.74%

# ----SEED----
# Seed 0 Accuracy: 111/114 = 97.37%
# Seed 1 Accuracy: 112/114 = 98.25%
# Seed 7 Accuracy: 110/114 = 96.49%
# Seed 13 Accuracy: 110/114 = 96.49%
# Seed 42 Accuracy: 109/114 = 95.61%
# Seed 99 Accuracy: 108/114 = 94.74%
