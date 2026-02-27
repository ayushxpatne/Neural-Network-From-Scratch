import numpy as np
from loss_functions import MeanSquaredError
from activation_functions import ActivationFunction


def generate_parity_dataset(num_sequences, seq_length):
    X, Y = [], []
    for _ in range(num_sequences):
        seq = np.random.randint(0, 2, seq_length)
        labels = np.cumsum(seq) % 2  # running parity
        X.append(seq)
        Y.append(labels)
    return np.array(X), np.array(Y)


# class RNNTrainType(Enum):
#     batch = 'seq2seq'


class RecurrentNeuralNetwork:
    def __init__(
        self,
        sequence_in,
        sequence_out,
        batch_size=None,
        hidden_size=10,  # = n,  No of neurons in hidden layer/weights
        hidden_activation=ActivationFunction.type.TANH.value,
        output_activation=ActivationFunction.type.SIGMOID.value,
        epochs = 10,
        learning_rate = 0.1
    ) -> None:
        # currently assumes the sequence in and sequence out are already = batch size (b)
        self.epochs = epochs
        self.learning_rate = learning_rate
        if batch_size == None:
            # print(self.batch_size)
            self.batch_size = sequence_in[0].__len__()
        else:
            self.batch_size = batch_size
        # x = Input at Current Time Step (t)
        # This will most likely be a list of Vectors; Can be a word embedding vector (1, v) or Some Value 1D, ex: x1
        self.sequence_in = sequence_in  # ([(1,b), (1,b)]), len(sequence_in) = s = k + 1
        # h_prev = Input from Previous Time Step (t-1)
        # This will most likely be a list of Vectors corresponding to each sequence
        self.sequence_out = (
            sequence_out  # (([(1,b), (1,b)]), len(sequence_out) = s = k + 1
        )

        # Current Time Step (t)
        # self.time_step = time_step
        self.hidden_activation = hidden_activation
        self.hidden_size = hidden_size

        self.output_activation = output_activation

        h_init = ActivationFunction.get_initializer(
            activation_type=self.hidden_activation
        )
        o_init = ActivationFunction.get_initializer(
            activation_type=self.output_activation
        )
        # I am kind of unsure- how this will be in batch size like
        # whenn we want to calculate loss after each batch and not at
        # each time step
        x = self.sequence_in[0][0]
        y = self.sequence_out[0][0]

        # print(type_x)
        self.x_dim = 1 if x.ndim == 0 else x.shape[-1]
        self.y_dim = 1 if x.ndim == 0 else y.shape[-1]

        self.x_weights = h_init(self.x_dim, self.hidden_size)  # Wx = (x_dim, n)
        self.h_weights = h_init(self.hidden_size, self.hidden_size)  # Wh =  (n,n)
        self.output_weights = o_init(self.hidden_size, self.y_dim)  # Wo (n, y_dim)

        self.output_bias = np.array([[0.0 for _ in range(self.y_dim)]])  # (1, y_dim)
        self.bias = np.array([[0.0 for _ in range(self.hidden_size)]])  # (1,n)

    def train(self,):
        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            epoch_loss = 0

            for i in range(len(self.sequence_in)):
                seq_in = self.sequence_in[i]
                seq_out = self.sequence_out[i]

                # 1. Forward
                self.forward_pass(seq_in)

                # 2. Compute Loss for reporting
                for t in range(len(seq_out)):
                    epoch_loss += MeanSquaredError.get_loss_single(self.o_values[t], seq_out[t])

                # 3. Backward (Using the BPTT logic from our previous step)
                self.backprop(seq_in, seq_out)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch} - Avg Loss: {epoch_loss / len(self.sequence_in)}"
                )

    def forward_pass(self, seq_in):
        print("-" * 50)
        print("FORWARD PASS")
        print("-" * 50)
        # Say for a seq_in, S_i, S_i = [x0, x1, x2, x3 ... xk], len(seq_in) = k + 1 = s
        # H-1           | X     | Z                              | H               | if_output #NOTE: For the Sequence Parity Example, I am using Sigmoid activation
        # empty_list    | x0    | z0 = x0 @ Wx + 0 @ Wh + b      | h0 = tanh(z0)   | y0 = softmax(h0 @ Wo + bo)  |  Z_o_0 = h0 @ Wo + bo (1, y_dim)
        # h0            | x1    | z1 = x1 @ Wx + h0 @ Wh + b     | h1 = tanh(z1)   | y1 = softmax(h1 @ Wo + bo)  |  Z_o_1 = h1 @ Wo + bo
        # h1            | x2    | z2 = x2 @ Wx + h1 @ Wh + b     | h2 = tanh(z2)   | y2 = softmax(h2 @ Wo + bo)  |  Z_o_2 = h2 @ Wo + bo
        # .             | .     |           .                    |      .          |        .
        # .             | .     |           .                    |      .          |        .
        # .             | .     |           .                    |      .          |        .
        # h_k-1         | xk    | zk = xk @ Wx + hK-1 @ Wh + b   | hk = tanh(zk)   | yk = softmax(hk @ Wo + bo)  |  Z_o_k = hk @ Wo + bo

        # THE Forward Pass for Many to Many May Look Like:
        # x, y_prev -> z -> h = tanh(z) -> zo = h @ wo + b0 -> y_pred = sigmoid(zo)
        #                               -> h, x_nex -> z ....
        self.o_values = []
        self.z_values = []
        self.z_o_values = []
        self.h_values = []

        # self.
        # Assuming len(seq_in) == len(seq_out) == batch_size
        for j in range(self.batch_size):  # here, j => timestamp
            print(f"\n[train]Element {j}")

            x = np.array(seq_in[j])
            if j == 0:
                h_prev = np.array([0 for _ in range(self.hidden_size)])
            x = np.array([[x]]) if x.ndim == 0 else np.array(x)
            # print(f"[train]x Shape: {x.shape}")
            # print(f"[train]h_prev Shape: {h_prev.shape}")

            y_pred = self.cell(
                x=x,
                h_prev=h_prev,
            )

            h_prev = y_pred
            o_af = ActivationFunction.get(self.output_activation)
            z_out = self.z_value(output=True, h_prev=y_pred, x=x)

            self.z_o_values.append(z_out)

            o = o_af(z_out)

            self.o_values.append(o)

        # print(self.o_values)
        # print(self.)
        return self.o_values

    def backprop(self, seq_in, seq_out):
        print("-" * 50)
        print("BACKPROP")
        print("-" * 50)
        # We have to calculate dL/dw at every time step and sum it in the end to get loss for the sequence as net.
        # THE Forward Pass for Many to Many May Look Like:
        # x, y_prev -> z -> h = tanh(z) -> zo = h @ wo + b0 -> y_pred = sigmoid(zo)
        #                               -> h, x_nex -> z ....
        # In Backprop through Time, we have to sum up the weights throughout the sequence, but we start from t till 0.
        # In Reverse, at every time step we add the weight, and then finally subtract this accumulated weight from the network's weight ie self.weight

        # * INIT GRADIENTS (same shape as weights/bias)
        dWx, dWh, db = (
            np.zeros_like(self.x_weights),
            np.zeros_like(self.h_weights),
            np.zeros_like(self.bias),
        )
        dWo, dbo = np.zeros_like(self.output_weights), np.zeros_like(self.output_bias)

        # This carries the gradient from h(t) back to h(t-1)
        dh_next = np.zeros((1, self.hidden_size))

        dydh_derivative = ActivationFunction.get_derivative(self.output_activation)
        dhdz_derivative = ActivationFunction.get_derivative(self.hidden_activation)

        for t in reversed(range(seq_out.__len__())):
            x = seq_in[t]  # (1, x_dim)
            y_pred = self.o_values[t]  # (1, y_dim)
            y_true = seq_out[t]  # (1, y_dim)
            h = self.h_values[t]  # (1, n)
            z_o = self.z_o_values[t]  # (1, y_dim)
            h_prev = self.h_values[t - 1] if t > 0 else np.zeros_like(h)

            # OUTPUT GRADIENTS
            dl_dy = MeanSquaredError.d_Loss_d_y_pred(y_pred, y_true)
            dz_o = dl_dy * dydh_derivative(y_pred)  # Element-wise (1, y_dim)

            dWo += h.T @ dz_o  # (n, 1) @ (1, y_dim) -> (n, y_dim)
            dbo += dz_o

            # HIDDEN LAYER GRADIENTS (BPTT) ---
            # The gradient for h(t) comes from TWO places:
            # 1. The current output (dz_o @ Wo.T)
            # 2. The next hidden state in the future (dh_next)
            dh = dz_o @ self.output_weights.T + dh_next

            # Backprop through the hidden activation (e.g., tanh)
            # dz = dh * f'(z_h)
            dz = dh * dhdz_derivative(h)  # Element-wise (1, n)

            # Accumulate gradients for weights
            dWx += x.reshape(-1,1) @ dz  # (x_dim, 1) @ (1, n) -> (x_dim, n)
            dWh += h_prev.T @ dz  # (n, 1) @ (1, n) -> (n, n)
            db += dz

            # Pass the gradient to the previous time step: dL/dh(t-1) = dz @ Wh.T
            dh_next = dz @ self.h_weights.T

        self.x_weights -= self.learning_rate * dWx
        self.h_weights -= self.learning_rate * dWh
        self.output_weights -= self.learning_rate * dWo
        self.bias -= self.learning_rate * db
        self.output_bias -= self.learning_rate * dbo

    def predict(self, seq_in):
        # Simplified forward pass without saving gradients
        h = np.zeros((1, self.hidden_size))
        preds = []
        h_af = ActivationFunction.get(self.hidden_activation)
        o_af = ActivationFunction.get(self.output_activation)

        for x_t in seq_in:
            x_t = np.array(x_t).reshape(1, -1)
            z = x_t @ self.x_weights + h @ self.h_weights + self.bias
            h = h_af(z)
            z_o = h @ self.output_weights + self.output_bias
            preds.append(o_af(z_o))
        return preds

    def cell(self, x, h_prev):
        tanh = ActivationFunction.get(self.hidden_activation)
        # print(self.hidden_activation)
        z = self.z_value(x, h_prev)
        h = tanh(z)
        self.z_values.append(z)
        self.h_values.append(h)

        return h

    def z_value(self, x, h_prev, output=False):

        # print(f"[z_value]x Shape: {x.shape}")
        # print(f"[z_value]h_prev Shape: {h_prev.shape}")
        if output:
            z = h_prev @ self.output_weights + self.output_bias
            print(f"\n[z_value]Output Weight Shape: {self.output_weights.shape}")
            print(f"[z_value]Output Bias: {self.output_bias.shape}")
            print(f"[z_value]Output Shape {z.shape}")  # (y_dim,y_dim)

        else:
            print(f"\n[z_value]X Weight Shape: {self.x_weights.shape}")
            print(f"[z_value]Y Weight Shape: {self.h_weights.shape}")
            print(f"[z_value]Bias Shape: {self.bias.shape}")

            # z = (1, x) @ (x,v) + (1,v) @ (v,v) + (1,v)
            z = x @ self.x_weights + h_prev @ self.h_weights + self.bias  # (1,h)
            print(f"[z_value]Z Shape: {z.shape}")

        return z


if __name__ == "__main__":
    X_train, Y_train = generate_parity_dataset(100, seq_length=5)
    X_test, Y_test = generate_parity_dataset(20, seq_length=5)

    rnn = RecurrentNeuralNetwork(
        sequence_in=X_train,
        sequence_out=Y_train,
        epochs= 50,
    )
    # 2. Train
    rnn.train()

    # 3. Evaluate
    correct = 0
    total = 0
    print("\n--- EVALUATION ---")
    for i in range(len(X_test)):
        predictions = rnn.predict(X_test[i])
        # Convert sigmoid output to binary (0 or 1)
        binary_preds = [1 if p > 0.5 else 0 for p in predictions]
        actual = [int(y) for y in Y_test[i]]

        if binary_preds == actual:
            correct += 1
        total += 1

        if i < 3:  # Print first 3 results
            print(
                f"In: {X_test[i].flatten()} | Pred: {binary_preds} | Actual: {actual}"
            )

    print(f"\nAccuracy: {(correct/total)*100:.2f}%")
