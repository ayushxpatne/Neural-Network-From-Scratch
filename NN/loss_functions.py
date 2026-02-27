import math


class BinaryCrossEntropy:
    def get_loss(y_pred, y_true):
        return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))

    def d_Loss_d_y_pred(y_pred, y_true):
        # DERIVATIVE OF LOSS wrt Y PRED
        dLoss_dY_pred = (y_pred - y_true) / (y_pred * (1 - y_pred))
        return dLoss_dY_pred

    def dLoss_dypred_sigmoid(y_pred, y_true):
        """
        BCE when used with sigmoid as activation function for output simplifies the delta, 
        ie dLoss/d_y_pred * d_y_pred/d_z_k to y_pred - y_true. 

        Returns y_pred - y_true
        """
        return y_pred - y_true


class MeanSquaredError:
    """
    MSE loss — commonly used in RNNs for regression/sequence prediction tasks.
    For sequence outputs, apply per timestep and sum/average across timesteps.

    Loss = (1/n) * sum((y_pred - y_true)^2)
    """

    def get_loss(y_pred: list, y_true: list) -> float:
        """Computes MSE over a sequence of predictions."""
        n = len(y_pred)
        return sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / n

    def get_loss_single(y_pred: float, y_true: float) -> float:
        """MSE for a single timestep output."""
        return (y_pred - y_true) ** 2

    def d_Loss_d_y_pred(y_pred: float, y_true: float) -> float:
        """
        Derivative of MSE loss wrt y_pred (single sample/timestep).
        dL/dy_pred = 2 * (y_pred - y_true)

        The factor of 2 is often absorbed into the learning rate in practice.
        """
        return 2 * (y_pred - y_true)

    def d_Loss_d_y_pred_sequence(y_pred: list, y_true: list) -> list:
        """
        Derivatives for each timestep in a sequence.
        Used for BPTT (Backpropagation Through Time).
        Returns list of gradients, one per timestep.
        """
        n = len(y_pred)
        return [2 * (p - t) / n for p, t in zip(y_pred, y_true)]