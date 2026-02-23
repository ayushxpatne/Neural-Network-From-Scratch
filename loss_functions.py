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
