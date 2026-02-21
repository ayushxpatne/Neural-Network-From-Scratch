import math


class BinaryCrossEntropy():

    def __init__(self, y_pred, y_true) -> None:
        self.y_pred = y_pred
        self.y_true = y_true
    

    def get_loss(self):
        return -(self.y_true * math.log(self.y_pred) + (1 - self.y_true) * math.log(1 - self.y_pred))
    
    def derivative_bce_wrt_y_pred(self):
        # DERIVATIVE OF LOSS wrt Y PRED
         dLoss_dY_pred = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred))
         return dLoss_dY_pred

