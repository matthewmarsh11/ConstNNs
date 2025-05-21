import torch 
from torch import nn
from torch.nn import functional
# https://github.com/Javicadserres/wind-production-forecast/blob/28310d7dab7b47d7db3d690580505c1a456e471b/src/model/losses.py#L5
class PinballLoss(nn.Module):
    """
    Calculates the quantile loss function.

    Attributes
    ----------
    self.pred : torch.tensor
        Predictions.
    self.target : torch.tensor
        Target to predict.
    self.quantiles : torch.tensor
    """
    def __init__(self, quantiles):
        super(PinballLoss, self).__init__()
        self.pred = None
        self.targes = None
        self.quantiles = quantiles
        if isinstance(quantiles, list):
            self.quantiles = torch.tensor(quantiles)
        
    def forward(self, pred, target):
        """
        Computes the loss for the given prediction.
        Prediction of shape: (batch_size, features, quantiles)
        """
        
        losses = []

        # Compute loss for each quantile
        for i, q in enumerate(self.quantiles):
            # Select the predictions for the i-th quantile
            pred_q = pred[:, :, i]  # Shape: (batch_size, features)

            # Compute the error (difference) between target and predicted quantile
            errors = target - pred_q

            # Quantile loss formula
            loss_q = torch.max((q - 1) * errors, q * errors)

            # Add the loss for this quantile to the list
            losses.append(loss_q.mean())

        # Mean loss across all quantiles
        total_loss = torch.stack(losses).mean()
        return total_loss

class SmoothPinballLoss(nn.Module):
    """
    Smoth version of the pinball loss function.

    Parameters
    ----------
    quantiles : torch.tensor
    alpha : int
        Smoothing rate.

    Attributes
    ----------
    self.pred : torch.tensor
        Predictions.
    self.target : torch.tensor
        Target to predict.
    self.quantiles : torch.tensor
    """
    def __init__(self, quantiles, alpha=0.001):
        super(SmoothPinballLoss,self).__init__()
        self.pred = None
        self.targes = None
        self.quantiles = quantiles
        self.alpha = alpha

    def forward(self, pred, target):
        """
        Computes the loss for the given prediction.
        """
        error = target - pred
        q_error = self.quantiles * error
        beta = 1 / self.alpha
        soft_error = functional.softplus(-error, beta)

        losses = q_error + soft_error
        loss = torch.mean(torch.sum(losses, dim=1))
        return loss