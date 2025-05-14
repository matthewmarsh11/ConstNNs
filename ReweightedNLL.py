import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class ReweightedNLL_FullCovariance(nn.Module):
    def __init__(self, lambda_reg: float, min_diag_L_eps: float = 1e-7):
        """
        Implements the Reweighted Negative Log Likelihood loss for models outputting
        a full covariance matrix (via its Cholesky factor L).
        The regularization term is based on the difference of log-determinants.

        Args:
            lambda_reg (float): The regularization strength (lambda in the paper's formula).
            min_diag_L_eps (float): A small epsilon for numerical stability when calculating
                                    log(diag(L)) within this loss function. Your model's
                                    Cholesky factor L_pred should ideally already have strictly
                                    positive diagonals from its construction (e.g., softplus + epsilon).
        """
        super(ReweightedNLL_FullCovariance, self).__init__()
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        self.lambda_reg = lambda_reg
        self.min_diag_L_eps = min_diag_L_eps

    def _calculate_log_det_from_L(self, L: torch.Tensor) -> torch.Tensor:
        """
        Calculates log_det(Sigma) from Cholesky factor L, where Sigma = L L^T.
        log_det(Sigma) = 2 * sum(log(diag(L))).

        Args:
            L (torch.Tensor): Cholesky factor. Shape: [batch_size, num_features, num_features]

        Returns:
            torch.Tensor: Log determinant of Sigma. Shape: [batch_size]
        """
        diag_L = torch.diagonal(L, dim1=-2, dim2=-1) # Shape: [batch_size, num_features]

        # Ensure diagonal elements are positive before log.
        # This clamp is a safeguard within the loss function.
        clamped_diag_L = torch.clamp(diag_L, min=self.min_diag_L_eps)
        log_diag_L = torch.log(clamped_diag_L)

        # Sum over feature dimension for each sample in batch
        # log_det(Sigma) = log(det(L L^T)) = log(det(L)^2) = 2 * log(det(L))
        # Since L is lower triangular, det(L) is product of its diagonal elements.
        # So log(det(L)) = sum(log(diag(L))).
        log_det_sigma = 2 * torch.sum(log_diag_L, dim=-1) # Shape: [batch_size]
        return log_det_sigma

    def forward(self,
                means_pred: torch.Tensor,    # Predicted means: [batch_size, num_features]
                L_pred: torch.Tensor,        # Predicted Cholesky: [batch_size, num_features, num_features]
                targets: torch.Tensor,       # Ground truth: [batch_size, num_features]
                target_log_dets: torch.Tensor # Target log_det(Sigma_target): [batch_size]
               ) -> torch.Tensor:
        """
        Compute the Reweighted NLL for full covariance matrices.

        Args:
            means_pred: Predicted means from the model.
            L_pred: Predicted Cholesky factor (L_phi) from the model.
                    Must be lower triangular with strictly positive diagonal elements.
            targets: Ground truth values.
            target_log_dets: Precomputed target log determinants (log_det(Sigma_target)).
                             Shape: [batch_size]. This needs to be generated based on
                             your chosen method for defining Sigma_target (e.g., from an ensemble).

        Returns:
            torch.Tensor: The mean loss value over the batch.
        """

        # --- Part 1: NLL Term ---
        # Using torch.distributions.MultivariateNormal for stable NLL calculation.
        # It expects L_pred to be a valid Cholesky factor.
        try:
            # Ensure L_pred is on the same device as means_pred for MultivariateNormal
            L_pred_device = L_pred.to(means_pred.device)
            mvn_dist = MultivariateNormal(loc=means_pred, scale_tril=L_pred_device)
        except RuntimeError as e:
            print(f"Error creating MultivariateNormal distribution for NLL part: {e}")
            diag_L_pred_values = torch.diagonal(L_pred, dim1=-2, dim2=-1)
            min_diag_val = torch.min(diag_L_pred_values) if diag_L_pred_values.numel() > 0 else float('inf')
            print(f"Min diagonal element found in L_pred batch: {min_diag_val}")
            if L_pred.shape[0] > 0 and diag_L_pred_values.numel() > 0 :
                print(f"Example L_pred diag (first sample, first 5 elements): {diag_L_pred_values[0, :min(5, diag_L_pred_values.shape[1])]}")
            raise ValueError(
                f"L_pred is not valid for MultivariateNormal. Min diagonal was {min_diag_val}. Original error: {e}. "
                "Check your model's Cholesky output to ensure diagonals are strictly positive."
            ) from e

        # log_prob() returns log P(targets | means_pred, L_pred) for each sample.
        log_likelihood_per_sample = mvn_dist.log_prob(targets) # Shape: [batch_size]
        nll_per_sample = -log_likelihood_per_sample # Shape: [batch_size]

        # --- Part 2: Regularization Term ---
        # Regularizer: 0.5 * lambda * (log_det(Sigma_pred) - log_det(Sigma_target))^2

        # Calculate log_det(Sigma_pred) from L_pred
        pred_log_dets = self._calculate_log_det_from_L(L_pred_device) # Shape: [batch_size]

        # Ensure target_log_dets is on the correct device and has the same shape as pred_log_dets
        actual_target_log_dets = target_log_dets.to(pred_log_dets.device).view_as(pred_log_dets)

        log_det_diff_sq = (pred_log_dets - actual_target_log_dets)**2 # Shape: [batch_size]
        reg_per_sample = 0.5 * self.lambda_reg * log_det_diff_sq # Shape: [batch_size]

        # --- Combine and average ---
        total_loss_per_sample = nll_per_sample + reg_per_sample # Shape: [batch_size]
        mean_loss = torch.mean(total_loss_per_sample)

        return mean_loss