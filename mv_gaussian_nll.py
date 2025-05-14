import torch
import torch.nn as nn


class GaussianMVNLL(nn.Module):
    def __init__(self, eps = 1e-6):
        super(GaussianMVNLL, self).__init__()
        self.eps = eps
        
    def forward(self, means, targets, L):
        """Vectorized implementation of multivariate Gaussian NLL"""
        batch_size, n_features = means.shape
        
        covs = torch.matmul(L, L.transpose(-1, -2))  # Compute covariance matrices
        # covs = L
        
        # Reshape means and targets for vectorized operations
        means_flat = means.reshape(-1, n_features) 
        targets_flat = targets.reshape(-1, n_features) 
        covs_flat = covs.reshape(-1, n_features, n_features)  # [batch*horizon, features, features]

        covs_flat = covs_flat.clone()
        # with torch.no_grad():
        #     covs_flat.clamp_(min=self.eps)
        
        identity = torch.eye(n_features, device=covs.device).unsqueeze(0).expand(batch_size, -1, -1)
        covs_flat = covs_flat + self.eps * identity
            
        # Calculate difference vectors
        diff = targets_flat - means_flat 
        
        # Initialise loss tensor
        loss = torch.zeros(batch_size, device=means.device)

        # Calculate loss for each sample
        for i in range(batch_size):
            # Log determinant term
            cov_mat = covs_flat[i]
            det = cov_mat.det()
            logdet = torch.logdet(covs_flat[i].cpu())
            # Quadratic term
            quad = torch.matmul(torch.matmul(diff[i].unsqueeze(0), torch.inverse(covs_flat[i])), diff[i].unsqueeze(1)).squeeze()
            
            # Full loss
            loss[i] = 0.5 * (logdet + quad)
        # Return average loss

        return torch.mean(loss)