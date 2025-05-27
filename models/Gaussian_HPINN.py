import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Union
from base import *
import cvxpy as cp
from scipy.stats import chi2
import numpy as np
from cvxpylayers.torch import CvxpyLayer
import scipy

class KKT_PPINN(BaseModel):
    """
    Probabilistic Neural Network with projection onto the constraint set

    Args:
        config: int, Configuration object of network specs
        input_dim: int, Dimension of input tensor
        output_dim: int, Dimension of output tensor
        A: torch.tensor: Constraint matrix: of shape (m, * input_dim) - m is the number of constraints
        B: Constraint matrix: of shape (m, * output_dim) - m is the number of constraints
        b: Constraint vector: of shape (m, 1) - m is the number of constraints
    """
    
    def __init__(self, config: MLPConfig, input_dim: int, output_dim: int,
                    A: torch.Tensor, B: torch.Tensor, b: torch.Tensor,
                    epsilon: Union[torch.Tensor, float],
                    probability_level: float = 0.95,
                    ):
        super().__init__(config)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.probability_level = probability_level
        
        self.chi2_thresh = chi2.ppf(self.probability_level, df=self.output_dim)
        
        self.register_buffer("A", A.unsqueeze(0) if A.shape[0] == input_dim else A)
        self.register_buffer("B", B.unsqueeze(0) if B.shape[0] == output_dim else B)
        self.register_buffer("b", b)
        
        self.num_constraints = self.A.shape[0]
        
        # Check if epsilon is a scalar
        if isinstance(self.epsilon, (int, float)):
            self.epsilon = torch.eye(self.num_constraints, device=config.device) * self.epsilon
        elif not torch.all(self.epsilon == torch.diag(torch.diagonal(self.epsilon))):
            raise ValueError("Epsilon must be either a scalar or a diagonal matrix")
        
        inv_BBT = torch.inverse(torch.matmul(self.B, self.B.T))
        BT_inv_BBT = torch.matmul(self.B.T, inv_BBT)
        
        
        Astar = torch.matmul(BT_inv_BBT, self.A)
        Bstar = torch.eye(self.output_dim, device=config.device) - torch.matmul(BT_inv_BBT, self.B)
        bstar = torch.matmul(BT_inv_BBT, self.b)
        
        Cstar = torch.matmul(torch.matmul(self.B.T, inv_BBT), self.B)
        Dstar = torch.inverse(torch.matmul(self.B, self.B.T))
        
        eps_region = self.chi2_thresh * self.epsilon
        
        self.register_buffer("Astar", Astar)
        self.register_buffer("Bstar", Bstar)
        self.register_buffer("bstar", bstar)
        self.register_buffer("Cstar", Cstar)
        self.register_buffer("Dstar", Dstar)
        self.register_buffer("eps_region", eps_region)
        

        
        self.activation = getattr(nn, config.activation)()
        
        # Build layers
        layers = []
        current_dim = self.input_dim
        
        # Input layer
        for _ in range(config.num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, config.hidden_dim),
                self.activation
            ])
            
            layers.append(nn.Dropout(p=config.dropout))
        
            current_dim = config.hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(current_dim, self.output_dim)
        
        self.cholesky_size = self.output_dim * (self.output_dim + 1) // 2
        self.fc_cholesky = nn.Linear(current_dim, self.cholesky_size)
        
        self.fc_fixed1 = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
        self.fc_fixed2 = nn.Linear(input_dim, self.output_dim, bias=False)
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False)
        

        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        z = self.layers(x)
        mu = self.fc(z)

        # Cholesky output size = number of lower-triangular elements
        tril_indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=0, device=x.device)
        cholesky_raw_elements = self.fc_cholesky(z)  # shape: (batch_size, num_tril_elements)

        L_batch = torch.zeros(batch_size, self.output_dim, self.output_dim, device=x.device)

        for i in range(batch_size):
            # Fill lower triangle
            L = torch.zeros(self.output_dim, self.output_dim, device=x.device)
            L[tril_indices[0], tril_indices[1]] = cholesky_raw_elements[i]

            # Apply softplus + epsilon to diagonal to ensure positive entries
            diag_idx = torch.arange(self.output_dim, device=x.device)
            L[diag_idx, diag_idx] = torch.nn.functional.softplus(L[diag_idx, diag_idx]) + 1e-6
            L_batch[i] = L

        
        # project the mean onto the constraint set
        mu_Q = self.fc_fixed1(mu) + self.fc_fixed2(x)
        
        # Cov matrix
        cov = torch.matmul(L_batch, L_batch.transpose(1, 2))
        # Project covariance matrix

        # bias term: B^T * D^* * eps_region * D^* * B^

        tr_1 = torch.matmul(self.B.T, self.Dstar)
        tr_2 = torch.matmul(self.Dstar, self.B)
        
        bias = torch.matmul(tr_1, torch.matmul(self.eps_region, tr_2))
        
        # Second projection: \Sigma_P - C^* * (Sigma_P) * C^*
        pr2 = torch.matmul(self.Cstar, torch.matmul(cov, self.Cstar))
        
        # all together
        
        cov_out = cov - pr2 + bias
        
        # Ensure covariance matrix is positive semi-definite
        for i in range(batch_size):
            L, Q = torch.linalg.eigh(cov_out[i])
        # Check if any eigenvalue is negative
            if torch.any(L.real < 0):
                # If so, set it to zero
                L = torch.clamp(L.real, min=0)
            # Reconstruct the covariance matrix
            cov_out[i] = torch.matmul(Q, torch.matmul(torch.diag(L + 1e-6), Q.transpose(0, 1)))
            
        L_out = torch.linalg.cholesky(cov_out)


        return mu_Q, L_out, mu, L_batch
        