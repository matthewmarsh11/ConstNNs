import torch 
import torch.nn as nn
import numpy as np
from base import *
from typing import List, Optional, Tuple
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class mpi_nn(BaseModel):
    
    """
    Neural Network with Moore-Penrose Pseudo-Inverse non-trainable layer for projection of constraints
    
    Args:
        config: int, Configuration object of network specs
        input_dim: int, Dimension of input tensor
        output_dim: int, Dimension of output tensor
        A: torch.tensor: Constraint matrix: of shape (m, * input_dim) - m is the number of constraints
        B: Constraint matrix: of shape (m, * output_dim) - m is the number of constraints
        b: Constraint vector: of shape (m, 1) - m is the number of constraints
        
    """
    
    def __init__(self, config: MLPConfig, input_dim: int, output_dim: int, 
                 horizon: int, A: torch.Tensor, B: torch.Tensor, b: torch.Tensor):
    
        super().__init__(config)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        
        self.A = A
        self.B = B
        self.b = b
        
        # Transform the constraint matrices so the output is constrained
        
        # assert self.A.shape[1] == input_dim / 10, "A matrix has incorrect dimensions"
        # assert self.B.shape[1] == output_dim, "B matrix has incorrect dimensions"
        # assert self.b.shape[0] == self.A.shape[0], "b vector has incorrect dimensions"
        
        self.num_constraints = self.A.shape[0]
        

        self.B_inv = torch.linalg.pinv(self.B) # Invert B using Moore-Penrose Pseudo-Inverse
        
        self.proj_space = torch.eye(self.output_dim*self.horizon, self.output_dim*self.horizon) - self.B_inv @ self.B  # Projection onto the null space of B

        self.activation = getattr(nn, config.activation)()
        
        # Build layers
        layers = []
        current_dim = self.input_dim
        
        # Input layer
        layers.extend([
            nn.Linear(current_dim, config.hidden_dim),
            self.activation
        ])
        
        current_dim = config.hidden_dim
        
        # Hidden layers
        for _ in range(config.num_layers - 2):
            layers.extend([
                nn.Linear(current_dim, config.hidden_dim),
                self.activation
            ])
                
        current_dim = config.hidden_dim
            
        self.layers = nn.Sequential(*layers)
        
        # Output layers for mean prediction
        # hidden -> full * horizon
        # should go from hidden -> reduced * horizon
        self.fc = nn.Linear(current_dim, self.output_dim * self.horizon)
        
        # Lower triangular Cholesky factor raw values
        # We only need n*(n+1)/2 values for the lower triangular part
        n = self.output_dim
        self.cholesky_size = n * (n + 1) // 2
        self.fc_cholesky = nn.Linear(current_dim, self.cholesky_size * self.horizon)
        
        #########
        # train based here
        #########
        
        # Output layers adjusted
        self.fc_adj1 = nn.Linear(self.output_dim * self.horizon, self.output_dim * self.horizon, bias = False)
        self.fc_adj1.weight = nn.Parameter(self.proj_space, requires_grad=False)
        self.fc_adj2 = nn.Linear(self.output_dim * self.horizon, self.output_dim * self.horizon, bias = False)
        self.fc_adj2.weight = nn.Parameter(-self.B_inv @ self.A, requires_grad=False)
        self.fc_adj2.bias = nn.Parameter(self.B_inv @ self.b)
        
        self.fc_cholesky_adj = nn.Linear(self.cholesky_size * self.horizon, self.cholesky_size * self.horizon, bias = False)
        self.fc_cholesky_adj.weight = nn.Parameter(self.proj_space, requires_grad=False)
        self.fc_cholesky_adj2 = nn.Linear(self.cholesky_size * self.horizon, self.cholesky_size * self.horizon, bias = False)
        self.fc_cholesky_adj2.weight = nn.Parameter(self.proj_space.transpose(-1,-2), requires_grad=False)

        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean predictions and covariance matrices
        """
        
        batch_size = x.size(0)
        # Reshape to deal with batching
        input_x = x.view(batch_size, -1)
        x = self.layers(input_x)

        # Get mean predictions
        means = self.fc(x)
        
        # Get Cholesky factor values - this has shape [batch_size, cholesky_size * horizon]
        cholesky_values = self.fc_cholesky(x)
        
        n = self.output_dim
        
        # Create an empty tensor for our lower triangular matrices
        L = torch.zeros(batch_size, self.horizon * n, self.horizon * n, device=x.device)
        
        # Convert the flat cholesky values to lower triangular matrices
        for b in range(batch_size):
            for h in range(self.horizon):
                start_idx = h * self.cholesky_size
                end_idx = (h + 1) * self.cholesky_size
                
                # Extract cholesky values for this horizon step
                step_values = cholesky_values[b, start_idx:end_idx]
                
                idx = 0
                for i in range(n):
                    for j in range(i + 1):
                        if i == j:
                            # Diagonal elements must be positive for a valid Cholesky factor
                            # Using softplus to ensure positivity
                            L[b, h * n + i, h * n + j] = F.softplus(step_values[idx]) + 1e-6
                        else:
                            # Off-diagonal elements can be any value
                            L[b, h * n + i, h * n + j] = step_values[idx]
                        idx += 1
        
        cov = L @ L.transpose(1, 2)
        cov_proj = self.fc_cholesky_adj(cov)  # Apply P
        cov_proj = cov_proj @ self.fc_cholesky_adj2(cov_proj)  # Apply P^T (P is symmetric)
              
        
        # cov_proj = self.fc_cholesky_adj(cov) + self.fc_cholesky_adj2(cov)  # Shape: [batch, horizon*n, horizon*n]
        self.epsilon = 1e-3
        # Regularize to ensure non-singularity
        identity = torch.eye(cov_proj.size(-1), device=cov_proj.device).unsqueeze(0).expand(batch_size, -1, -1)
        cov_proj_reg = cov_proj + self.epsilon * identity
        
        # Recompute Cholesky decomposition after regularization
        L_proj = torch.linalg.cholesky(cov_proj_reg)  # Differentiable Cholesky
        
        # Reshape to [batch, horizon, output_dim, output_dim]
        L_proj = L_proj.view(batch_size, self.horizon, self.output_dim, self.output_dim)
        
        means_adj = self.fc_adj1(means) + self.fc_adj2(input_x)
        # Reshape outputs to the expected format
        means_adj = means_adj.view(batch_size, self.horizon, self.output_dim)
        
        return means_adj, L_proj

     