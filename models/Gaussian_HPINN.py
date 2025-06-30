import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Union
from base import *
from scipy.stats import chi2
import numpy as np
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
        
        self.chi2_thresh = chi2.ppf(self.probability_level, df=1)
        
        self.register_buffer("A", A.unsqueeze(0) if A.shape[0] == input_dim else A)
        self.register_buffer("B", B.unsqueeze(0) if B.shape[0] == output_dim else B)
        self.register_buffer("b", b)
        
        self.num_constraints = self.A.shape[0]
        
        # Check if epsilon is a scalar
        if isinstance(self.epsilon, (int, float)):
            self.epsilon = torch.full((self.num_constraints, 1), self.epsilon, device=config.device)
            self.epsilon = self.epsilon.squeeze(-1)  # Make it a vector
        # elif self.epsilon.dim() == 1:
        #     self.epsilon = self.epsilon.unsqueeze(1)
        elif self.epsilon.shape != self.b.shape:
            raise ValueError(f"Epsilon must be a scalar, vector, or have the same shape as b ({self.b.shape})")
        
        inv_BBT = torch.inverse(torch.matmul(self.B, self.B.T)) # (BB^T)^-1, shape (num_constraints, num_constraints)
        BT_inv_BBT = torch.matmul(self.B.T, inv_BBT) # B^T * (BB^T)^-1, shape (output_dim, num_constraints)
        
        
        Astar = -torch.matmul(BT_inv_BBT, self.A)
        Bstar = torch.eye(self.output_dim, device=config.device) - torch.matmul(BT_inv_BBT, self.B)
        bstar = torch.matmul(BT_inv_BBT, self.b).squeeze(-1)
        
        Cstar = 1/self.chi2_thresh * torch.matmul(self.B.T, inv_BBT) * self.epsilon

        self.register_buffer("Astar", Astar)
        self.register_buffer("Bstar", Bstar)
        self.register_buffer("bstar", bstar)
        self.register_buffer("Cstar", Cstar)
        

        
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
        
        # LDL decomposition: predict L (lower triangular with 1s on diagonal) and D (diagonal)
        self.num_lower_triangular = self.output_dim * (self.output_dim - 1) // 2  # strictly lower triangular
        self.fc_L = nn.Linear(current_dim, self.num_lower_triangular)  # lower triangular elements
        self.fc_D = nn.Linear(current_dim, self.output_dim)  # diagonal elements
        
        self.fc_fixed1 = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
        self.fc_fixed2 = nn.Linear(input_dim, self.output_dim, bias=False)
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False)
        
        self.var_fixed1 = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.var_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
        self.var_fixed2 = nn.Linear(self.epsilon.shape[0], self.output_dim, bias=False)
        self.var_fixed2.weight = nn.Parameter(self.Cstar, requires_grad=False)
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        z = self.layers(x)
        mu = self.fc(z)

        # Get LDL decomposition components
        L_raw_elements = self.fc_L(z)  # shape: (batch_size, num_lower_triangular)
        D_raw_elements = self.fc_D(z)  # shape: (batch_size, output_dim)

        # Get strictly lower triangular indices (below diagonal)
        tril_indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=-1, device=x.device)
        
        L_batch = torch.zeros(batch_size, self.output_dim, self.output_dim, device=x.device)
        sigma_P = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        for i in range(batch_size):
            # Create L matrix with 1s on diagonal
            L = torch.eye(self.output_dim, device=x.device)
            L[tril_indices[0], tril_indices[1]] = L_raw_elements[i]
            
            # Create D matrix (diagonal) with positive entries
            D_diag = torch.nn.functional.softplus(D_raw_elements[i]) + 1e-6
            
            # Compute Cholesky factor from LDL: C = L * sqrt(D)
            sqrt_D = torch.sqrt(D_diag)
            C = L * sqrt_D.unsqueeze(0)  # broadcast sqrt_D to multiply each row
            
            L_batch[i] = C
            sigma_P[i] = sqrt_D  # diagonal of the Cholesky factor

        
        # project the mean onto the constraint set
        mu_Q = self.fc_fixed1(mu) + self.fc_fixed2(x)
        
        # project the covariance matrix onto the constraint set
        # we output the lower-triangular matrix, where \sigma_P are the diagonal elements
        # project the standard deviation
        sigma_Q = self.var_fixed1(sigma_P) + self.var_fixed2(self.epsilon)

        L_out = L_batch.clone()
        diag_idx = torch.arange(self.output_dim, device=x.device)
        for i in range(batch_size):
            L_out[i, diag_idx, diag_idx] = sigma_Q[i]

        return mu_Q, L_out
        