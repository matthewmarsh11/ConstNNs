import torch
import torch.nn as nn
from base import *
import torch.nn.functional as F
from typing import Optional, List

class EC_NN(BaseModel):
    
    """
    Neural Network with Equality Completion to project smaller dimension into larger one
    Args:
        config: int, Configuration object of network specs
        input_dim: int, Dimension of input tensor
        output_dim: int, Dimension of output tensor
        A: torch.tensor: Constraint matrix: of shape (m, * input_dim) - m is the number of constraints
        B: Constraint matrix: of shape (m, * output_dim) - m is the number of constraints
        b: Constraint vector: of shape (m, 1) - m is the number of constraints
        dependent_ids: List[int]: List of dependent variable indices
        num_samples: Optional[int]: Number of samples for Monte Carlo Dropout
    
    """
    
    def __init__(self, config: MLPConfig, input_dim: int, output_dim: int, 
                A: torch.Tensor, B: torch.Tensor, b: torch.Tensor,  dependent_ids: List, 
                num_samples: Optional[int] = None):

        super().__init__(config)
        
        self.input_dim = input_dim
        
        self.register_buffer("A", A.unsqueeze(0) if A.shape[0] == input_dim else A)
        self.register_buffer("B", B.unsqueeze(0) if B.shape[0] == output_dim else B)
        self.register_buffer("b", b)
        
        self.num_constraints = self.A.shape[0]
        self.output_dim = output_dim - self.num_constraints
        self.dependent_ids = dependent_ids
        self.num_samples = num_samples
        
        self.gnll = False
        # if we arent using samples, we need to use the gnll and hence output covariance
        if self.num_samples is None:
            self.gnll = True
        
        # Separate the dependent and independent variables
        
        assert self.num_constraints == len(self.dependent_ids), "Number of dependent variables does not match the number of constraints"
        
        # Create a list of all output indices
        all_output_indices = list(range(self.B.shape[1]))
        # Identify the independent indices
        independent_ids = [i for i in all_output_indices if i not in self.dependent_ids]

        # Split B according to dependent and independent indices
        self.B_dep = self.B[:, self.dependent_ids]
        self.B_indep = self.B[:, independent_ids]
        
        Astar = -torch.matmul(torch.inverse(self.B_dep), self.A)
        Bstar = -torch.matmul(torch.inverse(self.B_dep), self.B_indep)
        bstar = torch.matmul(torch.inverse(self.B_dep), self.b).squeeze(-1)

        self.register_buffer("Astar", Astar)
        self.register_buffer("Bstar", Bstar)
        self.register_buffer("bstar", bstar)
        
        self.num_constraints = self.A.shape[0]
        
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

        self.fc_fixed1 = nn.Linear(self.output_dim, self.num_constraints, bias=False)
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
        self.fc_fixed2 = nn.Linear(input_dim, self.num_constraints, bias=False)
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False)
        
        # Project the covariance matrix 
        if self.gnll:
            
            # lower triangular Cholesky factor raw values
            # n*(n+1)/2 values for the lower triangular part
            n = self.output_dim
            self.cholesky_size = n * (n + 1) // 2
            self.fc_cholesky = nn.Linear(current_dim, self.cholesky_size)
            
            # # Modify covariance projection to handle dimensions correctly
            # self.cov_proj1 = nn.Linear(self.output_dim, self.num_constraints, bias=False)
            # self.cov_proj2 = nn.Linear(self.output_dim, self.num_constraints, bias=False)
            
            # self.cov_proj1.weight = nn.Parameter(self.Bstar, requires_grad=False)
            # self.cov_proj2.weight = nn.Parameter(self.Bstar.T, requires_grad=False)

        if self.num_samples is not None:
            self.dropout_layers = []
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    self.dropout_layers.append(module)
        
    def enable_dropout(self):
        """
        Enable dropout layers
        """
        for module in self.dropout_layers:
            module.train()    
        
    def forward(self, x, num_samples: int = None):
        
        if num_samples is None:
            num_samples = self.num_samples

        if self.training or num_samples == None:
            # Standard forward pass
            batch_size = x.shape[0]
            input_x = x
            x = self.layers(x)
            indep_y = self.fc(x) #independent variable
            # can also return var here
            dep_y = self.fc_fixed1(indep_y) + self.fc_fixed2(input_x) #dependent variable projected
            
            # if self.gnll then we return the covariance matrix
            if self.gnll:    
                # get the 
                cholesky_values = self.fc_cholesky(x)
                # reconstruct the covariance matrix
                n = self.output_dim
            
                # Create an empty tensor for our lower triangular matrices
                L = torch.zeros(batch_size, n, n, device=x.device)
        
                # Convert the flat cholesky values to lower triangular matrices
                for b in range(batch_size):
                    idx = 0
                    for i in range(n):
                        for j in range(i + 1):
                            if i == j:
                                # Diagonal elements must be positive for a valid Cholesky factor
                                # Using softplus to ensure positivity
                                L[b, i, j] = F.softplus(cholesky_values[b, idx]) + 1e-6
                            else:
                                # Off-diagonal elements can be any value
                                L[b, i, j] = cholesky_values[b, idx]
                            idx += 1
                
                # --- Your existing code ---
                cov = L @ L.transpose(-1, -2) # Use -1 and -2 for last two dims, safer for batches
                # cov shape: [batch_size, Ny-m, Ny-m]

                # Correctly project the independent variable covariance
                # cov_indep_proj = Bstar @ cov @ Bstar.T
                temp = torch.matmul(self.Bstar, cov) # Shape: [batch_size, m, Ny-m]
                cov_dep = torch.matmul(temp, self.Bstar.T) # Shape: [batch_size, m, m]
                # Renamed cov_indep_proj to cov_dep for clarity (it's the covariance of the *dependent* variables)

                y = torch.cat([indep_y, dep_y], dim=1) # Shape: [batch_size, Ny]

                # --- Stacking the covariance matrices ---
                batch_size = cov.shape[0]
                ny_m = cov.shape[1]       # Dimension Ny - m
                m = cov_dep.shape[1]      # Dimension m
                ny = ny_m + m             # Total dimension Ny

                # Create the full covariance matrix using slicing
                # Initialize with zeros
                full_cov = torch.zeros(batch_size, ny, ny, dtype=cov.dtype, device=cov.device)

                # Place the diagonal blocks
                full_cov[:, :ny_m, :ny_m] = cov      # Top-left block
                full_cov[:, ny_m:, ny_m:] = cov_dep  # Bottom-right block

                # Now full_cov has shape [batch_size, Ny, Ny] and is block diagonal

                return y, full_cov
            
            else:
                return torch.stack([indep_y, dep_y], dim=0)
            
        else:
            # MC Dropout sampling
            dep_samples = [] # samples that are projected onto feasible set
            indep_samples = [] # samples that are not projected onto feasible set
            for _ in range(num_samples):
                # Enable dropout even in eval mode
                self.enable_dropout()
                x_sample = self.layers(x)
                indep_x_sample = self.fc(x_sample)
                indep_samples.append(indep_x_sample)
                dep_samples.append(self.fc_fixed1(indep_x_sample) + self.fc_fixed2(x))
            return torch.stack(dep_samples, dim=0), torch.stack(indep_samples, dim=0)