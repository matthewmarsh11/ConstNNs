import torch 
import torch.nn as nn
from typing import List, Optional
from base import *
class ConstrainedQuantileNN(BaseModel):
    """
    
    Neural Network for use with Quantile Regression, whereby the output
    is passed through non-trainable layer to enforce linear equality constraints
    

    Args:
        config: int, Configuration object of network specs
        input_dim: int, Dimension of input tensor
        output_dim: int, Dimension of output tensor
        A: torch.tensor: Constraint matrix: of shape (m, * input_dim) - m is the number of constraints
        B: Constraint matrix: of shape (m, * output_dim) - m is the number of constraints
        b: Constraint vector: of shape (m, 1) - m is the number of constraints
        quantiles: List[float]: List of quantiles to predict
    """
    
    def __init__(self, config: nn.Module, input_dim: int, output_dim: int, 
                 A: torch.Tensor, B: torch.Tensor, b: torch.Tensor,
                 quantiles: List[float]):
        super().__init__(config)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.quantiles = quantiles
        
        self.register_buffer("A", A.unsqueeze(0) if A.shape[0] == input_dim else A)
        self.register_buffer("B", B.unsqueeze(0) if B.shape[0] == output_dim else B)
        self.register_buffer("b", b)

        chunk = torch.mm(self.B.t(), torch.inverse(torch.mm(self.B, self.B.t())))
        Astar = -torch.mm(chunk, self.A)
        Bstar = torch.eye(output_dim, device=self.config.device) - torch.mm(chunk, self.B)
        bstar = torch.matmul(chunk, self.b).squeeze(-1)

        self.register_buffer("Astar", Astar)
        self.register_buffer("Bstar", Bstar)
        self.register_buffer("bstar", bstar)
        
        self.num_constraints = self.A.shape[0]
        
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
        
        self.fc = nn.Linear(current_dim, self.output_dim * len(self.quantiles))
        
        
        self.fc_fixed1 = nn.Linear(self.output_dim, self.output_dim)
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
    
        self.fc_fixed2 = nn.Linear(input_dim, self.output_dim)
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim * len(quantiles))
        """
        input_x = x
        
        # Pass through the layers
        x = self.layers(x)
        
        # Pass through the output layer
        x = self.fc(x)
        
        # Reshape to (batch_size, output_dim, len(quantiles))
        x = x.view(-1, self.output_dim, len(self.quantiles))
        out = torch.zeros_like(x)
        for i in range(len(self.quantiles)):
            out[:, :, i] = self.fc_fixed1(x[:, :, i]) + self.fc_fixed2(input_x)
        # Apply the fixed layers to enforce constraints

        
        return out, x # return the projected and non-projected samples