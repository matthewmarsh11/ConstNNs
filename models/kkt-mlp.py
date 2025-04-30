from base import *
from typing import List, Optional, Tuple
import torch.nn.functional as F

activations = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid", "Softmax", "LogSoftmax"]

class MLP(BaseModel):
    """Unified MLP implementation supporting multiple uncertainty estimation methods"""
    def __init__(self, config: MLPConfig, input_dim: int, output_dim: int,
                 horizon: int, A: torch.Tensor, B: torch.Tensor, b: torch.Tensor):
        """Initialize MLP model with various uncertainty estimation capabilities
        
        Args:
            config: MLPConfig object containing model parameters
            input_dim: Input dimension: Number of features x time horizon
            output_dim: Output dimension
            horizon: Prediction horizon

        """
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        
        self.A = A
        self.B = B
        self.b = B
        
        
        
        # Validate activation function
        assert config.activation in activations, "Activation function not supported"
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
        self.fc = nn.Linear(current_dim, self.output_dim * self.horizon)
        
        # Lower triangular Cholesky factor raw values
        # We only need n*(n+1)/2 values for the lower triangular part
        n = self.output_dim
        self.cholesky_size = n * (n + 1) // 2
        self.fc_cholesky = nn.Linear(current_dim, self.cholesky_size * self.horizon)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean predictions and covariance matrices
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple of (means, covariances)
            means: Tensor of shape (batch_size, horizon, output_dim)
            covariances: Tensor of shape (batch_size, horizon, output_dim, output_dim)
        """
        batch_size = x.size(0)
        
        # Reshape to deal with batching
        x = x.view(batch_size, -1)
        x = self.layers(x)

        # Get mean predictions
        means = self.fc(x).view(batch_size, self.horizon, self.output_dim)
        
        # Get Cholesky factor values
        cholesky_values = self.fc_cholesky(x).view(batch_size, self.horizon, self.cholesky_size)
        
        # Convert to lower triangular matrices
        n = self.output_dim
        L = torch.zeros(batch_size, self.horizon, n, n, device=x.device)
        
        # Fill in the lower triangular part with the raw values
        for b in range(batch_size):
            for h in range(self.horizon):
                idx = 0
                for i in range(n):
                    for j in range(i + 1):
                        if i == j:
                            # Diagonal elements must be positive for a valid Cholesky factor
                            # Using softplus to ensure positivity
                            L[b, h, i, j] = F.softplus(cholesky_values[b, h, idx]) + 1e-6
                        else:
                            # Off-diagonal elements can be any value
                            L[b, h, i, j] = cholesky_values[b, h, idx]
                        idx += 1
        
        # # Compute covariance matrices from Cholesky factors: Σ = L*L^T
        # covariances = torch.matmul(L, L.transpose(-1, -2))
        
        # # Add small values to diagonal for numerical stability
        # eye = torch.eye(n, device=x.device).unsqueeze(0).unsqueeze(0)
        # covariances = covariances + 1e-6 * eye
        
        return means, L
