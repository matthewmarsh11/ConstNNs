from base import *
from typing import List, Optional, Tuple
import torch.nn.functional as F

activations = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid", "Softmax", "LogSoftmax"]

class MLP(BaseModel):
    """Unified MLP implementation supporting multiple uncertainty estimation methods"""
    def __init__(self, config: MLPConfig, input_dim: int, output_dim: int,
                 num_samples: Optional[int] = None):
        """Initialize MLP model with various uncertainty estimation capabilities
        
        Args:
            config: MLPConfig object containing model parameters
            input_dim: Input dimension: Number of features x time horizon
            output_dim: Output dimension
            num_samples: Optional[int]: Number of samples for Monte Carlo Dropout, if None defaults to outputting covariance matrix
        """
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_samples = num_samples
        
        self.gnll = False
        # If we aren't using samples, we need to use the gnll and hence output covariance
        if self.num_samples is None:
            self.gnll = True
            

            
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
        self.fc = nn.Linear(current_dim, self.output_dim)
        
        # Lower triangular Cholesky factor raw values
        # We only need n*(n+1)/2 values for the lower triangular part
        if self.gnll:
            n = self.output_dim
            self.cholesky_size = n * (n + 1) // 2
            self.fc_cholesky = nn.Linear(current_dim, self.cholesky_size)
            
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
                
    def forward(self, x: torch.Tensor, num_samples: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean predictions and covariance matrices
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple of (means, covariances)
            means: Tensor of shape (batch_size, horizon, output_dim)
            covariances: Tensor of shape (batch_size, horizon, output_dim, output_dim)
        """
        if num_samples is None:
            num_samples = self.num_samples
            
        if self.training or num_samples == None:
            # Standard forward pass
            batch_size = x.shape[0]
            input_x = x
            x = self.layers(x)
            out = self.fc(x) #independent variable
            
            
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
                


                return out, L
            else: return out
            
        else:
            out_samples = []
            for _ in range(num_samples):
                self.enable_dropout
                x = self.layers(x)
                out = self.fc(x)
                out_samples.append()
            return out_samples
            
