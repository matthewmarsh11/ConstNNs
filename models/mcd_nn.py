import torch
import torch.nn as nn
from base import *

class MCD_NN(BaseModel):
    
    """
    
    Neural Network with Monte Carlo Dropout for uncertainty quantification, with projection of constraints on every single sample

    
    """
    
    def __init__(self, config: MLPConfig, input_dim: int, output_dim: int,
                num_samples: int, A: torch.Tensor, B: torch.Tensor, b: torch.Tensor):
    
        super().__init__(config)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_samples = num_samples
        
        self.A = A
        self.B = B
        self.b = b

        if self.A.shape[0] == self.input_dim:
            self.A = self.A.unsqueeze(0)
        if self.B.shape[0] == self.output_dim:
            self.B = self.B.unsqueeze(0)
        
        self.chunk = torch.mm(self.B.t(),
                             torch.inverse(
                                 torch.mm(self.B, self.B.t())
                             )
                             )
        self.Astar = - torch.mm(self.chunk, self.A)
        self.Bstar = torch.eye(output_dim).to(self.config.device) - torch.mm(self.chunk, self.B)
        self.bstar = torch.matmul(self.chunk, self.b).squeeze(-1)    
        
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

        self.fc_fixed1 = nn.Linear(self.output_dim, self.output_dim, bias=False)  ########
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
        self.fc_fixed2 = nn.Linear(input_dim, self.output_dim, bias=False)  ########
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False)
        
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

        if self.training or num_samples == 1:
            # Standard forward pass
            input_x = x
            x = self.layers(x)
            x = self.fc(x)
            prj = self.fc_fixed1(x) + self.fc_fixed2(input_x)
            
            return prj, x # return the projected and non-projected samples
        else:
            # MC Dropout sampling
            prj_samples = [] # samples that are projected onto feasible set
            np_samples = [] # samples that are not projected onto feasible set
            for _ in range(num_samples):
                # Enable dropout even in eval mode
                self.enable_dropout()
                x_sample = self.layers(x)
                x_sample = self.fc(x_sample)
                np_samples.append(x_sample)
                prj_samples.append(self.fc_fixed1(x_sample) + self.fc_fixed2(x))
            return torch.stack(prj_samples, dim=0), torch.stack(np_samples, dim=0)