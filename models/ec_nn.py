import torch
import torch.nn as nn
from base import *

class EC_NN(BaseModel):
    
    """
    
    Neural Network with Equality Completion to project smaller dimension into larger one

    
    """
    
    def __init__(self, config: MLPConfig, input_dim: int, output_dim: int,
                num_samples: int, A: torch.Tensor, B: torch.Tensor, b: torch.Tensor):
    
        super().__init__(config)
        
        self.input_dim = input_dim
        self.num_constraints = A.shape[0]
        self.output_dim = output_dim - self.num_constraints
        self.num_samples = num_samples
        
        self.register_buffer("A", A.unsqueeze(0) if A.shape[0] == input_dim else A)
        self.register_buffer("B", B.unsqueeze(0) if B.shape[0] == output_dim else B)
        self.register_buffer("b", b)
        # Separate the dependent and independent variables
        self.B_indep = self.B[:, :self.output_dim]
        self.B_dep = self.B[:, self.output_dim:]
        
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

        self.fc_fixed1 = nn.Linear(self.output_dim, self.num_constraints, bias=False)  ########
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
        self.fc_fixed2 = nn.Linear(input_dim, self.num_constraints, bias=False)  ########
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
            indep_x = self.fc(x) #independent variable
            # can also return var here
            dep_x = self.fc_fixed1(indep_x) + self.fc_fixed2(input_x) #dependent variable projected
            
            return indep_x, dep_x # return the projected and non-projected samples
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