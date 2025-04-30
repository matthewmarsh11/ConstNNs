
from typing import List, Tuple, Dict, Any
import numpy as np
import torch.nn as nn
import torch
from base import *

class lf_PINN(BaseModel):
    
    def __init__(self, config, input_dim: int, output_dim: int):
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return x
    

# Train it and add a penalty term to the loss function