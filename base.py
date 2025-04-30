from dataclasses import dataclass
import torch.nn as nn
from abc import ABC, abstractmethod
import torch

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    factor: float
    patience: int
    delta: float
    train_test_split: float = 0.6
    test_val_split: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class MLPConfig:
    """Configuration for MLP model"""
    hidden_dim: int
    num_layers: int
    dropout: float = 0.2
    activation: str = 'ReLU'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models"""
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    
    
    