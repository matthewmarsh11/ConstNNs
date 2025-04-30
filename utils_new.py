import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union, NamedTuple
from dataclasses import dataclass
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.distributions import Normal
import math
import seaborn as sns

from tqdm import tqdm
from base import *
from cstr import *
# np.random.seed(42)
# torch.manual_seed(42)
from collections import defaultdict
from abc import ABC, abstractmethod
from models.mlp import *
from models.mcd_nn import *

class ScalingResult(NamedTuple):
    """Container for scaled mean and variance results"""
    mean: np.ndarray
    variance: np.ndarray

class DataProcessor:
    """Handles data processing and preparation"""
    def __init__(self, config: TrainingConfig, features: np.ndarray, targets: np.ndarray, num_simulations: int):
        self.config = config
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.features = features
        self.targets = targets
        self.num_simulations = num_simulations
        
    def prepare_data(self, simulation_length):
        """
        Prepare data for training organized by simulations
        
        Args:
            simulation_length: Number of time steps in each simulation
            
        Returns:
            DataLoaders and tensors organized for training by simulation
        """
        # Scale data
                
        num_simulations = len(self.features) // simulation_length
        
        X_tensor = torch.FloatTensor(self.features.to_numpy())
        y_tensor = torch.FloatTensor(self.targets.to_numpy())
        
        # Split simulations into train/test/val sets
        train_idx = int(num_simulations * self.config.train_test_split) * simulation_length
        val_idx = int(num_simulations * self.config.test_val_split) * simulation_length
        
        train_X = torch.FloatTensor(X_tensor[:train_idx, :])
        train_y = torch.FloatTensor(y_tensor[:train_idx])
        
        self.feature_scaler.fit(train_X)
        self.target_scaler.fit(train_y)
        
        train_X = torch.FloatTensor(self.feature_scaler.transform(train_X))
        train_y = torch.FloatTensor(self.target_scaler.transform(train_y))
        
        test_X = torch.FloatTensor(X_tensor[train_idx:val_idx])
        test_y = torch.FloatTensor(y_tensor[train_idx:val_idx])
        
        test_X = torch.FloatTensor(self.feature_scaler.transform(test_X))
        test_y = torch.FloatTensor(self.target_scaler.transform(test_y))
        
        val_X = torch.FloatTensor(X_tensor[val_idx:])
        val_y = torch.FloatTensor(y_tensor[val_idx:])
        
        val_X = torch.FloatTensor(self.feature_scaler.transform(val_X))
        val_y = torch.FloatTensor(self.target_scaler.transform(val_y))
        
        X_tensor = torch.FloatTensor(self.feature_scaler.transform(X_tensor))
        y_tensor = torch.FloatTensor(self.target_scaler.transform(y_tensor))
        
        
        return train_X, test_X, val_X, train_y, test_y, val_y, X_tensor, y_tensor

    def prepare_data_by_simulation(self, simulation_length):
        """
        Prepare data for training organized by simulations
        
        Args:
            simulation_length: Number of time steps in each simulation
            
        Returns:
            DataLoaders and tensors organized for training by simulation
        """
        # Scale data
        
        scaled_features = self.feature_scaler.fit_transform(self.features)
        scaled_targets = self.target_scaler.fit_transform(self.targets)
        num_simulations = len(scaled_features) // simulation_length
        
        X_tensor = torch.FloatTensor(scaled_features)
        y_tensor = torch.FloatTensor(scaled_targets)
        
        # Split simulations into train/test/val sets
        train_idx = int(num_simulations * self.config.train_test_split) * simulation_length
        val_idx = int(num_simulations * self.config.test_val_split) * simulation_length
        
        train_X = torch.FloatTensor(X_tensor[:train_idx, :]).view(-1, simulation_length, X_tensor.shape[1])
        train_y = torch.FloatTensor(y_tensor[:train_idx]).view(-1, simulation_length, y_tensor.shape[1])
        
        test_X = torch.FloatTensor(X_tensor[train_idx:val_idx]).view(-1, simulation_length, X_tensor.shape[1])
        test_y = torch.FloatTensor(y_tensor[train_idx:val_idx]).view(-1, simulation_length, y_tensor.shape[1])
        
        val_X = torch.FloatTensor(X_tensor[val_idx:]).view(-1, simulation_length, X_tensor.shape[1])
        val_y = torch.FloatTensor(y_tensor[val_idx:]).view(-1, simulation_length, y_tensor.shape[1])
        
        X_tensor = torch.FloatTensor(X_tensor).view(-1, simulation_length, X_tensor.shape[1])
        y_tensor = torch.FloatTensor(y_tensor).view(-1, simulation_length, y_tensor.shape[1])
        
        
        return train_X, test_X, val_X, train_y, test_y, val_y, X_tensor, y_tensor
    
class EarlyStopping:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self.has_valid_state = False
    
    def __call__(self, test_loss, model):
        # Check if the loss is valid (not NaN)
        if not np.isnan(test_loss):
            # Only update best state if we have a valid loss that's better than previous best
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.best_model_state = model.state_dict()
                self.has_valid_state = True
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.config.patience:
                    self.early_stop = True
        else:
            # In case of NaN, increment counter but don't update model state
            self.counter += 1
            if self.counter >= self.config.patience:
                self.early_stop = True
    
    def load_best_model(self, model):
        if self.has_valid_state and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        else:
            # If we never got a valid state, keep the current model state
            pass
    
    def get_best_loss(self):
        return self.best_loss if self.has_valid_state else float('inf')

class ModelTrainer:
    """Handles model training and evaluation"""
    def __init__(self, model: BaseModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
    def train(self, X_train, y_train, X_test, y_test, X_val, y_val, criterion):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay = self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.config.factor, patience=self.config.patience, verbose=True)
        early_stopping = EarlyStopping(self.config)
        history = {'train_loss': [], 'test_loss': [], 'val_loss': [], 'avg_loss': []}
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            if criterion.__class__.__name__ == 'GaussianMVNLL':
                train_loss = self._NLL_train_epoch(train_loader, criterion, optimizer)
            else:
                train_loss = self._train_epoch(train_loader, criterion, optimizer)
                
            # Validation
            self.model.eval()
            self.model.enable_dropout()
            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)                
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            if criterion.__class__.__name__ == 'GaussianMVNLL':
                test_loss = self._NLL_validate_epoch(test_loader, criterion)
                val_loss = self._NLL_validate_epoch(val_loader, criterion)
            else:
                test_loss = self._validate_epoch(test_loader, criterion)
                val_loss = self._validate_epoch(val_loader, criterion)

            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_loss = (avg_train_loss + avg_test_loss) / 2
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['test_loss'].append(avg_test_loss)
            history['val_loss'].append(avg_val_loss)
            history['avg_loss'].append(avg_loss)
            
            # Use average loss for scheduler
            scheduler.step(avg_loss)
            
            print(f'Epoch [{epoch+1}/{self.config.num_epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, ' 
                f'Val Loss: {avg_val_loss:.4f}'
                f'Avg Loss: {avg_loss:.4f}')
            
            # early_stopping(avg_loss, self.model)  # Use average loss for early stopping
            # if early_stopping.early_stop:
            #     print("Early Stopping")
            #     break
            # early_stopping.load_best_model(self.model)
                
        return self.model, history, avg_loss        

    def train_sim(self, X_train, y_train, X_test, y_test, X_val, y_val, criterion):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay = self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.config.factor, patience=self.config.patience, verbose=True)
        early_stopping = EarlyStopping(self.config)
        history = {'train_loss': [], 'test_loss': [], 'val_loss': [], 'avg_loss': []}
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            
            for i in range(X_train.shape[0]):
                X_sim = X_train[i].to(self.device)
                y_sim = y_train[i].to(self.device)
                train_dataset = TensorDataset(X_sim, y_sim)
                train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
                if criterion.__class__.__name__ == 'GaussianMVNLL':
                    train_loss = self._NLL_train_epoch(train_loader, criterion, optimizer)
                else:
                    train_loss = self._train_epoch(train_loader, criterion, optimizer)
                
            # Validation
            self.model.eval()
            self.model.enable_dropout()
            for i in range(X_test.shape[0]):
                X_test_sim = X_test[i].to(self.device)
                y_test_sim = y_test[i].to(self.device)
                
                test_dataset = TensorDataset(X_test_sim, y_test_sim)
                test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
                
                X_val_sim = X_val[i].to(self.device)
                y_val_sim = y_val[i].to(self.device)
                
                val_dataset = TensorDataset(X_val_sim, y_val_sim)
                val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
                
                if criterion.__class__.__name__ == 'GaussianMVNLL':
                    test_loss = self._NLL_validate_epoch(test_loader, criterion)
                    val_loss = self._NLL_validate_epoch(val_loader, criterion)
                else:
                    test_loss = self._validate_epoch(test_loader, criterion)
                    val_loss = self._validate_epoch(val_loader, criterion)

            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_loss = (avg_train_loss + avg_test_loss) / 2
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['test_loss'].append(avg_test_loss)
            history['val_loss'].append(avg_val_loss)
            history['avg_loss'].append(avg_loss)
            
            # Use average loss for scheduler
            scheduler.step(avg_loss)
            
            print(f'Epoch [{epoch+1}/{self.config.num_epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, ' 
                f'Val Loss: {avg_val_loss:.4f}'
                f'Avg Loss: {avg_loss:.4f}')
            
            early_stopping(avg_loss, self.model)  # Use average loss for early stopping
            if early_stopping.early_stop:
                print("Early Stopping")
                break
            early_stopping.load_best_model(self.model)
                
        return self.model, history, avg_loss
    
    def _train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer) -> float:
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            predictions, _ = self.model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss
    
    def _NLL_train_epoch(self, train_loader: DataLoader, criterion: nn.Module,
                        optimizer: torch.optim.Optimizer) -> float:
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            mean, var = self.model(batch_X)
            loss = criterion(mean, batch_y, var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss
    
    def _validate_epoch(self, test_loader: DataLoader, criterion: nn.Module) -> float:
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions, _ = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                total_loss += loss.item()
        return total_loss

    def _NLL_validate_epoch(self, test_loader: DataLoader, criterion: nn.Module) -> float:
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                mean, var = self.model(batch_X)
                loss = criterion(mean, batch_y, var)
                total_loss += loss.item()
        return total_loss

class Visualizer:
    """Handles visualization of results."""
    @staticmethod
    def plot_preds(preds: Union[np.ndarray, Dict[float, np.ndarray]],    
                         noisy_data, noiseless_data: Union[np.ndarray, None], sequence_length: int,
                         time_horizon: int, feature_names: list, num_simulations: int, 
                         train_test_split: float, test_val_split: float,
                         vars: Optional[np.ndarray] = None):
        
        # If split the predictions into the train set and test set
        # If it is a dictionary (quantiles) need to account for this
        if isinstance(preds, np.ndarray):
            tt_idx = int(train_test_split * len(preds))
            tv_idx = int(test_val_split * len(preds))
        else:
            tt_idx = int(train_test_split * len(preds[0.5]))
            tv_idx = int(test_val_split * len(preds[0.5]))
                    
        pred_new = None
        
        feature_names = [f"{feature} Sim {i+1}" for feature in feature_names for i in range(num_simulations)]

        # Now have to deal with the case where the predictions are quantiles (dictionary)
        
        if isinstance(preds, dict):
            # Iterate through each quantile key and split the predictions
            tt_idx = int(train_test_split * len(preds[0.5]))
            tv_idx = int(test_val_split * len(preds[0.5]))

            pred_new = preds
            preds = preds[0.5]


        for i, sim in enumerate(feature_names):
            plt.figure(figsize=(10, 6))
            
            
            # Plot training predictions and ground truth
            plt.plot(range(sequence_length, sequence_length + len(preds[:tt_idx, i])),
                preds[:tt_idx, i], label=f'{sim} Train Predictions', color='blue', alpha=0.7)
            test_offset = sequence_length + len(preds[:tt_idx, i])
            plt.plot(range(test_offset, test_offset + len(preds[tt_idx:tv_idx, i])), 
                    preds[tt_idx:tv_idx, i], label=f'{sim} Test Predictions', color='red', alpha=0.7)
            val_offset = sequence_length + len(preds[:tt_idx, i]) + len(preds[tt_idx:tv_idx, i])
            plt.plot(range(val_offset, val_offset + len(preds[tv_idx:, i])), 
                    preds[tv_idx:, i], label=f'{sim} Val Predictions', color='cyan', alpha=0.7)
            
            plt.plot(noisy_data[:, i], label=f'{sim} Noisy Simulation', color='green', alpha=0.7)
            if noiseless_data is not None:
                plt.plot(noiseless_data[:, i], label=f'{sim} Noiseless Data', color='black', linestyle = 'dashed', alpha=0.7)
            
            plt.title(f'{sim} Predictions')
            plt.xlabel('Time Step')
            plt.ylabel(sim)
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
            
            if vars is not None:
                plt.fill_between(range(sequence_length, sequence_length + len(preds[:tt_idx, i])),
                                preds[:tt_idx, i] - np.sqrt(vars[:tt_idx, i]), preds[:tt_idx, i] + np.sqrt(vars[:tt_idx, i]),
                                color='blue', alpha=0.2, edgecolor = 'None',label='Train Uncertainty')
                plt.fill_between(range(test_offset, test_offset + len(preds[tt_idx:tv_idx, i])),
                                preds[tt_idx:tv_idx, i] - np.sqrt(vars[tt_idx:tv_idx, i]), preds[tt_idx:tv_idx, i] + np.sqrt(vars[tt_idx:tv_idx, i]),
                                color='red', alpha=0.2, edgecolor = 'None', label='Test Uncertainty')
                plt.fill_between(range(val_offset, val_offset + len(preds[tv_idx:, i])),
                                preds[tv_idx:, i] - np.sqrt(vars[tv_idx:, i]), preds[tv_idx:, i] + np.sqrt(vars[tv_idx:, i]),
                                color='cyan', alpha=0.2, edgecolor = 'None', label='Val Uncertainty')
            # Plot the dictionary of quantiles as the uncertainty
            if pred_new:
                keys = pred_new.keys()
                max_key = max(keys)
                min_key = min(keys)
                plt.fill_between(range(sequence_length, sequence_length + len(preds[:tt_idx, i])),
                                pred_new[min_key][:tt_idx, i], pred_new[max_key][:tt_idx, i], color='blue', alpha=0.2, edgecolor = 'None', label='Train Uncertainty')
                plt.fill_between(range(test_offset, test_offset + len(preds[tt_idx:tv_idx, i])), 
                                pred_new[min_key][tt_idx:tv_idx, i], pred_new[max_key][tt_idx:tv_idx, i], color='red', alpha=0.2, edgecolor = 'None',label='Test Uncertainty')
                plt.fill_between(range(val_offset, val_offset + len(preds[tv_idx:, i])), 
                                pred_new[min_key][tv_idx:, i], pred_new[max_key][tv_idx:, i], color='cyan', alpha=0.2, edgecolor = 'None',label='Val Uncertainty')
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_NLL(preds: Dict[int, np.ndarray], vars: Dict[int, np.ndarray],
                noisy_data: np.ndarray, noiseless_data: Union[np.ndarray, None],
                sequence_length: int, feature_names: list, num_simulations: int, 
                train_test_split: float, test_val_split: float):
        
        tt_idx = int(train_test_split * len(preds))
        tv_idx = int(test_val_split * len(preds))

        # For each simulation
        for sim_idx in preds.keys():
            # Get number of features from the prediction shape
            num_features = preds[sim_idx].shape[1] if len(preds[sim_idx].shape) > 1 else 1
            
            # Create figure with subplots (one for each feature)
            fig, axes = plt.subplots(num_features, 1, figsize=(8, 2 * num_features), sharex=True)
            fig.suptitle(f'Simulation {sim_idx + 1} Predictions', fontsize=16)
            
            # Convert axes to array if there's only one feature
            if num_features == 1:
                axes = np.array([axes])
            
            # For each feature
            for feature_idx in range(num_features):
                ax = axes[feature_idx]
                
                # Get time steps for x-axis
                time_steps = range(sequence_length, sequence_length + preds[sim_idx].shape[0])
                
                # Extract predictions for this feature
                if num_features == 1:
                    feature_preds = preds[sim_idx]
                else:
                    feature_preds = preds[sim_idx][:, feature_idx]
                
                # Plot predictions
                ax.plot(time_steps, feature_preds, 
                        label='Predictions', color='blue', alpha=0.7)
                
                # Plot uncertainty if available
                if vars is not None:
                    if num_features == 1:
                        feature_vars = vars[sim_idx]
                    else:
                        feature_vars = vars[sim_idx][:, feature_idx]
                    
                    ax.fill_between(time_steps,
                                    feature_preds - np.sqrt(feature_vars), 
                                    feature_preds + np.sqrt(feature_vars),
                                    color='blue', alpha=0.2, edgecolor='None', 
                                    label='Prediction Uncertainty')
                
                # Plot noisy data (ground truth)
                if len(noisy_data.shape) == 3:  # (sim, time, feature)
                    ax.plot(noisy_data[sim_idx, :, feature_idx], 
                            label='Noisy Data', color='green', alpha=0.7)
                else:  # Handle other formats if needed
                    ax.plot(noisy_data[sim_idx], 
                            label='Noisy Data', color='green', alpha=0.7)
                
                # Plot noiseless data if available
                if noiseless_data is not None:
                    if len(noiseless_data.shape) == 3:  # (sim, time, feature)
                        ax.plot(noiseless_data[sim_idx, :, feature_idx], 
                                label='Noiseless Data', color='black', 
                                linestyle='dashed', alpha=0.7)
                    else:  # Handle other formats if needed
                        ax.plot(noiseless_data[sim_idx], 
                                label='Noiseless Data', color='black', 
                                linestyle='dashed', alpha=0.7)
                
                # Set labels and legend
                ax.set_ylabel(feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature {feature_idx}')
                ax.legend(loc='upper right')
                
                # Only set x-label for the bottom subplot
                if feature_idx == num_features - 1:
                    ax.set_xlabel('Time Step')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)  # Make room for the suptitle
            plt.show()

    @staticmethod
    def plot_loss(history: Dict[float, np.ndarray]):
        """Plots the loss history for a model"""
        plt.figure(figsize=(10, 6))
        for key, loss in history.items():
            plt.plot(loss, label=key)
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        

def main():
    
    training_config = TrainingConfig(
        batch_size=50,
        num_epochs=50,
        learning_rate=0.0031,
        weight_decay=0.01,
        factor=0.1,
        patience=10,
        delta = 0.042,
        train_test_split=0.6,
        test_val_split=0.8,
        # device = "mps" if torch.backends.mps.is_available() else "cpu",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )
    
    MLP_Config = MLPConfig(
        hidden_dim = 1024,
        num_layers = 3,
        dropout = 0.2,
        activation = 'ReLU',
        # device = "mps" if torch.backends.mps.is_available() else "cpu",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )
    
    features_path = 'datasets/small_cstr_features.csv'
    targets_path = 'datasets/small_cstr_targets.csv'
    noiseless_path = 'datasets/small_cstr_noiseless_results.csv'
    
    features = pd.read_csv(features_path)
    features = features.iloc[:, :-1]
    targets = features.iloc[:, :4]
    features = features.iloc[:, 4:]
    noiseless_results = pd.read_csv(noiseless_path)
    noiseless_results = noiseless_results.iloc[:, :-1]
    noiseless_targets = noiseless_results.iloc[:, :4].to_numpy()
    noiseless_features = noiseless_results.iloc[:, 4:].to_numpy()
    num_simulations = 10
    
    data_processor = DataProcessor(training_config, features, targets, num_simulations)
    # Prepare data
    (X_train, X_test, X_val, y_train, y_test, y_val, X_tensor, y_tensor) = data_processor.prepare_data(simulation_length=99)
    # noiseless_targets = noiseless_targets.reshape(X_tensor.shape[0], X_tensor.shape[1], -1)
    # noisy_targets = targets.to_numpy().reshape(X_tensor.shape[0], X_tensor.shape[1], -1)
    # features = features.to_numpy().reshape(X_tensor.shape[0], X_tensor.shape[1], -1)
    # of shape [simulations, time steps, features]
    
    # X = [Tin, Caf, Tc]
    # y = [Caf, Cb, Cc, T]
    
    # Enforce mass balance: Cain = Ca + 2Cb + Cc
    A = torch.Tensor([0 , -1 , 0])
    B = torch.Tensor([1, 2, 1, 0])
    b = torch.Tensor([0])
    
    device = MLP_Config.device

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    A = A.to(device)
    B = B.to(device)
    b = b.to(device)
    
    model = MCD_NN(
        config = MLP_Config,
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        num_samples = 1000,
        A = A,
        B = B,
        b = b,
    )
    
    from mv_gaussian_nll import GaussianMVNLL
    # criterion = GaussianMVNLL()
    criterion = nn.MSELoss()
    
    trainer = ModelTrainer(model, training_config)
    model, history, avg_loss = trainer.train(X_train, y_train, X_test, y_test, X_val, y_val, criterion)
    X_tensor = X_tensor.reshape(num_simulations, -1, X_tensor.shape[1])
    noisy_targets = targets.to_numpy().reshape(num_simulations, -1, y_tensor.shape[1])
    noiseless_targets = noiseless_targets.reshape(num_simulations, -1, y_tensor.shape[1])
    features = features.to_numpy().reshape(num_simulations, -1, X_tensor.shape[2])
    noiseless_features = noiseless_features.reshape(num_simulations, -1, X_tensor.shape[2])
    
    model.eval()
    model.enable_dropout()
    feature_names = ['ca', 'cb', 'cc', 'temp', 'vol']
    simulations = {}
    simulations = {i: None for i in range(X_tensor.shape[0])}
    np_simulations = {}
    np_simulations = {i: None for i in range(X_tensor.shape[0])}
    for i in range (X_tensor.shape[0]):
        with torch.no_grad():        
            prj_preds, nprj_preds = model(X_tensor[i, :, :].to(training_config.device))
            simulations[i] = prj_preds.cpu().numpy()
            np_simulations[i] = nprj_preds.cpu().numpy()

    simulation_1 = simulations[0]
    np_simulation_1 = np_simulations[0]
    
    # plot the error distribution
    
    for j in range(simulation_1.shape[1]):
        simulation_1[:, j, :] = data_processor.target_scaler.inverse_transform(simulation_1[:, j, :])
        np_simulation_1[:, j, :] = data_processor.target_scaler.inverse_transform(np_simulation_1[:, j, :])
        # Plot distribution of simulation_1 with overlay of np_simulation_1
        # plt.figure(figsize=(15, 10))
        # for i in range(simulation_1.shape[2]):  # For each feature
        #     plt.subplot(math.ceil(simulation_1.shape[2]/2), 2, i+1)
        #     # Create histogram with KDE for both projected and non-projected data
        #     sns.histplot(simulation_1[:, j, i], kde=True, color='blue', alpha=0.6, label='Projected')
        #     sns.histplot(np_simulation_1[:, j, i], kde=True, color='red', alpha=0.6, label='Non-Projected')
        #     plt.axvline(noiseless_targets[0, j, i], color='black', linestyle='dashed', label='Noiseless Actual')
        #     plt.title(f'Distribution of {feature_names[i]} at time {j}')
        #     plt.xlabel(f'{feature_names[i]} value')
        #     plt.ylabel('Frequency')
        #     plt.legend()
        # plt.tight_layout()
        # plt.show()
    
    # plot the trajectory of the simulation
    
    sim_mean = simulation_1.mean(axis = 0)
    # sim_mean = data_processor.target_scaler.inverse_transform(sim_mean)
    np_sim_mean = np_simulation_1.mean(axis = 0)
    # np_sim_mean = data_processor.target_scaler.inverse_transform(np_sim_mean)
    plt.figure(figsize=(15, 10))
    for i in range(sim_mean.shape[1]):
        plt.subplot(math.ceil(sim_mean.shape[1]/2), 2, i+1)
        plt.plot(noisy_targets[0, :, i], label = 'Noisy Data', color = 'green')
        plt.plot(noiseless_targets[0, :, i], label = 'Noiseless Data', color = 'black', linestyle = 'dashed')
        plt.plot(sim_mean[:, i], label = feature_names[i])
        plt.plot(np_sim_mean[:, i], label = f'Non-Projected {feature_names[i]}', linestyle = 'dashed')
        plt.title(f'Mean of {feature_names[i]}')
        plt.xlabel('Time step')
        plt.ylabel(f'{feature_names[i]} value')
        plt.legend()
    plt.tight_layout()
    plt.show()
    
    constraint = np.zeros((sim_mean.shape[0], 1))
    unprojected_constraint = np.zeros((sim_mean.shape[0], 1))
    constraint_true = np.zeros((sim_mean.shape[0], 1))
    # plot the constraint violation Ax + By - b = 0
    for i in range(sim_mean.shape[0]):
        constraint[i] = A @ noiseless_features[0, i, :]  + B @ sim_mean[i, :] - b
        unprojected_constraint[i] = A @ noiseless_features[0, i, :]  + B @ np_sim_mean[i, :] - b
        constraint_true[i] = A @ noiseless_features[0, i, :]  + B @ noiseless_targets[0, i, :] - b
    
    plt.figure(figsize=(15, 10))
    plt.plot(constraint, label = f'Constraint violation at time {i}')
    plt.plot(unprojected_constraint, label = f'Non-Projected Constraint violation at time {i}', linestyle = 'dashed')
    plt.axhline(0, color = 'black', linestyle = 'dashed')
    plt.plot(constraint_true, label = f'True Constraint violation at time {i}', linestyle = 'dashed')
    plt.legend()
    
    action_names = ['inlet temp', 'feed conc', 'coolant temp']
    visualizer = Visualizer()

    
    visualizer.plot_loss(history)
    
if __name__ == "__main__":
    main()