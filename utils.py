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

from tqdm import tqdm
from base import *
from cstr import *
np.random.seed(42)
torch.manual_seed(42)
from collections import defaultdict
from abc import ABC, abstractmethod
from models.mlp import *
from mp_nn import mpi_nn

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

    def prepare_sequences(self, scaled_features, scaled_targets) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(len(scaled_features) - self.config.time_step - self.config.horizon):
            # Input sequence remains the same
            X.append(scaled_features[i:i + self.config.time_step])
            
            # Get future values for all target variables
            future_values = scaled_targets[i + self.config.time_step:i + self.config.time_step + self.config.horizon]
            y.append(future_values)
        return np.array(X), np.array(y)

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training"""
        # Scale data
        scaled_features = self.feature_scaler.fit_transform(self.features)
        scaled_targets = self.target_scaler.fit_transform(self.targets)

        # Create sequences
        X, y = self.prepare_sequences(scaled_features, scaled_targets)
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Split data
        tt_idx = int(len(X_tensor) * self.config.train_test_split)
        tv_idx = int(len(X_tensor) * self.config.test_val_split)
        
        X_train = X_tensor[:tt_idx]
        X_test = X_tensor[tt_idx:tv_idx]
        X_val = X_tensor[tv_idx:]
        y_train = y_tensor[:tt_idx]
        y_test = y_tensor[tt_idx:tv_idx]
        y_val = y_tensor[tv_idx:]
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, test_loader, val_loader, X_train, X_test, X_val, y_train, y_test, y_val, X_tensor, y_tensor


    def prepare_data_surrogate(self, num_simulations: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for training, testing, and validation split by simulation columns
        
        Args:
            num_features: Number of state features per simulation
            num_simulations: Total number of simulations in the dataset
        
        Returns:
            Tuple of tensors for training, testing, and validation sets
        """
        # Scale data
        
        
        self.features = self.feature_scaler.fit_transform(self.features)
        self.targets = self.target_scaler.fit_transform(self.targets)

        # Calculate split indices based on percentages
        train_split = int(self.features.shape[0] * self.config.train_test_split)
        test_split = int(self.features.shape[0] * self.config.test_val_split)
                
        X_train = self.features[:train_split, :]
        y_train = self.targets[:train_split, :]
        X_test = self.features[train_split:test_split, :]
        y_test = self.targets[train_split:test_split, :]
        X_val = self.features[test_split:, :]
        y_val = self.targets[test_split:, :]
        
        X, y = self.prepare_sequences(self.features, self.targets)
        X_train, y_train = self.prepare_sequences(X_train, y_train)
        X_test, y_test = self.prepare_sequences(X_test, y_test)
        X_val, y_val = self.prepare_sequences(X_val, y_val)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        return train_loader, test_loader, val_loader, X_train_tensor, X_test_tensor, X_val_tensor, y_train_tensor, y_test_tensor, y_val_tensor, X_tensor, y_tensor


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
        
        X = np.zeros((num_simulations, simulation_length, scaled_features.shape[1]))
        y = np.zeros((num_simulations, simulation_length, scaled_targets.shape[1]))
        X_seq = np.zeros((num_simulations, simulation_length - self.config.time_step - self.config.horizon, self.config.time_step, scaled_features.shape[1]))
        y_seq = np.zeros((num_simulations, simulation_length - self.config.time_step - self.config.horizon, self.config.horizon, scaled_targets.shape[1]))
        
        for i in range(num_simulations):
            start_idx = i * simulation_length
            end_idx = start_idx + simulation_length
            X[i] = scaled_features[start_idx:end_idx, :]
            y[i] = scaled_targets[start_idx:end_idx, :]
            X_seq[i], y_seq[i] = self.prepare_sequences(X[i], y[i])

        
        # Split simulations into train/test/val sets
        train_idx = int(num_simulations * self.config.train_test_split)
        val_idx = int(num_simulations * self.config.test_val_split)
        
        train_X = torch.FloatTensor(X_seq[:train_idx])
        train_y = torch.FloatTensor(y_seq[:train_idx])
        
        test_X = torch.FloatTensor(X_seq[train_idx:val_idx])
        test_y = torch.FloatTensor(y_seq[train_idx:val_idx])
        
        val_X = torch.FloatTensor(X_seq[val_idx:])
        val_y = torch.FloatTensor(y_seq[val_idx:])
        
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        X_seq = torch.FloatTensor(X_seq)
        y_seq = torch.FloatTensor(y_seq)
        
        return train_X, test_X, val_X, train_y, test_y, val_y, X_seq, y_seq, X, y


    def rescale_predictions(self, scaled_mean: np.ndarray, scaled_variance: np.ndarray) -> ScalingResult:
        """
        Rescale the mean and variance predictions from the network back to the original scale.
        
        Parameters:
        -----------
        scaled_mean: array-like
            The mean predictions from the network (scaled between 0 and 1)
        scaled_variance: array-like
            The variance/covariance predictions from the network
            
        Returns:
        --------
        ScalingResult
            Named tuple containing rescaled mean and variance
        """
        # Convert tensors to numpy if needed
        if isinstance(scaled_mean, torch.Tensor):
            scaled_mean = scaled_mean.detach().numpy()
        if isinstance(scaled_variance, torch.Tensor):
            scaled_variance = scaled_variance.detach().numpy()
        
        # Get the scale factors from the target scaler
        scale_factor = self.target_scaler.data_max_ - self.target_scaler.data_min_
        
        # Rescale the mean predictions
        rescaled_mean = self.target_scaler.inverse_transform(scaled_mean)
        
        # Create a copy of the covariance matrix to avoid modifying the original
        rescaled_variance = scaled_variance.copy()
        
        # Assuming scaled_variance shape is (batch_size, n_features, n_features)
        batch_size = scaled_variance.shape[0]
        n_features = scaled_variance.shape[1]
        
        # Create a scaling matrix for each sample in the batch
        for b in range(batch_size):
            # Create a diagonal matrix of scale factors
            scaling_matrix = np.diag(scale_factor)
            
            # Apply the scaling: S * Σ * S^T where S is the diagonal scaling matrix
            # This properly scales both variances and covariances
            rescaled_variance[b] = scaling_matrix @ scaled_variance[b] @ scaling_matrix.T
        
        return ScalingResult(rescaled_mean, rescaled_variance)
    
    
    def reconstruct_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Reconstruct the time series by averaging overlapping sequences.

        Parameters:
        -----------
        sequence: np.ndarray
            The sequence to reconstruct (shape: [num_sequences, time_horizon, num_features])
        train_data: bool
            Whether the sequence is training data or not
        Returns:
        --------
        np.ndarray
            The reconstructed time series (shape: [n_time_steps, num_features])
        """
        num_sequences, time_horizon, num_features = sequence.shape
        
        n_time_steps = self.targets.shape[0] // self.num_simulations - self.config.time_step - 1 # steps to predict - miss the last value because maths is pastied
        # if train_data:
        #     n_time_steps = int(num_sequences + self.config.train_test_split*(time_horizon + self.config.time_step))
        # else:
        #     n_time_steps = int(num_sequences + (1-self.config.train_test_split) * (time_horizon + self.config.time_step))
        # # Accumulators for sum and counts
        reconstructed = np.zeros((n_time_steps, num_features))
        count = np.zeros((n_time_steps, 1))

        for i in range(num_sequences):
            for h in range(time_horizon):
                t_index = i + h  # The actual time index in the full sequence
                reconstructed[t_index] += sequence[i, h]
                count[t_index] += 1

        # Avoid division by zero
        count[count == 0] = 1  

        return reconstructed / count
    
    def reconstruct_covariance(self, covariances: np.ndarray) -> np.ndarray:
        """
        Reconstruct the covariance matrices by averaging overlapping sequences.

        Parameters:
        -----------
        covariances: np.ndarray
            The covariance matrices to reconstruct 
            (shape: [num_sequences, time_horizon * num_features, time_horizon * num_features])
        
        Returns:
        --------
        np.ndarray
            The reconstructed covariance matrices (shape: [n_time_steps, num_features, num_features])
        """
        num_sequences = covariances.shape[0]
        time_horizon = self.config.horizon
        num_features = covariances.shape[1] // time_horizon
        
        n_time_steps = self.targets.shape[0] // self.num_simulations - self.config.time_step - 1
        reconstructed = np.zeros((n_time_steps, num_features, num_features))
        count = np.zeros((n_time_steps, 1, 1))
        
        for i in range(num_sequences):
            for h in range(time_horizon):
                t_index = i + h
                # Extract the block corresponding to time step h
                # Get the covariance block for the current time step (h)
                start_idx = h * num_features
                end_idx = (h + 1) * num_features
                cov_block = covariances[i, start_idx:end_idx, start_idx:end_idx]
                
                reconstructed[t_index] += cov_block
                count[t_index] += 1
        
        # Avoid division by zero
        count[count == 0] = 1
        
        return reconstructed / count
    
    # def surrogate_sequence(self, sequence: np.ndarray, train_data: bool) -> np.ndarray:
    
    # Deprecated
    def revert_sequences(self, train_mean: Union[np.ndarray, torch.Tensor], 
                            test_mean: Optional[Union[np.ndarray, torch.Tensor]] = None,
                            train_var: Optional[Union[np.ndarray, torch.Tensor]] = None,
                            test_var: Optional[Union[np.ndarray, torch.Tensor]] = None,
                            ) -> ScalingResult:
        """
        Process the model's output (mean and variance predictions) by rescaling them.
        
        Parameters:
        -----------
        train_mean: np.ndarray or torch.Tensor - the training mean output of the model
        train_var: Optional: np.ndarray or torch.Tensor - the test variance output of the model
        test_mean: Optional: np.ndarray or torch.Tensor - the test mean output of the model
        test_var: Optional: np.ndarray or torch.Tensor - the test variance output of the model
            
        Returns:
        --------
        ScalingResult
            Named tuple containing rescaled mean and variance
        """
        
        # Convert to numpy for scaling
        if isinstance(train_mean, torch.Tensor):
            train_mean = train_mean.detach().numpy()
        if isinstance(train_var, torch.Tensor):
            train_var = train_var.detach().numpy()
        if isinstance(test_mean, torch.Tensor):
            test_mean = test_mean.detach().numpy()
        if isinstance(test_var, torch.Tensor):
            test_var = test_var.detach().numpy()
        
        # Reconstruct the sequences into (time steps, observed preds, features)
        # The prediction only begins after the first sequence, so clip the last values
        train_mean = self.reconstruct_sequence(train_mean, True)
        if test_mean is not None:
            test_mean = self.reconstruct_sequence(test_mean, False)
# if its pastied remove last 10 again
        # Do it for the variance too
        if train_var is not None:
            train_var = self.reconstruct_sequence(train_var, True)

        if test_var is not None:
            test_var = self.reconstruct_sequence(test_var, False)


        if test_var is None:
            return self.target_scaler.inverse_transform(train_mean)
        
        means = np.concatenate([train_mean, test_mean], axis=0)
        if train_var is not None and test_var is not None:
            vars = np.concatenate([train_var, test_var], axis=0)
            return self.rescale_predictions(means, vars)
        else:
            return self.rescale_means(means)
        
        # if train_var is not None:
        #     return self.rescale_predictions(train_mean, train_var), self.rescale_predictions(test_mean, test_var)
        # else:
        #     return self.rescale_means(train_mean), self.rescale_means(test_mean)
        
    def quantile_invert(self, preds, quantiles):
        """
        Inverse transform the predictions from quantile space to original space.

        Args:
            preds (np.ndarray): Predictions of shape (batch_size, features * quantiles)

        Returns:
            np.ndarray: Predictions in original space of shape (batch_size, features)
        """

        # Initialize list to store inverse transformed predictions
        quantile_preds = {}
        # Inverse transform each quantile prediction
        for i, q in enumerate(quantiles):
            pred_q = preds[:, :, :, i]
            # This is a prediction of shape (no_sequences, horizon, features)
            # Reconstruct the sequences
            pred_q = self.revert_sequences(pred_q)
            quantile_preds[q] = pred_q # Shape: {quantile: (time_steps, features)}
        
        # Stack the predictions along the last axis
        return quantile_preds
 
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

    def train(self, train_loader: DataLoader, test_loader: DataLoader, val_loader: DataLoader,
            criterion: nn.Module) -> Dict[str, List[float]]:
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate, weight_decay = self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.config.factor, patience=self.config.patience, verbose=True)
        early_stopping = EarlyStopping(self.config)
        history = {'train_loss': [], 'test_loss': [], 'val_loss': [], 'avg_loss': []}

        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = self._NLL_train_epoch(train_loader, criterion, optimizer)

            
            # Validation

            test_loss = self._NLL_validate_epoch(test_loader, criterion)
            val_loss = self._NLL_validate_epoch(val_loader, criterion)
            
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

    def train_sim(self, X_train, y_train, X_test, y_test, X_val, y_val, criterion):
        
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate, weight_decay = self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.config.factor, patience=self.config.patience, verbose=True)
        early_stopping = EarlyStopping(self.config)
        history = {'train_loss': [], 'test_loss': [], 'val_loss': [], 'avg_loss': []}
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            
            for i in range(X_train.shape[0]):
                print('Training on device:', self.device)
                X_sim = X_train[i].to(self.device)
                y_sim = y_train[i].to(self.device)
                train_dataset = TensorDataset(X_sim, y_sim)
                train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
                
                train_loss = self._NLL_train_epoch(train_loader, criterion, optimizer)
                
            # Validation
            self.model.eval()
            for i in range(X_test.shape[0]):
                X_test_sim = X_test[i].to(self.device)
                y_test_sim = y_test[i].to(self.device)
                
                test_dataset = TensorDataset(X_test_sim, y_test_sim)
                test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
                
                X_val_sim = X_val[i].to(self.device)
                y_val_sim = y_val[i].to(self.device)
                
                val_dataset = TensorDataset(X_val_sim, y_val_sim)
                val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
                
                test_loss = self._NLL_validate_epoch(test_loader, criterion)
                val_loss = self._NLL_validate_epoch(val_loader, criterion)

            
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
            predictions = self.model(batch_X)
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
                predictions = self.model(batch_X)
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

class Projector(nn.Module):

    def __init__(self, config: TrainingConfig, A: torch.Tensor, B: torch.Tensor, b: torch.Tensor):
        super(Projector, self).__init__()
        self.config = config
        self.A = A
        self.B = B
        self.b = b
        
        self.B_inv = torch.linalg.pinv(B)
        self.proj_space = torch.eye(self.B.shape[1], self.B.shape[1]) - self.B_inv @ self.B  # Projection onto the null space of B

        # projection Layerz
        self.fc_adj1 = nn.Linear(self.A.shape[1], self.A.shape[1], bias = False)
        self.fc_adj1.weight = nn.Parameter(self.proj_space, requires_grad=False)
        self.fc_adj2 = nn.Linear(self.A.shape[1], self.A.shape[1], bias = False)
        self.fc_adj2.weight = nn.Parameter(-self.B_inv @ self.A, requires_grad=False)
        self.fc_adj2.bias = nn.Parameter(self.B_inv @ self.b)
        
        self.fc_cholesky_adj = nn.Linear(self.A.shape[1]*self.A.shape[1], self.A.shape[1]*self.A.shape[1], bias = False)
        self.fc_cholesky_adj.weight = nn.Parameter(self.proj_space, requires_grad=False)
        self.fc_cholesky_adj2 = nn.Linear(self.A.shape[1]*self.A.shape[1], self.A.shape[1]*self.A.shape[1], bias = False)
        self.fc_cholesky_adj2.weight = nn.Parameter(self.proj_space.transpose(-1,-2), requires_grad=False)

    def forward(self, inputs: torch.Tensor, means: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Project the mean onto the feasible set, and reduce the covariance
        """
        time_steps = inputs.shape[0]
        means = means.reshape(-1, self.config.horizon * means.shape[1])
        cov = cov.reshape(-1, means.shape[1], means.shape[1])
        inputs = inputs.reshape(-1, inputs.shape[1] * inputs.shape[2])
        
        cov_proj = self.fc_cholesky_adj(cov)  # Apply P
        cov_proj = cov_proj @ self.fc_cholesky_adj2(cov_proj)  # Apply P^T (P is symmetric)
              
        
        # cov_proj = self.fc_cholesky_adj(cov) + self.fc_cholesky_adj2(cov)  # Shape: [batch, horizon*n, horizon*n]
        self.epsilon = 1e-6
        # Regularize to ensure non-singularity
        identity = torch.eye(cov_proj.size(-1), device=cov_proj.device).unsqueeze(0).expand(time_steps, -1, -1)
        cov_proj = cov_proj + self.epsilon * identity
        
        # Recompute Cholesky decomposition after regularization
        # L_proj = torch.linalg.cholesky(cov_proj_reg)  # Differentiable Cholesky
        
        # Reshape to [batch, horizon, output_dim, output_dim]
        cov_proj = cov_proj.view(time_steps, means.shape[1], means.shape[1])
        
        means_adj = self.fc_adj1(means) + self.fc_adj2(inputs)
        # Reshape outputs to the expected format
        means_adj = means_adj.view(time_steps, self.config.horizon, means.shape[1] // self.config.horizon)
        
        return means_adj, cov_proj
        

def main():
    
    training_config = TrainingConfig(
        batch_size=85,
        num_epochs=1000,
        learning_rate=0.0031,
        time_step=10,
        horizon=5,
        weight_decay=0.01,
        factor=0.1,
        patience=58,
        delta = 0.042,
        train_test_split=0.6,
        test_val_split=0.8,
        # device = "mps" if torch.backends.mps.is_available() else "cpu",
        device = "cpu"
    )
    
    MLP_Config = MLPConfig(
        hidden_dim = 1024,
        num_layers = 6,
        dropout = 0.2,
        activation = 'ReLU',
        # device = "mps" if torch.backends.mps.is_available() else "cpu",
        device = "cpu"
    )
    
    LSTM_Config = LSTMConfig(
        num_layers = 2,
        hidden_dim = 512,
        dropout = 0.2,
        bidirectional = False,
        norm_type = 'batch',
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    )

    
    features_path = 'small_cstr_features.csv'
    targets_path = 'small_cstr_targets.csv'
    noiseless_path = 'small_cstr_noiseless_results.csv'
    
    features = pd.read_csv(features_path)
    features = features.iloc[:, :-1]
    targets = pd.read_csv(targets_path)
    noiseless_results = pd.read_csv(noiseless_path)
    noiseless_results = noiseless_results.iloc[:, :-1]
    
    data_processor = DataProcessor(training_config, features, targets, num_simulations = 10)
    # Prepare data
    (X_train, X_test, X_val, y_train, y_test, y_val, X_Seq, y_seq, X, y) = data_processor.prepare_data_by_simulation(simulation_length=99)
    
    no_constraints = training_config.horizon
    A = torch.zeros((no_constraints, X_train.shape[3] * training_config.time_step))
    B = torch.zeros((no_constraints, y_train.shape[3] * training_config.horizon))
    b = torch.zeros((no_constraints))
    



    # Example usage
    ipt_features = 5  # Example number of features
    opt_features = 9
    time_horizon = 5  # Example number of time steps
    ipt_idx = 4  # For Volume at time t (0-based)
    opt_idx = 6 # For Fin - 0 based
    num_constraints = time_horizon - 1  # Example number of constraints
    for i in range(num_constraints):
        B[i, ipt_idx + i * ipt_features] = -1    # feat5 at t_i
        B[i, ipt_idx + (i + 1) * ipt_features] = 1  # feat5 at t_(i+1)
        A[i, opt_idx + i * opt_features] = -1  # feat8 at t_i
    
    # model = mpi_nn(
    #     config = MLP_Config,
    #     input_dim=X_train.shape[3]*X_train.shape[2],
    #     output_dim=y_train.shape[3],
    #     horizon = training_config.horizon,
    #     A = A,
    #     B = B,
    #     b = b
    # )
    
    model = MLP(
        config = MLP_Config,
        input_dim=X_train.shape[3]*X_train.shape[2],
        output_dim=y_train.shape[3], 
        horizon = training_config.horizon,
    )
    
    from mv_gaussian_nll import GaussianMVNLL
    criterion = GaussianMVNLL()

    
    trainer = ModelTrainer(model, training_config)
    model, history, avg_loss = trainer.train_sim(X_train, y_train, X_test, y_test, X_val, y_val, criterion)
    
    simulation_preds = {}
    simulation_preds = {i: None for i in range(X.shape[0])}
    simulation_vars = {}
    simulation_vars = {i: None for i in range(X.shape[0])}
    
    for i in range (X.shape[0]):
        with torch.no_grad():
            preds, vars = model(X_Seq[i, :, :].to(training_config.device))
            projector = Projector(training_config, A, B, b)
            preds, vars = projector(X_Seq[i, :, :].to(training_config.device), preds, vars)
            means = preds.detach().numpy()
            vars = vars.detach().numpy()
            means = data_processor.reconstruct_sequence(means)
            vars = data_processor.reconstruct_covariance(vars)
            rescaled_pred = data_processor.rescale_predictions(means, vars)
            simulation_preds[i] = rescaled_pred[0]
            simulation_vars[i] = rescaled_pred[1]
            
            
    feature_names = ['ca', 'cb', 'cc', 'temp', 'vol']
    action_names = ['inlet temp', 'feed conc', 'coolant temp']
    visualizer = Visualizer()
    sequence_length = training_config.time_step
    simulation_time = 99
    num_simulations = 10
    features = np.array(features)
    noiseless_results = np.array(noiseless_results)
    nl_results = np.zeros((num_simulations, simulation_time, features.shape[1]))
    fts = np.zeros((num_simulations, simulation_time, features.shape[1]))
    for i in range(num_simulations):
        start_idx = i * simulation_time
        end_idx = start_idx + simulation_time
        nl_results[i] = noiseless_results[start_idx:end_idx, :]
        fts[i] = features[start_idx:end_idx, :]
    # For simplicity we will plot the first simulation
    for key in simulation_vars:
        simulation_vars[key] = np.diagonal(simulation_vars[key], axis1=1, axis2=2)
        
    visualizer.plot_NLL(simulation_preds, simulation_vars, fts,
                                nl_results,
                                sequence_length,
                                feature_names,
                                num_simulations = 10,
                                train_test_split = 0.6, test_val_split = 0.8)
    
    visualizer.plot_loss(history)
    
if __name__ == "__main__":
    main()