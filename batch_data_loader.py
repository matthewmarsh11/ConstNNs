from utils_new import *
from base import *
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(42)

def load_batch_data():
    # --- Training Config ---
    training_config = TrainingConfig(
        batch_size=512,
        num_epochs=500,
        learning_rate=0.001,
        weight_decay=0.0001,
        factor=0.1,
        patience=10,
        delta=0.0001,
        train_test_split=0.6,
        test_val_split=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    model_config = MLPConfig(
        hidden_dim=62,
        num_layers=1,
        dropout=0.2,
        activation='ReLU',
        device=training_config.device
    )
    

    # --- Load Data ---
    batch_data = pd.read_csv('datasets/small_batch_features.csv').to_numpy()
    noiseless_batch = pd.read_csv('datasets/small_batch_noiseless_results.csv').to_numpy()
    y = batch_data[:, :4]  # Outputs: [Ca, Cb, Cc, T]
    x = batch_data[:, 4:5]  # Input: [Tc]

    num_simulations = int(max(batch_data[:, -1]))
    simulation_length = y.shape[0] // num_simulations
    
    # The model should use the inlet conditions and the elapsed batch time as an input too:
    cumulative_time = np.tile(np.arange(simulation_length), num_simulations).reshape(-1, 1)

    # Get initial conditions from noiseless_batch
    initial_conditions = np.vstack([
        np.tile(batch_data[i * simulation_length, :-3], (simulation_length, 1))
        for i in range(num_simulations)
    ])

    # Combine features: Tc, cumulative_time, initial_conditions
    x_init = batch_data[:, 4:5]  # Tc
    x = np.hstack((x_init, cumulative_time, initial_conditions))


    # --- Add Constant Constraint Input ---
    constants_per_point = np.zeros_like(x_init)
    for i in range(num_simulations):
        start_idx = i * simulation_length
        end_idx = (i + 1) * simulation_length
        const_val = 2 * noiseless_batch[start_idx, 0] + noiseless_batch[start_idx, 1] + noiseless_batch[start_idx, 2]
        constants_per_point[start_idx:end_idx] = const_val

    x_const = np.hstack((x, constants_per_point))

    # --- Data Processors ---
    const_data_processor = DataProcessor(training_config, x_const, y, num_simulations)
    unconst_data_processor = DataProcessor(training_config, x, y, num_simulations)

    (X_train, X_test, X_val, y_train, y_test, y_val, X_tensor, y_tensor) = const_data_processor.prepare_data(simulation_length=simulation_length)
    (X_train_unconst, X_test_unconst, X_val_unconst, _, _, _, X_tensor_unconst, _) = unconst_data_processor.prepare_data(simulation_length=simulation_length)

    # --- Define Constraint ---
    A = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=torch.float32, device=training_config.device)
    B = torch.tensor([2.0, 1.0, 1.0, 0.0], dtype=torch.float32, device=training_config.device)
    b = torch.tensor([0.0], dtype=torch.float32, device=training_config.device)

    scaled_A, scaled_B, scaled_b = const_data_processor.scale_constraints(A, B, b)

    return {
        "training_config": training_config,
        "model_config": model_config,
        "const_data_processor": const_data_processor,
        "unconst_data_processor": unconst_data_processor,
        "X_train": X_train,
        "X_test": X_test,
        "X_val": X_val,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val,
        "X_train_unconst": X_train_unconst,
        "X_test_unconst": X_test_unconst,
        "X_val_unconst": X_val_unconst,
        "X_tensor": X_tensor,
        "y_tensor": y_tensor,
        "X_tensor_unconst": X_tensor_unconst,
        "scaled_A": scaled_A,
        "scaled_B": scaled_B,
        "scaled_b": scaled_b,
        "batch_data": batch_data,
        "noiseless_batch": noiseless_batch,
        "num_simulations": num_simulations,
        "simulation_length": simulation_length
    }