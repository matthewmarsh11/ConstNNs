from utils_new import *
from base import *
import pandas as pd
import numpy as np
import torch


np.random.seed(42)
torch.manual_seed(42)

def load_batch_data():
    # --- Training Config ---
    training_config = TrainingConfig(
        batch_size=300,
        num_epochs=5000,
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
        hidden_dim=1024,
        num_layers=3,
        dropout=0.2,
        activation='ReLU',
        device=training_config.device
    )
    

    # --- Load Data ---
    batch_data = pd.read_csv('datasets/small_batch_features.csv').to_numpy()
    noiseless_batch = pd.read_csv('datasets/small_batch_noiseless_results.csv').to_numpy()
    
    batch_data = pd.read_csv('datasets/small_batch_features.csv').to_numpy()

# Identify key variables
# Assuming batch_data[:, :4] = [Ca, Cb, Cc, T]
# batch_data[:, 4] = Tc
# batch_data[:, -1] = batch_id
# batch_data.shape = (N_total_steps, num_features)

    num_simulations = int(max(batch_data[:, -1]))
    simulation_length = batch_data.shape[0] // num_simulations

    X_list = []
    y_list = []
    noiseless_X_list = []
    noiseless_y_list = []

    for i in range(num_simulations):
        start_idx = i * simulation_length
        end_idx = (i + 1) * simulation_length

        sim_data = batch_data[start_idx:end_idx]
        noiseless_sim_data = noiseless_batch[start_idx:end_idx]
        for t in range(1, simulation_length):
            prev_state = sim_data[t - 1, :4]  # [Ca_{t-1}, Cb_{t-1}, Cc_{t-1}, T_{t-1}]
            noiseless_prev_state = noiseless_sim_data[t - 1, :4]  # [Ca_{t-1}, Cb_{t-1}, Cc_{t-1}, T_{t-1}]
            Tc = sim_data[t, 4]               # Tc at time t
            noiseless_Tc = noiseless_sim_data[t, 4]  # Tc at time t in noiseless data
            elapsed_time = t                 # assuming 1-step intervals

            input_vec = np.concatenate([prev_state, [Tc, elapsed_time]])
            noiseless_input_vec = np.concatenate([noiseless_prev_state, [noiseless_Tc, elapsed_time]])
            target_vec = sim_data[t, :4]     # [Ca_t, Cb_t, Cc_t, T_t]
            noiseless_target_vec = noiseless_sim_data[t, :4]  # [Ca_t, Cb_t, Cc_t, T_t]

            X_list.append(input_vec)
            y_list.append(target_vec)
            
            noiseless_X_list.append(noiseless_input_vec)
            noiseless_y_list.append(noiseless_target_vec)

    X = np.array(X_list)
    y = np.array(y_list)
    noiseless_X = np.array(noiseless_X_list)
    noiseless_y = np.array(noiseless_y_list)
    # inputs to model: [Ca_{t-1}, Cb_{t-1}, Cc_{t-1}, T_{t-1}, Tc, elapsed_batch_time]
    # outputs from model: [Ca_t, Cb_t, Cc_t, T_t]
    # constraint: 2* Ca_t + Cb_t + Cc_t = 2_ Ca_{t-1} + Cb_{t-1} + Cc_{t-1}
    # y = batch_data[:, :4]  # Outputs: [Ca, Cb, Cc, T]
    # x = batch_data[:, 4:5]  # Input: [Tc]


    # --- Data Processors ---
    data_processor = DataProcessor(training_config, X, y, num_simulations)
    

    (X_train, X_test, X_val, y_train, y_test, y_val, X_tensor, y_tensor) = data_processor.prepare_data(simulation_length=simulation_length)
    

    # --- Define Constraint ---
    A = torch.tensor([-2.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=training_config.device)
    B = torch.tensor([2.0, 1.0, 1.0, 0.0], dtype=torch.float32, device=training_config.device)
    b = torch.tensor([0.0], dtype=torch.float32, device=training_config.device)

    scaled_A, scaled_B, scaled_b = data_processor.scale_constraints(A, B, b)

    return {
        "training_config": training_config,
        "model_config": model_config,
        "data_processor": data_processor,
        "X_train": X_train,
        "X_test": X_test,
        "X_val": X_val,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val,
        # "X_train_unconst": X_train_unconst,
        # "X_test_unconst": X_test_unconst,
        # "X_val_unconst": X_val_unconst,
        "X_tensor": X_tensor,
        "y_tensor": y_tensor,
        # "X_tensor_unconst": X_tensor_unconst,
        "scaled_A": scaled_A,
        "scaled_B": scaled_B,
        "scaled_b": scaled_b,
        "batch_data": batch_data,
        "noiseless_batch": noiseless_batch,
        "num_simulations": num_simulations,
        "simulation_length": simulation_length
    }