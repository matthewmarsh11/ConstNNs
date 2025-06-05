from utils_new import *
from models.Gaussian_HPINN import KKT_PPINN
from models.mlp import MLP
from base import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch

def main():
    data = pd.read_csv('datasets/benchmark_CSTR.csv')
    x = data[['T x1','Ff_B x2','Ff_E x3']]
    y = data[['F_EB z1','F_B z2','F_E z3']]
    
    x = x.to_numpy()
    y = y.to_numpy()
    
    # Add Gaussian noise scaled by original values
    noise_level = 0.01  # Adjust noise level as needed (percentage of original value)
    
    # Add scaled Gaussian noise to output targets
    y_noise = np.random.normal(0, 1, y.shape) * y * noise_level
    y_noisy = y + y_noise
    
    x = torch.tensor(x, dtype=torch.float32)
    y_noisy = torch.tensor(y_noisy, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    training_config = TrainingConfig(
        batch_size=21,
        num_epochs=100,
        learning_rate=0.0004686825880910967,
        weight_decay=0.003761781637604189,
        factor=0.369045567244328,
        patience=0.006806821916779027,
        delta = 0.0001,
        train_test_split=0.6,
        test_val_split=0.8,
        # device = "mps" if torch.backends.mps.is_available() else "cpu",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )
    
    MLP_Config = MLPConfig(
        hidden_dim = 128,
        num_layers = 3,
        dropout = 0.2,
        activation = 'ReLU',
        # device = "mps" if torch.backends.mps.is_available() else "cpu",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Split the data
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y_noisy, test_size=1-training_config.train_test_split, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=training_config.test_val_split, random_state=42
    )
    
    # Scale the data using MinMaxScaler
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_val_scaled = x_scaler.transform(x_val)
    x_test_scaled = x_scaler.transform(x_test)
    
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)
    
    # Convert to tensors
    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
    
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    # Constraint: x2 - x3 -y2 + y3 = 0
    # Define original constraint matrices
    A = torch.tensor([[0, 1, -1]], dtype=torch.float32)  # Coefficients for x2 - x3
    B = torch.tensor([[0, -1, 1]], dtype=torch.float32)  # Coefficients for y2 - y3
    b = torch.tensor([[0]], dtype=torch.float32)  # Right-hand side of the constraint
    
    constr = torch.zeros((x_train_tensor.shape[0], 1), dtype=torch.float32)
    nn_constr = torch.zeros((x_train_tensor.shape[0], 1), dtype=torch.float32)
    for i in range(x_train_tensor.shape[0]):
        constr[i] = torch.matmul(A, x[i]) + torch.matmul(B, y_noisy[i]) - b
        nn_constr[i] = torch.matmul(A, x[i]) + torch.matmul(B, y[i]) - b
        
        
    
    # Scale the constraint matrices to work with scaled data
    # Scale A matrix according to input scaling
    x_scale = torch.tensor(x_scaler.scale_, dtype=torch.float32)
    A_constr_model = A / x_scale.unsqueeze(0)
    
    # Scale B matrix according to output scaling
    y_scale = torch.tensor(y_scaler.scale_, dtype=torch.float32)
    B_constr_model = B / y_scale.unsqueeze(0)
    
    # Scale b vector - for linear constraints, b typically doesn't need scaling
    # since the constraint Ax + By = b becomes (A/sx)x_scaled + (B/sy)y_scaled = b
    b_constr_model = b
    
    scaled_constr = torch.zeros((x_train_tensor.shape[0], 1), dtype=torch.float32)
    for i in range(x_train_tensor.shape[0]):
        scaled_constr[i] = torch.matmul(A_constr_model, x_train_tensor[i]) + torch.matmul(B_constr_model, y_train_tensor[i]) - b_constr_model
    
    epsilon = float(2)
    
    model = KKT_PPINN(
        config = MLP_Config,
        input_dim = x_train.shape[1],
        output_dim = y_train.shape[1],
        A = A_constr_model,  # Use scaled A
        B = B_constr_model,  # Use scaled B
        b = b_constr_model,  # Use scaled b
        epsilon = epsilon,
        probability_level = 0.95
    )
    
    np_model = MLP(
        config = MLP_Config,
        input_dim = x_train.shape[1],
        output_dim = y_train.shape[1],
        num_samples = None
    )
    
    trainer = ModelTrainer(model, training_config)
    trainer_np = ModelTrainer(np_model, training_config)
    from mv_gaussian_nll import GaussianMVNLL
    criterion = GaussianMVNLL()
    model, history, avg_loss = trainer.train(
        x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, x_val_tensor, y_val_tensor, criterion
    )
    np_model, np_history, np_avg_loss = trainer_np.train(
        x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, x_val_tensor, y_val_tensor, criterion
    )
    
    test_preds, test_covs = model(x_test_tensor)
    np_test_preds, np_test_covs = np_model(x_test_tensor)
    # test_preds = y_scaler.inverse_transform(test_preds.detach().numpy())
    # np_test_preds = y_scaler.inverse_transform(np_test_preds.detach().numpy())
    
    # test_covs = torch.matmul(test_covs, test_covs.transpose(1, 2))
    # np_test_covs = torch.matmul(np_test_covs, np_test_covs.transpose(1, 2))
    # scale_factor = y_scaler.data_max_ - y_scaler.data_min_
    # stds = np.sqrt(np.diagonal(test_covs.detach().numpy(), axis1=1, axis2=2)) * scale_factor**2
    # np_stds = np.sqrt(np.diagonal(np_test_covs.detach().numpy(), axis1=1, axis2=2)) * scale_factor**2
    
    import matplotlib.pyplot as plt

    # Convert test predictions and true values back to original scale
    test_preds_unscaled = y_scaler.inverse_transform(test_preds.detach().numpy())
    np_test_preds_unscaled = y_scaler.inverse_transform(np_test_preds.detach().numpy())
    y_test_unscaled = y_scaler.inverse_transform(y_test_tensor.detach().numpy())

    # Calculate covariance matrices and standard deviations
    test_covs_full = torch.matmul(test_covs, test_covs.transpose(1, 2))
    np_test_covs_full = torch.matmul(np_test_covs, np_test_covs.transpose(1, 2))

    # Scale covariances back to original scale
    scale_factor = torch.tensor(y_scaler.scale_, dtype=torch.float32)
    scale_matrix = torch.outer(scale_factor, scale_factor)
    test_covs_unscaled = test_covs_full / scale_matrix.unsqueeze(0)
    np_test_covs_unscaled = np_test_covs_full / scale_matrix.unsqueeze(0)

    # Extract standard deviations
    stds = torch.sqrt(torch.diagonal(test_covs_unscaled, dim1=1, dim2=2)).detach().numpy()
    np_stds = torch.sqrt(torch.diagonal(np_test_covs_unscaled, dim1=1, dim2=2)).detach().numpy()

    # Create parity plots with both models on same plot
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    output_names = ['F_EB z1', 'F_B z2', 'F_E z3']

    for i in range(3):
        # Sort data for smooth fill_between
        constrained_sorted = np.argsort(y_test_unscaled[:, i])
        unconstrained_sorted = np.argsort(y_test_unscaled[:, i])
        
        # Constrained model
        x_sorted = y_test_unscaled[constrained_sorted, i]
        y_sorted = test_preds_unscaled[constrained_sorted, i]
        std_sorted = stds[constrained_sorted, i]
        
        axes[i].plot(x_sorted, y_sorted, 'o', color='blue', alpha=0.6, label='Constrained', markersize=4)
        axes[i].fill_between(x_sorted, y_sorted - std_sorted, y_sorted + std_sorted, 
                            color='blue', alpha=0.2)
        
        # Unconstrained model
        x_sorted_np = y_test_unscaled[unconstrained_sorted, i]
        y_sorted_np = np_test_preds_unscaled[unconstrained_sorted, i]
        std_sorted_np = np_stds[unconstrained_sorted, i]
        
        axes[i].plot(x_sorted_np, y_sorted_np, 'o', color='red', alpha=0.6, label='Unconstrained', markersize=4)
        axes[i].fill_between(x_sorted_np, y_sorted_np - std_sorted_np, y_sorted_np + std_sorted_np, 
                            color='red', alpha=0.2)
        
        # Perfect prediction line
        axes[i].plot([y_test_unscaled[:, i].min(), y_test_unscaled[:, i].max()], 
                    [y_test_unscaled[:, i].min(), y_test_unscaled[:, i].max()], 
                    'k--', lw=2, label='Perfect prediction')
        
        axes[i].set_xlabel(f'True {output_names[i]}')
        axes[i].set_ylabel(f'Predicted {output_names[i]}')
        axes[i].set_title(f'{output_names[i]}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate constraint violations
    x_test_unscaled = x_scaler.inverse_transform(x_test_tensor.detach().numpy())
    
    # Original constraint matrices (unscaled)
    A_orig = torch.tensor([[0, 1, -1]], dtype=torch.float32)
    B_orig = torch.tensor([[0, -1, 1]], dtype=torch.float32)
    b_orig = torch.tensor([[0]], dtype=torch.float32)
    
    # Calculate constraint violations for both models
    constrained_violations = []
    unconstrained_violations = []
    
    for i in range(len(x_test_unscaled)):
        # Constrained model violation
        constr_viol = torch.matmul(A_orig, torch.tensor(x_test_unscaled[i], dtype=torch.float32)) + \
                     torch.matmul(B_orig, torch.tensor(test_preds_unscaled[i], dtype=torch.float32)) - b_orig
        constrained_violations.append(constr_viol.item())
        
        # Unconstrained model violation
        unconstr_viol = torch.matmul(A_orig, torch.tensor(x_test_unscaled[i], dtype=torch.float32)) + \
                       torch.matmul(B_orig, torch.tensor(np_test_preds_unscaled[i], dtype=torch.float32)) - b_orig
        unconstrained_violations.append(unconstr_viol.item())

    # Plot constraint violations
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(constrained_violations, bins=20, alpha=0.7, color='blue', label='Constrained Model')
    plt.hist(unconstrained_violations, bins=20, alpha=0.7, color='red', label='Unconstrained Model')
    plt.xlabel('Constraint Violation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Constraint Violations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sample_indices = range(len(constrained_violations))
    plt.plot(sample_indices, constrained_violations, 'o-', color='blue', alpha=0.7, 
             label='Constrained Model', markersize=3)
    plt.plot(sample_indices, unconstrained_violations, 'o-', color='red', alpha=0.7, 
             label='Unconstrained Model', markersize=3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Constraint Violation')
    plt.title('Constraint Violations by Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Constrained Model - Mean violation: {np.mean(constrained_violations):.6f}, "
          f"Std: {np.std(constrained_violations):.6f}, "
          f"Max abs: {np.max(np.abs(constrained_violations)):.6f}")
    print(f"Unconstrained Model - Mean violation: {np.mean(unconstrained_violations):.6f}, "
          f"Std: {np.std(unconstrained_violations):.6f}, "
          f"Max abs: {np.max(np.abs(unconstrained_violations)):.6f}")
    
if __name__ == "__main__":
    main()