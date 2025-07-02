from utils_new import *
from models.Gaussian_HPINN import KKT_PPINN
from models.mlp import MLP
from models.ec_nn import EC_NN
from base import *
import pandas as pd
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

def load_ss_cstr_data():
    # define configurations for training and model
    training_config = TrainingConfig(
        batch_size=300,
        num_epochs=150,
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
    
    # Load the dataset
    
    data = pd.read_csv('datasets/benchmark_CSTR.csv')
    
    x = data[['T x1','Ff_B x2','Ff_E x3']]
    y = data[['F_EB z1','F_B z2','F_E z3']]
    
    x = x.to_numpy()
    y = y.to_numpy()
    noiseless_data = np.hstack((x,y))    
    # Add Gaussian noise scaled by original values
    noise_level = 0.01 
    # Add scaled Gaussian noise to output targets
    y_noise = np.random.normal(0, 1, y.shape) * y * noise_level
    y_noisy = y + y_noise
    noisy_data = np.hstack((x, y_noisy))
    data_processor = DataProcessor(training_config, x, y_noisy)
    X_train, X_test, X_val, y_train, y_test, y_val, X_tensor, y_tensor = data_processor.prepare_ss_data()
       
    # Constraint: x2 - x3 -y2 + y3 = 0
    # Define original constraint matrices
    # A = torch.tensor([[0, 1, -1],
    #                   [0, 1, 0]], dtype=torch.float32)  # Coefficients for x2 - x3
    # B = torch.tensor([[0, -1, 1],
    #                   [-1, -1 ,0]], dtype=torch.float32)  # Coefficients for -y2 + y3
    # b = torch.tensor([[0], 
    #                   [0]], dtype=torch.float32)  # Right-hand side of the constraint
    
    A = torch.tensor([[0, 1, -1]], dtype=torch.float32)  # Coefficients for x2 - x3
    B = torch.tensor([[0, -1, 1]], dtype=torch.float32)  # Coefficients for -y2 + y3
    b = torch.tensor([[0]], dtype=torch.float32)  # Right-hand side of the constraint
    
    scaled_A, scaled_B, scaled_b = data_processor.scale_constraints(A, B, b)
    
    return{
        
        "training_config": training_config,
        "model_config": model_config,
        "data_processor": data_processor,
        "X_train": X_train,
        "X_test": X_test,
        "X_val": X_val,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val,
        "X_tensor": X_tensor,
        "y_tensor": y_tensor,
        "scaled_A": scaled_A,
        "scaled_B": scaled_B,
        "scaled_b": scaled_b,
        "noiseless_data": noiseless_data,
        "noisy_data": noisy_data
    }
    
    constr = torch.zeros((x_train_tensor.shape[0], b.shape[0]), dtype=torch.float32)
    nn_constr = torch.zeros((x_train_tensor.shape[0], b.shape[0]), dtype=torch.float32)
    scaled_constr = torch.zeros((x_train_tensor.shape[0], b.shape[0]), dtype=torch.float32)

    for i in range(x_train_tensor.shape[0]):
        for j in range(b.shape[0]):
            constr[i, j] = torch.matmul(A[j, :], x[i]) + torch.matmul(B[j, :], y_noisy[i]) - b[j]
            nn_constr[i, j] = torch.matmul(A[j, :], x[i]) + torch.matmul(B[j, :], y[i]) - b[j]
            scaled_constr[i, j] = torch.matmul(A_constr_model[j, :], x_train_tensor[i]) + torch.matmul(B_constr_model[j, :], y_train_tensor[i]) - b_constr_model[j]
        

    epsilon = float(0.2875)
    
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
    
    ec_model = EC_NN(
        config = MLP_Config,
        input_dim = x_train.shape[1],
        output_dim = y_train.shape[1],
        A = A_constr_model,  # Use scaled A
        B = B_constr_model,  # Use scaled B
        b = b_constr_model,  # Use scaled b
        dependent_ids=[2],  # Indices of dependent variables
    )
    
    pinn_model = MLP(
        config = MLP_Config,
        input_dim = x_train.shape[1],
        output_dim = y_train.shape[1],
        num_samples = None
    )
    
    trainer = ModelTrainer(model, training_config)
    trainer_np = ModelTrainer(np_model, training_config)
    trainer_ec = ModelTrainer(ec_model, training_config)
    trainer_pinn = ModelTrainer(pinn_model, training_config)
    from mv_gaussian_nll import GaussianMVNLL
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    criterion = GaussianMVNLL()
    model, history, avg_loss = trainer.train(
        x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, x_val_tensor, y_val_tensor, criterion
    )
    np_model, np_history, np_avg_loss = trainer_np.train(
        x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, x_val_tensor, y_val_tensor, criterion
    )
    ec_model, ec_history, ec_avg_loss = trainer_ec.train(
        x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, x_val_tensor, y_val_tensor, criterion
    )
    lamb = 0.1
    pinn_model, pinn_history, pinn_avg_loss = trainer_pinn.train(
        x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, x_val_tensor, y_val_tensor, criterion,
        A_constr_model, B_constr_model, b_constr_model, lamb
    )

    test_preds, test_covs = model(x_test_tensor)
    np_test_preds, np_test_covs = np_model(x_test_tensor)
    ec_test_preds, ec_test_covs = ec_model(x_test_tensor)
    pinn_test_preds, pinn_test_covs = pinn_model(x_test_tensor)
    
    # Get predictions for train and validation sets
    train_preds, train_covs = model(x_train_tensor)
    np_train_preds, np_train_covs = np_model(x_train_tensor)
    ec_train_preds, ec_train_covs = ec_model(x_train_tensor)
    pinn_train_preds, pinn_train_covs = pinn_model(x_train_tensor)
    
    val_preds, val_covs = model(x_val_tensor)
    np_val_preds, np_val_covs = np_model(x_val_tensor)
    ec_val_preds, ec_val_covs = ec_model(x_val_tensor)
    pinn_val_preds, pinn_val_covs = pinn_model(x_val_tensor)
    
    import matplotlib.pyplot as plt

    # Convert predictions and true values back to original scale
    def inverse_transform_and_process(preds, covs, y_true, y_scaler):
        preds_unscaled = y_scaler.inverse_transform(preds.detach().numpy())
        y_true_unscaled = y_scaler.inverse_transform(y_true.detach().numpy())
        
        # Calculate covariance matrices and standard deviations
        covs_full = torch.matmul(covs, covs.transpose(1, 2))
        
        # Scale covariances back to original scale
        scale_factor = torch.tensor(y_scaler.scale_, dtype=torch.float32)
        scale_matrix = torch.outer(scale_factor, scale_factor)
        covs_unscaled = covs_full / scale_matrix.unsqueeze(0)
        
        # Extract standard deviations
        stds = torch.sqrt(torch.diagonal(covs_unscaled, dim1=1, dim2=2)).detach().numpy()
        
        return preds_unscaled, y_true_unscaled, stds

    # Process all datasets
    test_preds_unscaled, y_test_unscaled, test_stds = inverse_transform_and_process(test_preds, test_covs, y_test_tensor, y_scaler)
    np_test_preds_unscaled, _, np_test_stds = inverse_transform_and_process(np_test_preds, np_test_covs, y_test_tensor, y_scaler)
    ec_test_preds_unscaled, _, ec_test_stds = inverse_transform_and_process(ec_test_preds, ec_test_covs, y_test_tensor, y_scaler)
    pinn_test_preds_unscaled, _, pinn_test_stds = inverse_transform_and_process(pinn_test_preds, pinn_test_covs, y_test_tensor, y_scaler)
    
    train_preds_unscaled, y_train_unscaled, train_stds = inverse_transform_and_process(train_preds, train_covs, y_train_tensor, y_scaler)
    np_train_preds_unscaled, _, np_train_stds = inverse_transform_and_process(np_train_preds, np_train_covs, y_train_tensor, y_scaler)
    ec_train_preds_unscaled, _, ec_train_stds = inverse_transform_and_process(ec_train_preds, ec_train_covs, y_train_tensor, y_scaler)
    pinn_train_preds_unscaled, _, pinn_train_stds = inverse_transform_and_process(pinn_train_preds, pinn_train_covs, y_train_tensor, y_scaler)
    
    val_preds_unscaled, y_val_unscaled, val_stds = inverse_transform_and_process(val_preds, val_covs, y_val_tensor, y_scaler)
    np_val_preds_unscaled, _, np_val_stds = inverse_transform_and_process(np_val_preds, np_val_covs, y_val_tensor, y_scaler)
    ec_val_preds_unscaled, _, ec_val_stds = inverse_transform_and_process(ec_val_preds, ec_val_covs, y_val_tensor, y_scaler)
    pinn_val_preds_unscaled, _, pinn_val_stds = inverse_transform_and_process(pinn_val_preds, pinn_val_covs, y_val_tensor, y_scaler)

    # Create parity plots for all datasets and models
    datasets = ['Train', 'Validation', 'Test']
    output_names = ['F_EB z1', 'F_B z2', 'F_E z3']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    for row, (dataset_name, y_true, const_preds, const_stds, np_preds, np_stds, ec_preds, ec_stds, pinn_preds, pinn_stds) in enumerate([
        ('Train', y_train_unscaled, train_preds_unscaled, train_stds, np_train_preds_unscaled, np_train_stds, ec_train_preds_unscaled, ec_train_stds, pinn_train_preds_unscaled, pinn_train_stds),
        ('Validation', y_val_unscaled, val_preds_unscaled, val_stds, np_val_preds_unscaled, np_val_stds, ec_val_preds_unscaled, ec_val_stds, pinn_val_preds_unscaled, pinn_val_stds),
        ('Test', y_test_unscaled, test_preds_unscaled, test_stds, np_test_preds_unscaled, np_test_stds, ec_test_preds_unscaled, ec_test_stds, pinn_test_preds_unscaled, pinn_test_stds)
    ]):
        
        for col in range(3):
            ax = axes[row, col]
            
            # Sort data by true values for smooth lines
            sort_idx = np.argsort(y_true[:, col])
            x_sorted = y_true[sort_idx, col]
            
            # Constrained model
            y_const_sorted = const_preds[sort_idx, col]
            std_const_sorted = const_stds[sort_idx, col]
            ax.plot(x_sorted, y_const_sorted, '-', color='blue', alpha=0.8, label='Constrained', linewidth=2)
            ax.fill_between(x_sorted, y_const_sorted - std_const_sorted, y_const_sorted + std_const_sorted, 
                            color='blue', alpha=0.2)
            
            # Unconstrained model
            y_np_sorted = np_preds[sort_idx, col]
            std_np_sorted = np_stds[sort_idx, col]
            ax.plot(x_sorted, y_np_sorted, '-', color='red', alpha=0.8, label='Unconstrained', linewidth=2)
            ax.fill_between(x_sorted, y_np_sorted - std_np_sorted, y_np_sorted + std_np_sorted, 
                            color='red', alpha=0.2)
            
            # EC model
            y_ec_sorted = ec_preds[sort_idx, col]
            std_ec_sorted = ec_stds[sort_idx, col]
            ax.plot(x_sorted, y_ec_sorted, '-', color='green', alpha=0.8, label='EC-NN', linewidth=2)
            ax.fill_between(x_sorted, y_ec_sorted - std_ec_sorted, y_ec_sorted + std_ec_sorted, 
                            color='green', alpha=0.2)
            
            # PINN model
            y_pinn_sorted = pinn_preds[sort_idx, col]
            std_pinn_sorted = pinn_stds[sort_idx, col]
            ax.plot(x_sorted, y_pinn_sorted, '-', color='orange', alpha=0.8, label='PINN', linewidth=2)
            ax.fill_between(x_sorted, y_pinn_sorted - std_pinn_sorted, y_pinn_sorted + std_pinn_sorted, 
                            color='orange', alpha=0.2)
            
            # Perfect prediction line
            ax.plot([y_true[:, col].min(), y_true[:, col].max()], 
                    [y_true[:, col].min(), y_true[:, col].max()], 
                    'k--', lw=2, label='Perfect prediction')
            
            ax.set_xlabel(f'True {output_names[col]}')
            ax.set_ylabel(f'Predicted {output_names[col]}')
            ax.set_title(f'{dataset_name} - {output_names[col]}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate constraint violations for all models
    x_test_unscaled = x_scaler.inverse_transform(x_test_tensor.detach().numpy())
    
    # Original constraint matrices (unscaled)
    A_orig = torch.tensor([[0, 1, -1]], dtype=torch.float32)
    B_orig = torch.tensor([[0, -1, 1]], dtype=torch.float32)
    b_orig = torch.tensor([[0]], dtype=torch.float32)
    
    # Calculate constraint violations for all models
    constrained_violations = []
    unconstrained_violations = []
    ec_violations = []
    pinn_violations = []
    
    for i in range(len(x_test_unscaled)):
        # Constrained model violation
        constr_viol = torch.matmul(A_orig, torch.tensor(x_test_unscaled[i], dtype=torch.float32)) + \
                        torch.matmul(B_orig, torch.tensor(test_preds_unscaled[i], dtype=torch.float32)) - b_orig
        constrained_violations.append(constr_viol.item())
        
        # Unconstrained model violation
        unconstr_viol = torch.matmul(A_orig, torch.tensor(x_test_unscaled[i], dtype=torch.float32)) + \
                        torch.matmul(B_orig, torch.tensor(np_test_preds_unscaled[i], dtype=torch.float32)) - b_orig
        unconstrained_violations.append(unconstr_viol.item())
        
        # EC model violation
        ec_viol = torch.matmul(A_orig, torch.tensor(x_test_unscaled[i], dtype=torch.float32)) + \
                    torch.matmul(B_orig, torch.tensor(ec_test_preds_unscaled[i], dtype=torch.float32)) - b_orig
        ec_violations.append(ec_viol.item())
        
        # PINN model violation
        pinn_viol = torch.matmul(A_orig, torch.tensor(x_test_unscaled[i], dtype=torch.float32)) + \
                    torch.matmul(B_orig, torch.tensor(pinn_test_preds_unscaled[i], dtype=torch.float32)) - b_orig
        pinn_violations.append(pinn_viol.item())

    # Plot constraint violations
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.hist(constrained_violations, bins=20, alpha=0.7, color='blue', label='Constrained Model')
    plt.hist(unconstrained_violations, bins=20, alpha=0.7, color='red', label='Unconstrained Model')
    plt.hist(ec_violations, bins=20, alpha=0.7, color='green', label='EC-NN Model')
    plt.hist(pinn_violations, bins=20, alpha=0.7, color='orange', label='PINN Model')
    plt.xlabel('Constraint Violation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Constraint Violations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 2)
    sample_indices = range(len(constrained_violations))
    plt.plot(sample_indices, constrained_violations, '-', color='blue', alpha=0.7, 
                label='Constrained Model', linewidth=1)
    plt.plot(sample_indices, unconstrained_violations, '-', color='red', alpha=0.7, 
                label='Unconstrained Model', linewidth=1)
    plt.plot(sample_indices, ec_violations, '-', color='green', alpha=0.7, 
                label='EC-NN Model', linewidth=1)
    plt.plot(sample_indices, pinn_violations, '-', color='orange', alpha=0.7, 
                label='PINN Model', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Constraint Violation')
    plt.title('Constraint Violations by Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 3)
    violations_data = [np.abs(constrained_violations), np.abs(unconstrained_violations), np.abs(ec_violations), np.abs(pinn_violations)]
    plt.boxplot(violations_data, labels=['Constrained', 'Unconstrained', 'EC-NN', 'PINN'])
    plt.ylabel('Absolute Constraint Violation')
    plt.title('Absolute Constraint Violations Comparison')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 4)
    models = ['Constrained', 'Unconstrained', 'EC-NN', 'PINN']
    mean_violations = [np.mean(np.abs(constrained_violations)), 
                      np.mean(np.abs(unconstrained_violations)), 
                      np.mean(np.abs(ec_violations)),
                      np.mean(np.abs(pinn_violations))]
    colors = ['blue', 'red', 'green', 'orange']
    plt.bar(models, mean_violations, color=colors, alpha=0.7)
    plt.ylabel('Mean Absolute Constraint Violation')
    plt.title('Mean Constraint Violation by Model')
    plt.yscale('log')
    plt.xticks(rotation=45)
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
    print(f"EC-NN Model - Mean violation: {np.mean(ec_violations):.6f}, "
            f"Std: {np.std(ec_violations):.6f}, "
              f"Max abs: {np.max(np.abs(ec_violations)):.6f}")
    print(f"PINN Model - Mean violation: {np.mean(pinn_violations):.6f}, "
            f"Std: {np.std(pinn_violations):.6f}, "
              f"Max abs: {np.max(np.abs(pinn_violations)):.6f}")
    
    # Calculate model complexity (number of parameters)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    const_params = count_parameters(model)
    np_params = count_parameters(np_model)
    ec_params = count_parameters(ec_model)
    pinn_params = count_parameters(pinn_model)
    
    print("\nModel Complexity (Number of Parameters):")
    print(f"Constrained Model: {const_params:,}")
    print(f"Unconstrained Model: {np_params:,}")
    print(f"EC-NN Model: {ec_params:,}")
    print(f"PINN Model: {pinn_params:,}")
    
    # Calculate accuracy metrics (MSE and MAE) for all datasets
    
    
    # Get clean (non-noisy) target values for comparison
    # Split clean data the same way as noisy data
    x_train_clean, x_temp_clean, y_train_clean, y_temp_clean = train_test_split(
        x, y, test_size=1-training_config.train_test_split, random_state=42
    )
    x_val_clean, x_test_clean, y_val_clean, y_test_clean = train_test_split(
        x_temp_clean, y_temp_clean, test_size=training_config.test_val_split, random_state=42
    )
    
    # Convert clean targets to numpy for metrics calculation
    y_train_clean_np = y_train_clean.detach().numpy()
    y_val_clean_np = y_val_clean.detach().numpy()
    y_test_clean_np = y_test_clean.detach().numpy()
    
    # Metrics against clean (non-noisy) data - Test set
    const_test_mse_clean = mean_squared_error(y_test_clean_np, test_preds_unscaled)
    const_test_mae_clean = mean_absolute_error(y_test_clean_np, test_preds_unscaled)
    
    np_test_mse_clean = mean_squared_error(y_test_clean_np, np_test_preds_unscaled)
    np_test_mae_clean = mean_absolute_error(y_test_clean_np, np_test_preds_unscaled)
    
    ec_test_mse_clean = mean_squared_error(y_test_clean_np, ec_test_preds_unscaled)
    ec_test_mae_clean = mean_absolute_error(y_test_clean_np, ec_test_preds_unscaled)
    
    pinn_test_mse_clean = mean_squared_error(y_test_clean_np, pinn_test_preds_unscaled)
    pinn_test_mae_clean = mean_absolute_error(y_test_clean_np, pinn_test_preds_unscaled)
    
    # Metrics against clean (non-noisy) data - Train set
    const_train_mse_clean = mean_squared_error(y_train_clean_np, train_preds_unscaled)
    const_train_mae_clean = mean_absolute_error(y_train_clean_np, train_preds_unscaled)
    
    np_train_mse_clean = mean_squared_error(y_train_clean_np, np_train_preds_unscaled)
    np_train_mae_clean = mean_absolute_error(y_train_clean_np, np_train_preds_unscaled)
    
    ec_train_mse_clean = mean_squared_error(y_train_clean_np, ec_train_preds_unscaled)
    ec_train_mae_clean = mean_absolute_error(y_train_clean_np, ec_train_preds_unscaled)
    
    pinn_train_mse_clean = mean_squared_error(y_train_clean_np, pinn_train_preds_unscaled)
    pinn_train_mae_clean = mean_absolute_error(y_train_clean_np, pinn_train_preds_unscaled)
    
    # Metrics against clean (non-noisy) data - Validation set
    const_val_mse_clean = mean_squared_error(y_val_clean_np, val_preds_unscaled)
    const_val_mae_clean = mean_absolute_error(y_val_clean_np, val_preds_unscaled)
    
    np_val_mse_clean = mean_squared_error(y_val_clean_np, np_val_preds_unscaled)
    np_val_mae_clean = mean_absolute_error(y_val_clean_np, np_val_preds_unscaled)
    
    ec_val_mse_clean = mean_squared_error(y_val_clean_np, ec_val_preds_unscaled)
    ec_val_mae_clean = mean_absolute_error(y_val_clean_np, ec_val_preds_unscaled)
    
    pinn_val_mse_clean = mean_squared_error(y_val_clean_np, pinn_val_preds_unscaled)
    pinn_val_mae_clean = mean_absolute_error(y_val_clean_np, pinn_val_preds_unscaled)
    
    # Create parity plots based on non-noisy (clean) data
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    for row, (dataset_name, y_true_clean, const_preds, const_stds, np_preds, np_stds, ec_preds, ec_stds, pinn_preds, pinn_stds) in enumerate([
        ('Train (vs Clean)', y_train_clean_np, train_preds_unscaled, train_stds, np_train_preds_unscaled, np_train_stds, ec_train_preds_unscaled, ec_train_stds, pinn_train_preds_unscaled, pinn_train_stds),
        ('Validation (vs Clean)', y_val_clean_np, val_preds_unscaled, val_stds, np_val_preds_unscaled, np_val_stds, ec_val_preds_unscaled, ec_val_stds, pinn_val_preds_unscaled, pinn_val_stds),
        ('Test (vs Clean)', y_test_clean_np, test_preds_unscaled, test_stds, np_test_preds_unscaled, np_test_stds, ec_test_preds_unscaled, ec_test_stds, pinn_test_preds_unscaled, pinn_test_stds)
    ]):
        
        for col in range(3):
            ax = axes[row, col]
            
            # Sort data by true values for smooth lines
            sort_idx = np.argsort(y_true_clean[:, col])
            x_sorted = y_true_clean[sort_idx, col]
            
            # Constrained model
            y_const_sorted = const_preds[sort_idx, col]
            std_const_sorted = const_stds[sort_idx, col]
            ax.plot(x_sorted, y_const_sorted, '-', color='blue', alpha=0.8, label='Constrained', linewidth=2)
            ax.fill_between(x_sorted, y_const_sorted - std_const_sorted, y_const_sorted + std_const_sorted, 
                            color='blue', alpha=0.2)
            
            # Unconstrained model
            y_np_sorted = np_preds[sort_idx, col]
            std_np_sorted = np_stds[sort_idx, col]
            ax.plot(x_sorted, y_np_sorted, '-', color='red', alpha=0.8, label='Unconstrained', linewidth=2)
            ax.fill_between(x_sorted, y_np_sorted - std_np_sorted, y_np_sorted + std_np_sorted, 
                            color='red', alpha=0.2)
            
            # EC model
            y_ec_sorted = ec_preds[sort_idx, col]
            std_ec_sorted = ec_stds[sort_idx, col]
            ax.plot(x_sorted, y_ec_sorted, '-', color='green', alpha=0.8, label='EC-NN', linewidth=2)
            ax.fill_between(x_sorted, y_ec_sorted - std_ec_sorted, y_ec_sorted + std_ec_sorted, 
                            color='green', alpha=0.2)
            
            # # PINN model
            # y_pinn_sorted = pinn_preds[sort_idx, col]
            # std_pinn_sorted = pinn_stds[sort_idx, col]
            # ax.plot(x_sorted, y_pinn_sorted, '-', color='orange', alpha=0.8, label='PINN', linewidth=2)
            # ax.fill_between(x_sorted, y_pinn_sorted - std_pinn_sorted, y_pinn_sorted + std_pinn_sorted, 
            #                 color='orange', alpha=0.2)
            
            # Perfect prediction line
            ax.plot([y_true_clean[:, col].min(), y_true_clean[:, col].max()], 
                    [y_true_clean[:, col].min(), y_true_clean[:, col].max()], 
                    'k--', lw=2, label='Perfect prediction')
            
            ax.set_xlabel(f'True {output_names[col]} (Clean)')
            ax.set_ylabel(f'Predicted {output_names[col]}')
            ax.set_title(f'{dataset_name} - {output_names[col]}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    print("\nAccuracy Metrics (vs Clean Data):")
    print("Test Set:")
    print(f"  Constrained Model - MSE: {const_test_mse_clean:.6f}, MAE: {const_test_mae_clean:.6f}")
    print(f"  Unconstrained Model - MSE: {np_test_mse_clean:.6f}, MAE: {np_test_mae_clean:.6f}")
    print(f"  EC-NN Model - MSE: {ec_test_mse_clean:.6f}, MAE: {ec_test_mae_clean:.6f}")
    print(f"  PINN Model - MSE: {pinn_test_mse_clean:.6f}, MAE: {pinn_test_mae_clean:.6f}")
    
    print("Train Set:")
    print(f"  Constrained Model - MSE: {const_train_mse_clean:.6f}, MAE: {const_train_mae_clean:.6f}")
    print(f"  Unconstrained Model - MSE: {np_train_mse_clean:.6f}, MAE: {np_train_mae_clean:.6f}")
    print(f"  EC-NN Model - MSE: {ec_train_mse_clean:.6f}, MAE: {ec_train_mae_clean:.6f}")
    print(f"  PINN Model - MSE: {pinn_train_mse_clean:.6f}, MAE: {pinn_train_mae_clean:.6f}")
    
    print("Validation Set:")
    print(f"  Constrained Model - MSE: {const_val_mse_clean:.6f}, MAE: {const_val_mae_clean:.6f}")
    print(f"  Unconstrained Model - MSE: {np_val_mse_clean:.6f}, MAE: {np_val_mae_clean:.6f}")
    print(f"  EC-NN Model - MSE: {ec_val_mse_clean:.6f}, MAE: {ec_val_mae_clean:.6f}")
    print(f"  PINN Model - MSE: {pinn_val_mse_clean:.6f}, MAE: {pinn_val_mae_clean:.6f}")
    
    # Calculate coverage ratio (percentage of true values within confidence intervals)
    def calculate_coverage(y_true, y_pred, stds, confidence_level=0.95):
        z_score = 1.96  # For 95% confidence
        lower_bound = y_pred - z_score * stds
        upper_bound = y_pred + z_score * stds
        
        within_bounds = np.logical_and(y_true >= lower_bound, y_true <= upper_bound)
        coverage = np.mean(within_bounds)
        return coverage
    
    const_coverage = calculate_coverage(y_test_clean_np, test_preds_unscaled, test_stds)
    np_coverage = calculate_coverage(y_test_clean_np, np_test_preds_unscaled, np_test_stds)
    ec_coverage = calculate_coverage(y_test_clean_np, ec_test_preds_unscaled, ec_test_stds)
    pinn_coverage = calculate_coverage(y_test_clean_np, pinn_test_preds_unscaled, pinn_test_stds)
    
    print("\nCoverage Ratios (95% confidence):")
    print(f"Constrained Model: {const_coverage:.3f} ({const_coverage*100:.1f}%)")
    print(f"Unconstrained Model: {np_coverage:.3f} ({np_coverage*100:.1f}%)")
    print(f"EC-NN Model: {ec_coverage:.3f} ({ec_coverage*100:.1f}%)")
    print(f"PINN Model: {pinn_coverage:.3f} ({pinn_coverage*100:.1f}%)")
    
    # Calculate uncertainty bound widths for each output variable
    def calculate_bound_widths(stds, confidence_level=0.95):
        z_score = 1.96  # For 95% confidence
        bound_widths = 2 * z_score * stds  # Total width of confidence interval
        return bound_widths
    
    # Calculate bound widths for test set
    const_bound_widths = calculate_bound_widths(test_stds)
    np_bound_widths = calculate_bound_widths(np_test_stds)
    ec_bound_widths = calculate_bound_widths(ec_test_stds)
    pinn_bound_widths = calculate_bound_widths(pinn_test_stds)
    
    print("\nUncertainty Bound Widths (95% confidence intervals) - Test Set:")
    for i, var_name in enumerate(output_names):
        print(f"{var_name}:")
        print(f"  Constrained Model - Mean: {np.mean(const_bound_widths[:, i]):.6f}, "
              f"Std: {np.std(const_bound_widths[:, i]):.6f}, "
              f"Min: {np.min(const_bound_widths[:, i]):.6f}, "
              f"Max: {np.max(const_bound_widths[:, i]):.6f}")
        print(f"  Unconstrained Model - Mean: {np.mean(np_bound_widths[:, i]):.6f}, "
              f"Std: {np.std(np_bound_widths[:, i]):.6f}, "
              f"Min: {np.min(np_bound_widths[:, i]):.6f}, "
              f"Max: {np.max(np_bound_widths[:, i]):.6f}")
        print(f"  EC-NN Model - Mean: {np.mean(ec_bound_widths[:, i]):.6f}, "
              f"Std: {np.std(ec_bound_widths[:, i]):.6f}, "
              f"Min: {np.min(ec_bound_widths[:, i]):.6f}, "
              f"Max: {np.max(ec_bound_widths[:, i]):.6f}")
        print(f"  PINN Model - Mean: {np.mean(pinn_bound_widths[:, i]):.6f}, "
              f"Std: {np.std(pinn_bound_widths[:, i]):.6f}, "
              f"Min: {np.min(pinn_bound_widths[:, i]):.6f}, "
              f"Max: {np.max(pinn_bound_widths[:, i]):.6f}")
        print()
    
if __name__ == "__main__":
    main()