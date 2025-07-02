from batch_data_loader import load_batch_data
from models.Gaussian_HPINN import KKT_PPINN
from utils_new import ModelSaver
import torch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

def main():
    # Load processed data and constraint matrices
    data = load_batch_data()

    training_config = data["training_config"]
    model_config = data["model_config"]
    data_processor = data["data_processor"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    X_val = data["X_val"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    y_val = data["y_val"]
    # X_train_unconst = data["X_train_unconst"]
    # X_test_unconst = data["X_test_unconst"]
    # X_val_unconst = data["X_val_unconst"]
    X_tensor = data["X_tensor"]
    y_tensor = data["y_tensor"]
    # X_tensor_unconst = data["X_tensor_unconst"]
    scaled_A = data["scaled_A"]
    scaled_B = data["scaled_B"]
    scaled_b = data["scaled_b"]
    batch_data = data["batch_data"]
    noiseless_batch = data["noiseless_batch"]
    num_simulations = data["num_simulations"]
    simulation_length = data["simulation_length"]

    # Load trained model
    model_saver = ModelSaver()
    kkt_ppinn = model_saver.load_full_model('models/batch_kkt_ppinn')
    mlp = model_saver.load_full_model('models/batch_mlp')
    ec_nn = model_saver.load_full_model('models/batch_ec_nn')
    kkt_ppinn.eval()
    mlp.eval()
    ec_nn.eval()

    # Prepare predictions container
    
    X_tensor = X_tensor.reshape(num_simulations, -1, X_tensor.shape[1])
    # X_tensor_unconst = X_tensor_unconst.reshape(num_simulations, -1, X_tensor_unconst.shape[1])
    simulations = {}

    for i in range(num_simulations):
        with torch.no_grad():
            # KKT-PPINN predictions
            kkt_preds, kkt_covs = kkt_ppinn(X_tensor[i].to(training_config.device))
            
            # MLP predictions
            mlp_preds, mlp_covs = mlp(X_tensor[i].to(training_config.device))
            
            # EC-NN predictions
            ec_preds, ec_covs = ec_nn(X_tensor[i].to(training_config.device))
            
            simulations[i] = {
                'kkt': (kkt_preds.cpu().numpy(), kkt_covs.cpu().numpy()),
                'mlp': (mlp_preds.cpu().numpy(), mlp_covs.cpu().numpy()),
                'ec': (ec_preds.cpu().numpy(), ec_covs.cpu().numpy())
            }

    # Visualisation
    for sim_idx, preds_dict in simulations.items():
        # Inverse transform predictions and unpack covariances
        kkt_preds, kkt_covs = preds_dict['kkt']
        mlp_preds, mlp_covs = preds_dict['mlp']
        ec_preds, ec_covs = preds_dict['ec']
        
        kkt_preds = data_processor.target_scaler.inverse_transform(kkt_preds)
        mlp_preds = data_processor.target_scaler.inverse_transform(mlp_preds)
        ec_preds = data_processor.target_scaler.inverse_transform(ec_preds)
        inputs = data_processor.feature_scaler.inverse_transform(X_tensor[sim_idx])

        
        # Compute std from covariance (scaled)
        scale_factor = data_processor.target_scaler.data_max_ - data_processor.target_scaler.data_min_
        
        kkt_stds = np.sqrt(np.diagonal(kkt_covs, axis1=1, axis2=2)) * (scale_factor**2)
        mlp_stds = np.sqrt(np.diagonal(mlp_covs, axis1=1, axis2=2)) * (scale_factor**2)
        ec_stds = np.sqrt(np.diagonal(ec_covs, axis1=1, axis2=2)) * (scale_factor**2)

        A = np.array([-2.0, -1.0, -1.0, 0.0, 0.0, 0.0])
        B = np.array([2.0, 1.0, 1.0, 0.0])
        b = np.array([0.0])
        unconst_A = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
        # Find constraint violations:
        kkt_violations = A @ inputs.T + B @ kkt_preds.T - b
        mlp_violations = A @ inputs.T + B @ mlp_preds.T - b
        ec_violations = A @ inputs.T + B @ ec_preds.T - b
        b_sim = B @ noiseless_batch[sim_idx * simulation_length:(sim_idx + 1) * simulation_length, :4].T
        noisy_violations = B @ batch_data[sim_idx * simulation_length:(sim_idx + 1) * simulation_length, :4].T - b_sim
        
        # Plot violations
        plt.figure(figsize=(15, 6))
        plt.suptitle(f'Simulation {sim_idx + 1} Constraint Violations')
        
        plt.subplot(1, 2, 1)
        plt.plot(kkt_violations.flatten(), label='KKT-PPINN Violations', linewidth=2)
        plt.plot(mlp_violations.flatten(), label='MLP Violations', linewidth=2)
        plt.plot(ec_violations.flatten(), label='EC-NN Violations', linewidth=2)
        plt.plot(noisy_violations.flatten(), label='Noisy Data Violations', linestyle=':', color='gray')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Constraint Boundary')
        plt.title('Constraint Violations')
        plt.xlabel('Time Step')
        plt.ylabel('Violation Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Plot violation uncertainties (propagated through constraint)
        kkt_violation_std = np.sqrt((B**2) @ kkt_stds.T**2).flatten()
        mlp_violation_std = np.sqrt((B**2) @ mlp_stds.T**2).flatten()
        ec_violation_std = np.sqrt((B**2) @ ec_stds.T**2).flatten()
        
        plt.fill_between(range(len(kkt_violations.flatten())), 
                kkt_violations.flatten() - kkt_violation_std, 
                kkt_violations.flatten() + kkt_violation_std, 
                alpha=0.3, label='KKT Violation Uncertainty')
        # plt.fill_between(range(len(mlp_violations.flatten())), 
        #         mlp_violations.flatten() - mlp_violation_std, 
        #         mlp_violations.flatten() + mlp_violation_std, 
        #         alpha=0.3, label='MLP Violation Uncertainty')
        # plt.fill_between(range(len(ec_violations.flatten())), 
        #         ec_violations.flatten() - ec_violation_std, 
        #         ec_violations.flatten() + ec_violation_std, 
        #         alpha=0.3, label='EC Violation Uncertainty')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Constraint Boundary')
        plt.title('Violation Uncertainty Bands')
        plt.xlabel('Time Step')
        plt.ylabel('Violation Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        # Plotting
        plt.figure(figsize=(15, 8))
        plt.suptitle(f'Simulation {sim_idx + 1} Predictions')

        for i in range(kkt_preds.shape[1]):
            plt.subplot(2, 2, i + 1)
            plt.plot(kkt_preds[:, i], label='KKT-PPINN', linewidth=2)
            plt.fill_between(range(len(kkt_preds)), kkt_preds[:, i] - kkt_stds[:, i], kkt_preds[:, i] + kkt_stds[:, i], alpha=0.3, label='KKT Uncertainty')
            plt.plot(mlp_preds[:, i], label='MLP', linewidth=2)
            # plt.fill_between(range(len(mlp_preds)), mlp_preds[:, i] - mlp_stds[:, i], mlp_preds[:, i] + mlp_stds[:, i], alpha=0.3, label='MLP Uncertainty')
            plt.plot(ec_preds[:, i], label='EC-NN', linewidth=2)
            # plt.fill_between(range(len(ec_preds)), ec_preds[:, i] - ec_stds[:, i], ec_preds[:, i] + ec_stds[:, i], alpha=0.3, label='EC Uncertainty')
            plt.plot(noiseless_batch[sim_idx * simulation_length:(sim_idx + 1) * simulation_length, i], label='True', linestyle='--', color='black')
            plt.plot(batch_data[sim_idx * simulation_length:(sim_idx + 1) * simulation_length, i], label='Noisy', linestyle=':', color='gray')
            plt.title(f'Output {i + 1}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()