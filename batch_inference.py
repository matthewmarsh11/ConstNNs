from batch_data_loader import load_batch_data
from models.Gaussian_HPINN import KKT_PPINN
from utils_new import ModelSaver
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

def main():
    # Load processed data and constraint matrices
    data = load_batch_data()

    training_config = data["training_config"]
    model_config = data["model_config"]
    const_data_processor = data["const_data_processor"]
    unconst_data_processor = data["unconst_data_processor"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    X_val = data["X_val"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    y_val = data["y_val"]
    X_train_unconst = data["X_train_unconst"]
    X_test_unconst = data["X_test_unconst"]
    X_val_unconst = data["X_val_unconst"]
    X_tensor = data["X_tensor"]
    y_tensor = data["y_tensor"]
    X_tensor_unconst = data["X_tensor_unconst"]
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
    kkt_ppinn.eval()

    # Prepare predictions container
    X_tensor = X_tensor.reshape(num_simulations, -1, X_tensor.shape[1])
    simulations = {}

    for i in range(num_simulations):
        with torch.no_grad():
            preds, covs = kkt_ppinn(X_tensor[i].to(training_config.device))
            simulations[i] = (preds.cpu().numpy(), covs.cpu().numpy())

    # Visualisation
    for sim_idx, (preds, covs) in simulations.items():
        # Inverse transform predictions
        preds = const_data_processor.target_scaler.inverse_transform(preds)
        
        # Compute std from covariance (scaled)
        scale_factor = const_data_processor.target_scaler.data_max_ - const_data_processor.target_scaler.data_min_
        stds = np.sqrt(np.diagonal(covs, axis1=1, axis2=2)) * (scale_factor**2)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.suptitle(f'Simulation {sim_idx + 1} Predictions')

        for i in range(preds.shape[1]):
            plt.subplot(2, 2, i + 1)
            plt.plot(preds[:, i], label='Predicted')
            plt.fill_between(range(len(preds)), preds[:, i] - stds[:, i], preds[:, i] + stds[:, i], alpha=0.3, label='Uncertainty')
            plt.plot(noiseless_batch[sim_idx * simulation_length:(sim_idx + 1) * simulation_length, i], label='True', linestyle='--')
            plt.plot(batch_data[sim_idx * simulation_length:(sim_idx + 1) * simulation_length, i], label='Noisy', linestyle=':')
            plt.title(f'Output {i + 1}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()