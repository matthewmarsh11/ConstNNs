from ss_cstr_data_loader import load_ss_cstr_data
from utils_new import ModelSaver
import torch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

def main():
    # load the processed data and constraint matrices
    data = load_ss_cstr_data()
    
    training_config = data["training_config"]
    model_config = data["model_config"]
    data_processor = data["data_processor"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    X_val = data["X_val"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    y_val = data["y_val"]
    X_tensor = data["X_tensor"]
    y_tensor = data["y_tensor"]
    scaled_A = data["scaled_A"]
    scaled_B = data["scaled_B"]
    scaled_b = data["scaled_b"]
    noiseless_data = data["noiseless_data"]
    noisy_data = data["noisy_data"]
    
    model_saver = ModelSaver()
    kkt_ppinn = model_saver.load_full_model('models/cstr_kkt_ppinn')
    mlp = model_saver.load_full_model('models/cstr_mlp')
    ec_nn = model_saver.load_full_model('models/cstr_ec_nn')
    # standard pinn
    kkt_ppinn.eval()
    mlp.eval()
    ec_nn.eval()

    with torch.no_grad():
        kkt_preds, kkt_covs = kkt_ppinn(X_tensor.to(training_config.device))
        mlp_preds, mlp_covs = mlp(X_tensor.to(training_config.device))
        ec_preds, ec_covs = ec_nn(X_tensor.to(training_config.device))
    
    # inverse the predictions
    
    kkt_preds = data_processor.target_scaler.inverse_transform(kkt_preds.cpu().numpy())
    mlp_preds = data_processor.target_scaler.inverse_transform(mlp_preds.cpu().numpy())
    ec_preds = data_processor.target_scaler.inverse_transform(ec_preds.cpu().numpy())
    
    scale_factor = data_processor.target_scaler.data_max_ - data_processor.target_scaler.data_min_
    kkt_covs = kkt_covs.cpu().numpy() * (scale_factor ** 2)
    mlp_covs = mlp_covs.cpu().numpy() * (scale_factor ** 2)
    ec_covs = ec_covs.cpu().numpy() * (scale_factor ** 2)
    
  
if __name__ == "__main__":
    main()