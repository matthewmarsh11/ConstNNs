from utils_new import *
from models.Gaussian_HPINN import KKT_PPINN
from models.mlp import MLP
from models.ec_nn import EC_NN
from base import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
np.random.seed(42)
torch.manual_seed(42)

def main():
    fts = pd.read_csv('datasets/small_cstr_features.csv')
    fts = fts.to_numpy()
    fts = fts[:, :-1]  # remove the last column (simulation number)
    
    
    q = 100 # parameters in the material balance
    V = 100
    k01 = 7.2e10
    EA1overR = 8750
    
    # standard x and y prior to lifting transformation
    # previous y = [ca, cb, cc, t]; new y = [1/ca dc\dt, cain / ca, cb, cc, exp(-ea/RT)]
    y_orig = fts[:, :4]
    x_orig = fts[:, 4:]
    
    y0dcadt =  np.array([0]+ [(1/y_orig[i, 0]) * (y_orig[i, 0] - y_orig[i+1, 0]) for i in range(len(y_orig)-1)])# finite difference for dc/dt
    y0dcadt[::99] = 0  # set the first element of each simulation to 0
    y1cainca = x_orig[:, 1] / y_orig[:, 0]
    y4 = np.exp(-EA1overR / y_orig[:, 3])  # exp(-EA/RT)
    
    y_new = np.column_stack((y0dcadt, y1cainca, y_orig[:, 1], y_orig[:, 2], y4))
    
    x_orig = pd.DataFrame(x_orig)
    y_new = pd.DataFrame(y_new)
    
    training_config = TrainingConfig(
        batch_size=82,
        num_epochs=100,
        learning_rate=0.0027572347845456853,
        weight_decay=0.004978847817701316,
        factor=0.41760043679687436,
        patience=11,
        delta = 0.00013169798514440654,
        train_test_split=0.6,
        test_val_split=0.8,
        # device = "mps" if torch.backends.mps.is_available() else "cpu",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )
    
    MLP_Config = MLPConfig(
        hidden_dim = 323,
        num_layers = 3,
        dropout = 0.2,
        activation = 'ReLU',
        # device = "mps" if torch.backends.mps.is_available() else "cpu",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )
    
    data_processor = DataProcessor(training_config, x_orig, y_new, num_simulations=10)

    (X_train, X_test, X_val, y_train, y_test, y_val, X, y) = data_processor.prepare_data(simulation_length=99)

    # constraint: q/V * y2 - q/v - k0 * y4 - 1 * y1 = 0
    # constraint: [-1, q/v, 0, 0, -k0] y + [0] x = [q/V]
    
    A = torch.tensor([[0, 0, 0]], dtype=torch.float32)  # Coefficients for x2 - x3
    B = torch.tensor([[-1, q/V, 0, 0, -k01]], dtype=torch.float32)  # Coefficients for -y2 + y3
    b = torch.tensor([[q/V]], dtype=torch.float32)  # Right-hand side of the constraint
    
    # Scale the constraint matrices to work with scaled data
    # Scale A matrix according to input scaling
    x_scale = torch.tensor(data_processor.feature_scaler.scale_, dtype=torch.float32)
    A_constr_model = A / x_scale.unsqueeze(0)
    
    # Scale B matrix according to output scaling
    y_scale = torch.tensor(data_processor.target_scaler.scale_, dtype=torch.float32)
    B_constr_model = B / y_scale.unsqueeze(0)
    
    # Scale b vector - for linear constraints, b typically doesn't need scaling
    # since the constraint Ax + By = b becomes (A/sx)x_scaled + (B/sy)y_scaled = b
    b_constr_model = b
    
    constr = torch.zeros((X_train.shape[0], b.shape[0]), dtype=torch.float32)
    nn_constr = torch.zeros((X_train.shape[0], b.shape[0]), dtype=torch.float32)
    scaled_constr = torch.zeros((X_train.shape[0], b.shape[0]), dtype=torch.float32)

    # for i in range(X_train.shape[0]):
    #     for j in range(b.shape[0]):
    #         constr[i, j] = torch.matmul(A[j, :], x[i]) + torch.matmul(B[j, :], y) - b[j]
    #         nn_constr[i, j] = torch.matmul(A[j, :], x[i]) + torch.matmul(B[j, :], y[i]) - b[j]
    #         scaled_constr[i, j] = torch.matmul(A_constr_model[j, :], X_train[i]) + torch.matmul(B_constr_model[j, :], y_train[i]) - b_constr_model[j]
        

    epsilon = float(0.2875)
    
    model = KKT_PPINN(
        config = MLP_Config,
        input_dim = X_train.shape[1],
        output_dim = y_train.shape[1],
        A = A_constr_model,  # Use scaled A
        B = B_constr_model,  # Use scaled B
        b = b_constr_model,  # Use scaled b
        epsilon = epsilon,
        probability_level = 0.95
    )
    
    np_model = MLP(
        config = MLP_Config,
        input_dim = X_train.shape[1],
        output_dim = y_train.shape[1],
        num_samples = None
    )
    
    ec_model = EC_NN(
        config = MLP_Config,
        input_dim = X_train.shape[1],
        output_dim = y_train.shape[1],
        A = A_constr_model,  # Use scaled A
        B = B_constr_model,  # Use scaled B
        b = b_constr_model,  # Use scaled b
        dependent_ids=[1],  # Indices of dependent variables
    )
    
    pinn_model = MLP(
        config = MLP_Config,
        input_dim = X_train.shape[1],
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
        X_train, y_train, X_test, y_test, X_val, y_val, criterion
    )
    np_model, np_history, np_avg_loss = trainer_np.train(
        X_train, y_train, X_test, y_test, X_val, y_val, criterion
    )
    ec_model, ec_history, ec_avg_loss = trainer_ec.train(
        X_train, y_train, X_test, y_test, X_val, y_val, criterion
    )
    lamb = 0.1
    pinn_model, pinn_history, pinn_avg_loss = trainer_pinn.train(
        X_train, y_train, X_test, y_test, X_val, y_val, criterion,
        A_constr_model, B_constr_model, b_constr_model, lamb
    )
    
    model.eval()
    np_model.eval()
    ec_model.eval()
    pinn_model.eval()
    
    test_preds, test_covs = model(X_test)
    np_test_preds, np_test_covs = np_model(X_test)
    ec_test_preds, ec_test_covs = ec_model(X_test)
    pinn_test_preds, pinn_test_covs = pinn_model(X_test)
    
    # Get predictions for train and validation sets
    train_preds, train_covs = model(X_train)
    np_train_preds, np_train_covs = np_model(X_train)
    ec_train_preds, ec_train_covs = ec_model(X_train)
    pinn_train_preds, pinn_train_covs = pinn_model(X_train)
    
    val_preds, val_covs = model(X_val)
    np_val_preds, np_val_covs = np_model(X_val)
    ec_val_preds, ec_val_covs = ec_model(X_val)
    pinn_val_preds, pinn_val_covs = pinn_model(X_val)
    num_simulations = 10
    X = X.reshape(num_simulations, -1, X.shape[-1])
    # Create simulation dictionaries for all models
    simulations = {i: None for i in range(X.shape[0])}
    np_simulations = {i: None for i in range(X.shape[0])}
    ec_simulations = {i: None for i in range(X.shape[0])}
    pinn_simulations = {i: None for i in range(X.shape[0])}
    inputs = {i: X[i, :, :].cpu().numpy() for i in range(X.shape[0])}

    for i in range(X.shape[0]):
        with torch.no_grad():
            preds, covs = model(X[i, :, :].to(training_config.device))
            np_preds, np_covs = np_model(X[i, :, :].to(training_config.device))
            ec_preds, ec_covs = ec_model(X[i, :, :].to(training_config.device))
            pinn_preds, pinn_covs = pinn_model(X[i, :, :].to(training_config.device))
            
            # Process KKT_PPINN predictions
            preds = preds.cpu().numpy()
            covs = torch.matmul(covs, covs.transpose(1, 2)).cpu().numpy()
            simulations[i] = (preds, covs)
            
            # Process MLP predictions
            np_preds = np_preds.cpu().numpy()
            np_covs = torch.matmul(np_covs, np_covs.transpose(1, 2)).cpu().numpy()
            np_simulations[i] = (np_preds, np_covs)
            
            # Process EC_NN predictions
            ec_preds = ec_preds.cpu().numpy()
            ec_covs = torch.matmul(ec_covs, ec_covs.transpose(1, 2)).cpu().numpy()
            ec_simulations[i] = (ec_preds, ec_covs)
            
            # Process PINN predictions
            pinn_preds = pinn_preds.cpu().numpy()
            pinn_covs = torch.matmul(pinn_covs, pinn_covs.transpose(1, 2)).cpu().numpy()
            pinn_simulations[i] = (pinn_preds, pinn_covs)
            
    # Plot the trajectory for each simulation
    for sim_idx in range(len(simulations)):
        preds, covs = simulations[sim_idx]
        np_preds, np_covs = np_simulations[sim_idx]
        ec_preds, ec_covs = ec_simulations[sim_idx]
        pinn_preds, pinn_covs = pinn_simulations[sim_idx]
        inputs_sim = inputs[sim_idx]
        # Inverse transform if needed
        
        x = data_processor.feature_scaler.inverse_transform(inputs_sim)
        preds = data_processor.target_scaler.inverse_transform(preds)
        np_preds = data_processor.target_scaler.inverse_transform(np_preds)
        ec_preds = data_processor.target_scaler.inverse_transform(ec_preds)
        pinn_preds = data_processor.target_scaler.inverse_transform(pinn_preds)
        
        scale_factor = data_processor.target_scaler.data_max_ - data_processor.target_scaler.data_min_
        stds = np.sqrt(np.diagonal(covs, axis1=1, axis2=2) * (scale_factor**2))
        np_stds = np.sqrt(np.diagonal(np_covs, axis1=1, axis2=2) * (scale_factor**2))
        ec_stds = np.sqrt(np.diagonal(ec_covs, axis1=1, axis2=2) * (scale_factor**2))
        pinn_stds = np.sqrt(np.diagonal(pinn_covs, axis1=1, axis2=2) * (scale_factor**2))
        
        # change the variables back to original dimensions
        Ca = 1/preds[:, 1] * x[:, 0]
        np_Ca = 1/np_preds[:, 1] * x[:, 0]
        ec_Ca = 1/ec_preds[:, 1] * x[:, 0]
        pinn_Ca = 1/pinn_preds[:, 1] * x[:, 0]
        
        Ca_std = stds[:, 1] * x[:, 0]
        np_Ca_std = np_stds[:, 1] * x[:, 0]
        ec_Ca_std = ec_stds[:, 1] * x[:, 0]
        pinn_Ca_std = pinn_stds[:, 1] * x[:, 0]
        
        T = 1/np.log(preds[:,4]) * (EA1overR)  # exp(-EA/RT) -> log transformation
        np_T = 1/np.log(np_preds[:,4]) * (EA1overR)
        ec_T = 1/np.log(ec_preds[:,4]) * (EA1overR)
        pinn_T = 1/np.log(pinn_preds[:,4]) * (EA1overR)
        
        T_std = np.log(stds[:, 4]) * (EA1overR)
        np_T_std = np.log(np_stds[:, 4]) * (EA1overR)
        ec_T_std = np.log(ec_stds[:, 4]) * (EA1overR)
        pinn_T_std = np.log(pinn_stds[:, 4]) * (EA1overR)
        
        preds = np.column_stack((Ca, preds[:, 2], preds[:, 3], T))
        np_preds = np.column_stack((np_Ca, np_preds[:, 2], np_preds[:, 3], np_T))
        ec_preds = np.column_stack((ec_Ca, ec_preds[:, 2], ec_preds[:, 3], ec_T))
        pinn_preds = np.column_stack((pinn_Ca, pinn_preds[:, 2], pinn_preds[:, 3], pinn_T))
        
        stds = np.column_stack((Ca_std, stds[:, 2], stds[:, 3], T_std))
        np_stds = np.column_stack((np_Ca_std, np_stds[:, 2], np_stds[:, 3], np_T_std))
        ec_stds = np.column_stack((ec_Ca_std, ec_stds[:, 2], ec_stds[:, 3], ec_T_std))
        pinn_stds = np.column_stack((pinn_Ca_std, pinn_stds[:, 2], pinn_stds[:, 3], pinn_T_std))

        feature_names = ['Ca', 'Cb', 'Cc', 'T']
        y_orig = y_orig.reshape(-1, 99, 4)
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Simulation {sim_idx+1}', fontsize=16)
        
        for i in range(preds.shape[1]):
            plt.subplot(math.ceil(preds.shape[1]/2), 2, i+1)
            
            # # Plot ground truth data
            plt.plot(y_orig[sim_idx, :, i], label='Noisy Data', color='green')
            # plt.plot(noiseless_targets[sim_idx, :, i], label='Noiseless Data', color='black', linestyle='dashed')
            
            # Plot predictions with uncertainty for all models
            time_steps = range(len(preds))
            
            # KKT_PPINN
            plt.plot(preds[:, i], label=f'KKTPPINN {feature_names[i]}', color='blue')
            plt.fill_between(time_steps, 
                           preds[:, i] - 1.8*stds[:, i],
                           preds[:, i] + 1.8*stds[:, i],
                           color='blue', alpha=0.2, label=f'{feature_names[i]} Uncertainty')
            
            # MLP
            plt.plot(np_preds[:, i], label=f'Non-Projected {feature_names[i]}',
                     linestyle='dashed', color='red')
            plt.fill_between(time_steps,
                             np_preds[:, i] - 1.8*np_stds[:, i],
                             np_preds[:, i] + 1.8*np_stds[:, i],
                             color='red', alpha=0.2, label=f'Non-Projected Uncertainty')
            
            # EC_NN
            plt.plot(ec_preds[:, i], label=f'EC_NN {feature_names[i]}',
                     linestyle='dotted', color='orange')
            plt.fill_between(time_steps,
                             ec_preds[:, i] - 1.8*ec_stds[:, i],
                             ec_preds[:, i] + 1.8*ec_stds[:, i],
                             color='orange', alpha=0.2, label=f'EC_NN Uncertainty')
            
            # PINN
            plt.plot(pinn_preds[:, i], label=f'PINN {feature_names[i]}',
                     linestyle='dashdot', color='purple')
            plt.fill_between(time_steps,
                             pinn_preds[:, i] - 1.8*pinn_stds[:, i],
                             pinn_preds[:, i] + 1.8*pinn_stds[:, i],
                             color='purple', alpha=0.2, label=f'PINN Uncertainty')
            
            plt.title(f'{feature_names[i]} Trajectory')
            plt.xlabel('Time step')
            plt.ylabel(f'{feature_names[i]} value')
            plt.legend()
            
        plt.tight_layout()
        plt.show()
    
    
if __name__ == "__main__":
    main()