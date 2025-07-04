from utils_new import *
from models.mlp import *
from models.mcd_nn import *
from models.ec_nn import *
from base import *

def main():
    
    training_config = TrainingConfig(
        batch_size=92,
        num_epochs=100,
        learning_rate=3.266982523281954e-05,
        weight_decay=1.0275474727234102e-05,
        factor=0.2790682806199674,
        patience=19,
        delta = 0.0037385829906140906,
        train_test_split=0.6,
        test_val_split=0.8,
        # device = "mps" if torch.backends.mps.is_available() else "cpu",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )
    
    MLP_Config = MLPConfig(
        hidden_dim = 1212,
        num_layers = 4,
        dropout = 0.2,
        activation = 'ReLU',
        # device = "mps" if torch.backends.mps.is_available() else "cpu",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )
    
    features_path = 'datasets/tank_system_features.csv'
    targets_path = 'datasets/tank_system_targets.csv'
    noiseless_path = 'datasets/tank_system_noiseless_features.csv'
    
    features = pd.read_csv(features_path)
    features = features.drop('V1_s', axis=1)
    targets = features[['V1', 'C1', 'V2', 'C2']]
    features = features[['F_in1', 'F_in2']]
    noiseless_results = pd.read_csv(noiseless_path)
    noiseless_results = noiseless_results.drop('V1_s', axis=1)
    noiseless_targets = noiseless_results[['V1', 'C1', 'V2', 'C2']].to_numpy()
    noiseless_features = noiseless_results[['F_in1', 'F_in2']].to_numpy()
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
    A = torch.Tensor([0, 0])
    B = torch.Tensor([1, 0, 1, 0])
    b = torch.Tensor([100])
    
    device = MLP_Config.device

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    
    model = MLP(
        config = MLP_Config,
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        num_samples = None
    )

    from ReweightedNLL import ReweightedNLL_FullCovariance
    from mv_gaussian_nll import GaussianMVNLL
    criterion = GaussianMVNLL()
    # criterion = ReweightedNLL_FullCovariance()
    # criterion = nn.MSELoss()
    
    trainer = ModelTrainer(model, training_config)
    model, history, avg_loss = trainer.train(X_train, y_train, X_test, y_test, X_val, y_val, criterion)
    X_tensor = X_tensor.reshape(num_simulations, -1, X_tensor.shape[1])
    noisy_targets = targets.to_numpy().reshape(num_simulations, -1, y_tensor.shape[1])
    noiseless_targets = noiseless_targets.reshape(num_simulations, -1, y_tensor.shape[1])
    features = features.to_numpy().reshape(num_simulations, -1, X_tensor.shape[2])
    noiseless_features = noiseless_features.reshape(num_simulations, -1, X_tensor.shape[2])
    
    model.eval()
    # model.enable_dropout()
    feature_names = ['V1', 'C1', 'V2', 'C2']
    simulations = {}
    simulations = {i: None for i in range(X_tensor.shape[0])}
    np_simulations = {}
    np_simulations = {i: None for i in range(X_tensor.shape[0])}
    # for i in range (X_tensor.shape[0]):
    #     with torch.no_grad():
    #         prj_preds, nprj_preds = model(X_tensor[i, :, :].to(training_config.device))
    #         simulations[i] = prj_preds.cpu().numpy()
    #         np_simulations[i] = nprj_preds.cpu().numpy()
            
    for i in range(X_tensor.shape[0]):
        with torch.no_grad():
            preds, covs = model(X_tensor[i, :, :].to(training_config.device))
            preds = preds.cpu().numpy()
            covs = covs.cpu().numpy()
            simulations[i] = (preds, covs)
            
    # Plot the trajectory for each simulation
    for sim_idx in range(len(simulations)):
        preds, covs = simulations[sim_idx]
        
        # Inverse transform if needed
        preds = data_processor.target_scaler.inverse_transform(preds)
        
        scale_factor = np.diag(data_processor.target_scaler.data_max_ - data_processor.target_scaler.data_min_)
        vars = scale_factor @ covs @ scale_factor.T
        stds = np.diagonal(vars, axis1 = 1, axis2 = 2)
        stds = np.sqrt(stds)
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Simulation {sim_idx+1}', fontsize=16)
        
        for i in range(preds.shape[1]):
            plt.subplot(math.ceil(preds.shape[1]/2), 2, i+1)
            
            # Plot ground truth data
            plt.plot(noisy_targets[sim_idx, :, i], label='Noisy Data', color='green')
            plt.plot(noiseless_targets[sim_idx, :, i], label='Noiseless Data', color='black', linestyle='dashed')
            
            # Plot predictions with uncertainty
            time_steps = range(len(preds))
            plt.plot(preds[:, i], label=f'PNN {feature_names[i]}', color='blue')
            plt.fill_between(time_steps, 
                           preds[:, i] - 1.8*stds[:, i],
                           preds[:, i] + 1.8*stds[:, i],
                           color='blue', alpha=0.2, label=f'{feature_names[i]} Uncertainty')
            
            plt.title(f'{feature_names[i]} Trajectory')
            plt.xlabel('Time step')
            plt.ylabel(f'{feature_names[i]} value')
            plt.legend()
            
        plt.tight_layout()
        plt.show()
        
        # Plot constraint violation
        constraint = np.zeros((preds.shape[0], 1))
        constraint_true = np.zeros((preds.shape[0], 1))
        
        for i in range(preds.shape[0]):
            constraint[i] = A.cpu().numpy() @ noiseless_features[sim_idx, i, :] + B.cpu().numpy() @ preds[i, :] - b.cpu().numpy()
            constraint_true[i] = A.cpu().numpy() @ noiseless_features[sim_idx, i, :] + B.cpu().numpy() @ noiseless_targets[sim_idx, i, :] - b.cpu().numpy()
        
        plt.figure(figsize=(15, 6))
        plt.title(f'Constraint Violation - Simulation {sim_idx+1}')
        plt.plot(constraint, label='ECNN Model')
        plt.axhline(0, color='black', linestyle='dashed', label='Zero Violation')
        plt.plot(constraint_true, label='True Data', linestyle='dashed', color='green')
        plt.xlabel('Time step')
        plt.ylabel('Constraint Violation')
        plt.legend()
        plt.show()



    # Plot for all simulations
    plt.figure(figsize=(15, 10))

    # plot the error distribution
    ######### MC Dropout Plots ###########
    
    
    
    # for j in range(simulation_1.shape[1]):
        # simulation_1[:, j, :] = data_processor.target_scaler.inverse_transform(simulation_1[:, j, :])
        # np_simulation_1[:, j, :] = data_processor.target_scaler.inverse_transform(np_simulation_1[:, j, :])
    #     # Plot distribution of simulation_1 with overlay of np_simulation_1
    #     plt.figure(figsize=(15, 10))
    #     for i in range(simulation_1.shape[2]):  # For each feature
    #         plt.subplot(math.ceil(simulation_1.shape[2]/2), 2, i+1)
    #         # Create histogram with KDE for both projected and non-projected data
    #         sns.histplot(simulation_1[:, j, i], kde=True, color='blue', alpha=0.6, label='Projected')
    #         sns.histplot(np_simulation_1[:, j, i], kde=True, color='red', alpha=0.6, label='Non-Projected')
    #         plt.axvline(noiseless_targets[0, j, i], color='black', linestyle='dashed', label='Noiseless Actual')
    #         plt.title(f'Distribution of {feature_names[i]} at time {j}')
    #         plt.xlabel(f'{feature_names[i]} value')
    #         plt.ylabel('Frequency')
    #         plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    
    action_names = ['inlet temp', 'feed conc', 'coolant temp']
    visualizer = Visualizer()

    
    visualizer.plot_loss(history)
    
if __name__ == "__main__":
    main()