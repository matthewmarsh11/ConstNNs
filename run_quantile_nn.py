from utils_new import *
from models.mlp import *
from models.mcd_nn import *
from models.ec_nn import *
from models.constrained_quantile import *
from base import *

def main():
    
    training_config = TrainingConfig(
        batch_size=96,
        num_epochs=100,
        learning_rate=0.00294,
        weight_decay=0.0001,
        factor=0.14,
        patience=14,
        delta = 0.000042,
        train_test_split=0.6,
        test_val_split=0.8,
        # device = "mps" if torch.backends.mps.is_available() else "cpu",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )
    
    MLP_Config = MLPConfig(
        hidden_dim = 128,
        num_layers = 2,
        dropout = 0.2,
        activation = 'ReLU',
        # device = "mps" if torch.backends.mps.is_available() else "cpu",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )
    
    features_path = 'datasets/small_cstr_features.csv'
    targets_path = 'datasets/small_cstr_targets.csv'
    noiseless_path = 'datasets/small_cstr_noiseless_results.csv'
    
    features = pd.read_csv(features_path)
    features = features.iloc[:, :-1]
    targets = features.iloc[:, :4]
    features = features.iloc[:, 4:]
    noiseless_results = pd.read_csv(noiseless_path)
    noiseless_results = noiseless_results.iloc[:, :-1]
    noiseless_targets = noiseless_results.iloc[:, :4].to_numpy()
    noiseless_features = noiseless_results.iloc[:, 4:].to_numpy()
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
    A = torch.Tensor([0 , -1 , 0])
    B = torch.Tensor([1, 2, 1, 0])
    b = torch.Tensor([0])
    
    device = MLP_Config.device

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    A = A.to(device)
    B = B.to(device)
    b = b.to(device)
    
    quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    
    model = ConstrainedQuantileNN(
        config = MLP_Config,
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        A = A,
        B = B,
        b = b,
        quantiles=quantiles
    )
    from pinball_loss import PinballLoss
    criterion = PinballLoss(quantiles)
    
    trainer = ModelTrainer(model, training_config)
    model, history, avg_loss = trainer.train(X_train, y_train, X_test, y_test, X_val, y_val, criterion)
    X_tensor = X_tensor.reshape(num_simulations, -1, X_tensor.shape[1])
    noisy_targets = targets.to_numpy().reshape(num_simulations, -1, y_tensor.shape[1])
    noiseless_targets = noiseless_targets.reshape(num_simulations, -1, y_tensor.shape[1])
    features = features.to_numpy().reshape(num_simulations, -1, X_tensor.shape[2])
    noiseless_features = noiseless_features.reshape(num_simulations, -1, X_tensor.shape[2])
    
    model.eval()
    # model.enable_dropout()
    feature_names = ['ca', 'cb', 'cc', 'temp']
    simulations = {}
    simulations = {i: None for i in range(X_tensor.shape[0])}
    np_simulations = {}
    np_simulations = {i: None for i in range(X_tensor.shape[0])}
    for i in range (X_tensor.shape[0]):
        with torch.no_grad():
            prj_preds, nprj_preds = model(X_tensor[i, :, :].to(training_config.device))
            simulations[i] = prj_preds.cpu().numpy()
            np_simulations[i] = nprj_preds.cpu().numpy()
            


    # plot the error distribution
    ######## MC Dropout Plots ###########
    
    
    
    # for j in range(simulation_1.shape[1]):
    #     simulation_1[:, j, :] = data_processor.target_scaler.inverse_transform(simulation_1[:, j, :])
    #     np_simulation_1[:, j, :] = data_processor.target_scaler.inverse_transform(np_simulation_1[:, j, :])
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

    # Plot the trajectory of all simulations
    for sim_idx in range(len(simulations)):
        simulation = simulations[sim_idx]
        np_simulation = np_simulations[sim_idx]
        # shape [simulations, features, quantiles]
        for j in range(simulation.shape[2]):
            simulation[:, :, j] = data_processor.target_scaler.inverse_transform(simulation[:, :, j])
            np_simulation[:, :, j] = data_processor.target_scaler.inverse_transform(np_simulation[:, :, j])
            

        
        # Plot the trajectory of each simulation with quantiles
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Simulation {sim_idx+1}', fontsize=16)
        for i in range(simulation.shape[1]):  # for each feature
            plt.subplot(math.ceil(simulation.shape[1]/2), 2, i+1)
            
            # Plot ground truth data
            plt.plot(noisy_targets[sim_idx, :, i], label='Noisy Data', color='green')
            plt.plot(noiseless_targets[sim_idx, :, i], label='Noiseless Data', color='black', linestyle='dashed')
            
            # Plot projected predictions with quantiles
            time_steps = range(len(simulation))
            plt.plot(simulation[:, i, 1], label=feature_names[i], color='blue')  # median (0.5 quantile)
            plt.fill_between(time_steps, 
                   simulation[:, i, 0],  # lower quantile (0.05)
                   simulation[:, i, 2],  # upper quantile (0.95)
                   color='blue', alpha=0.2, label='Projected 90% CI')
            
            # Plot non-projected predictions with quantiles
            plt.plot(np_simulation[:, i, 1], label=f'Non-Projected {feature_names[i]}', 
                linestyle='dashed', color='red')  # median
            plt.fill_between(time_steps,
                   np_simulation[:, i, 0],  # lower quantile
                   np_simulation[:, i, 2],  # upper quantile
                   color='red', alpha=0.2, label='Non-Projected 90% CI')
            
            plt.title(f'Quantile Predictions for {feature_names[i]}')
            plt.xlabel('Time step')
            plt.ylabel(f'{feature_names[i]} value')
            plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot constraint violation for each simulation
        constraint = np.zeros((simulation.shape[0], 1))
        unprojected_constraint = np.zeros((simulation.shape[0], 1))
        constraint_true = np.zeros((simulation.shape[0], 1))
        
        for i in range(simulation.shape[0]):
            # Use median (0.5 quantile) predictions for constraint calculation
            constraint[i] = A.cpu().numpy() @ noiseless_features[sim_idx, i, :] + B.cpu().numpy() @ simulation[i, :, 1] - b.cpu().numpy()
            unprojected_constraint[i] = A.cpu().numpy() @ noiseless_features[sim_idx, i, :] + B.cpu().numpy() @ np_simulation[i, :, 1] - b.cpu().numpy()
            constraint_true[i] = A.cpu().numpy() @ noiseless_features[sim_idx, i, :] + B.cpu().numpy() @ noiseless_targets[sim_idx, i, :] - b.cpu().numpy()
        
        plt.figure(figsize=(15, 6))
        plt.title(f'Constraint Violation - Simulation {sim_idx+1}')
        plt.plot(constraint, label='Projected Model')
        plt.plot(unprojected_constraint, label='Non-Projected Model', linestyle='dashed')
        plt.axhline(0, color='black', linestyle='dashed', label='Zero Violation')
        plt.plot(constraint_true, label='True Data', linestyle='dashed', color='green')
        plt.xlabel('Time step')
        plt.ylabel('Constraint Violation')
        plt.legend()
        plt.show()

    # # Plot the loss history
    action_names = ['inlet temp', 'feed conc', 'coolant temp']
    visualizer = Visualizer()

    
    visualizer.plot_loss(history)
    
if __name__ == "__main__":
    main()