import torch
import torch.nn as nn
import optuna
from utils_new import *
from models.mcd_nn import MCD_NN
# from models.sdp_pnn import SDP_PNN
from models.Gaussian_HPINN import KKT_PPINN
from base import TrainingConfig, MLPConfig
from optuna.trial import TrialState
import multiprocessing


def objective(trial, training_config: TrainingConfig, model_config: MLPConfig, 
              X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor,
              y_test: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, 
              A: torch.Tensor, B: torch.Tensor, b: torch.Tensor, trainer_class: type,
              criterion: nn.Module, data_processor: DataProcessor):
    """
    Objective function for Optuna Study
    Args:
        trial (optuna.Trial): Optuna trial object
        config (MLPConfig): Configuration for MLP model
        X_train (torch.Tensor): Training data
        y_train (torch.Tensor): Training labels
        X_val (torch.Tensor): Validation data
        y_val (torch.Tensor): Validation labels
        trainer (ModelTrainer): Model trainer object
        criterion (nn.Module): Loss function
    """
    
    # Hyperparameters for training
    training_config.batch_size = trial.suggest_int("batch_size", 16, 900)
    training_config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    training_config.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    training_config.patience = trial.suggest_int("patience", 5, 20)
    training_config.delta = trial.suggest_float("delta", 1e-5, 1e-2, log=True)
    training_config.factor = trial.suggest_float("factor", 0.1, 0.5)
    
    # Model configuration
    model_config.hidden_dim = trial.suggest_int("hidden_dim", 32, 4096)
    model_config.num_layers = trial.suggest_int("num_layers", 1, 15)
    # model_config.activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "Tanh", "Softplus"])
    model_config.activation = trial.suggest_categorical("activation", ["ReLU"])
    device = model_config.device

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    A_orig = torch.tensor([0.0, 0.0], device=device)
    B_orig = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device) # Coeffs for V1 and V2
    b_orig_val = torch.tensor([100.0], device=device) # Example: total volume is 100

    # --- 2. Get scaling parameters from fitted scalers ---
    # For features (x)
    Mx = torch.from_numpy(data_processor.feature_scaler.data_min_).float().to(device)
    x_data_max = torch.from_numpy(data_processor.feature_scaler.data_max_).float().to(device)
    Sx_range = x_data_max - Mx
    Sx_range[Sx_range == 0] = 1.0 # Avoid issues if a feature is constant, though A_orig_j * 0 would handle it too

    # For targets (y)
    My = torch.from_numpy(data_processor.target_scaler.data_min_).float().to(device)
    y_data_max = torch.from_numpy(data_processor.target_scaler.data_max_).float().to(device)
    Sy_range = y_data_max - My
    Sy_range[Sy_range == 0] = 1.0 # Avoid issues if a target is constant

    # --- 3. Calculate A_model, B_model, b_model for scaled variables ---
    A_constr_model = A_orig * Sx_range  # Element-wise product
    B_constr_model = B_orig * Sy_range  # Element-wise product
    
    b_constr_model = b_orig_val - torch.dot(A_orig, Mx) - torch.dot(B_orig, My)

    epsilon = float(2)
    
    
    model = KKT_PPINN(
        config = model_config,
        input_dim = X_train.shape[1],
        output_dim = y_train.shape[1],
        A = A_constr_model,  # Use scaled A
        B = B_constr_model,  # Use scaled B
        b = b_constr_model,  # Use scaled b
        epsilon = epsilon,
        probability_level = 0.95
    )
    
    
    # model = MLP(
    #     config = model_config,
    #     input_dim = X_train.shape[1],
    #     output_dim = y_train.shape[1],
    #     num_samples = None
    #     )
    
    # Create a trainer instance from the trainer class
    model_trainer = trainer_class(model, training_config)
    model, history, avg_loss = model_trainer.train(X_train, y_train, X_test, y_test, X_val, y_val, criterion)
    
    return avg_loss


def run_optimization():
    training_config = TrainingConfig(
        batch_size=64,
        num_epochs=100,
        learning_rate=0.001,
        weight_decay=0.0001,
        factor=0.1,
        patience=10,
        delta=0.0001
    )
    model_config = MLPConfig(
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        activation='ReLU',
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    features_path = 'datasets/tank_system_features.csv'
    targets_path = 'datasets/tank_system_targets.csv'
    noiseless_path = 'datasets/tank_system_noiseless_features.csv'

    print(model_config.device)

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

    # Enforce mass balance: V1 + V2 = 1 (scaled value)
    A = torch.Tensor([0, 0])
    B = torch.Tensor([1, 0, 1, 0])
    b = torch.Tensor([1])

    trainer_class = ModelTrainer
    # criterion = nn.MSELoss()
    from mv_gaussian_nll import GaussianMVNLL
    criterion = GaussianMVNLL()

    # Create and run Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, training_config, model_config, X_train, y_train, X_test, y_test, X_val, y_val, A, B, b, trainer_class, criterion, data_processor), n_trials=100)
    
    # Print study results
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    # This is required for Windows to properly use multiprocessing
    multiprocessing.freeze_support()
    run_optimization()