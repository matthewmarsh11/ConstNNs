import torch
import torch.nn as nn
import optuna
from utils_new import *
from models.mcd_nn import MCD_NN
from base import TrainingConfig, MLPConfig
from optuna.trial import TrialState
import multiprocessing


def objective(trial, training_config: TrainingConfig, model_config: MLPConfig, 
              X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor,
              y_test: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, 
              A: torch.Tensor, B: torch.Tensor, b: torch.Tensor, trainer_class: type,
              criterion: nn.Module):
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

    A = A.to(device)
    B = B.to(device)
    b = b.to(device)
    
    model = EC_NN(
        config = model_config,
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        A = A,
        B = B,
        b = b,
        dependent_ids=[2],
        num_samples = None
    )
    
    
    # Create a trainer instance from the trainer class
    model_trainer = trainer_class(model, training_config)
    model, history, avg_loss = model_trainer.train(X_train, y_train, X_test, y_test, X_val, y_val, criterion)
    
    return avg_loss


def run_optimization():
    training_config = TrainingConfig(
        batch_size=64,
        num_epochs=10,
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

    features_path = 'datasets/cstr_features.csv'
    targets_path = 'datasets/cstr_targets.csv'
    noiseless_path = 'datasets/cstr_noiseless_results.csv'

    print(model_config.device)

    features = pd.read_csv(features_path)
    features = features.iloc[:, :-1]
    targets = features.iloc[:, :4]
    features = features.iloc[:, 4:]
    noiseless_results = pd.read_csv(noiseless_path)
    noiseless_results = noiseless_results.iloc[:, :-1]
    noiseless_targets = noiseless_results.iloc[:, :4].to_numpy()
    noiseless_features = noiseless_results.iloc[:, 4:].to_numpy()
    num_simulations = 1000

    data_processor = DataProcessor(training_config, features, targets, num_simulations)
    # Prepare data
    (X_train, X_test, X_val, y_train, y_test, y_val, X_tensor, y_tensor) = data_processor.prepare_data(simulation_length=99)

    # Enforce mass balance: Cain = Ca + 2Cb + Cc
    A = torch.Tensor([0, -1, 0])
    B = torch.Tensor([1, 2, 1, 0])
    b = torch.Tensor([0])

    trainer_class = ModelTrainer
    # criterion = nn.MSELoss()
    from mv_gaussian_nll import GaussianMVNLL
    criterion = GaussianMVNLL()

    # Create and run Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, training_config, model_config, X_train, y_train, X_test, y_test, X_val, y_val, A, B, b, trainer_class, criterion), n_trials=100)
    
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