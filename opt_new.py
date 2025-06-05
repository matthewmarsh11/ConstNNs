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
from mv_gaussian_nll import GaussianMVNLL
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def objective(trial, training_config: TrainingConfig, model_config: MLPConfig,
              X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor,
              y_test: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor,
              A_orig: torch.Tensor, B_orig: torch.Tensor, b_orig: torch.Tensor,
              x_scaler: MinMaxScaler, y_scaler: MinMaxScaler,
              trainer_class: type, criterion: nn.Module):
    """
    Objective function for Optuna Study
    """


    # Hyperparameters for training
    training_config.batch_size = trial.suggest_int("batch_size", 16, 128) # Adjusted range
    training_config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    training_config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) # Adjusted range
    # In ReduceLROnPlateau, 'patience' is an int.
    training_config.patience = trial.suggest_int("patience_scheduler", 5, 20) # For LR scheduler
    patience_early_stopping = trial.suggest_int("patience_early_stopping", 10, 30) # For early stopping
    training_config.delta = trial.suggest_float("delta_scheduler", 1e-5, 1e-2, log=True) # For LR scheduler (min_delta/eps)
    training_config.factor = trial.suggest_float("factor_scheduler", 0.1, 0.7) # For LR scheduler

    # Model configuration
    model_config.hidden_dim = trial.suggest_int("hidden_dim", 32, 512) # Adjusted range
    model_config.num_layers = trial.suggest_int("num_layers", 1, 5) # Adjusted range
    model_config.dropout = trial.suggest_float("dropout", 0.0, 0.5)
    # model_config.activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "Tanh", "Softplus"])
    model_config.activation = trial.suggest_categorical("activation", ["ReLU", "Tanh"]) # Simplified for now

    # KKT_PPINN specific hyperparameters
    epsilon = trial.suggest_float("epsilon", 0.1, 10.0, log=True)
    probability_level = trial.suggest_float("probability_level", 0.80, 0.99)

    device = model_config.device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    A_orig = A_orig.to(device)
    B_orig = B_orig.to(device)
    b_orig = b_orig.to(device)

    # --- Scale constraint matrices ---
    # Using scaler.scale_ (which is 1 / (data_max_ - data_min_))
    # And scaler.min_ (which is -X_min / (data_max_ - data_min_))
    # X_orig = (X_scaled - scaler.min_) / scaler.scale_
    # X_orig = X_scaled / scaler.scale_ + X_min_original
    
    x_scale_vals = torch.tensor(x_scaler.scale_, dtype=torch.float32, device=device)
    y_scale_vals = torch.tensor(y_scaler.scale_, dtype=torch.float32, device=device)
    
    # Avoid division by zero if a feature/target was constant (scale_ would be inf or large)
    # MinMaxScaler typically assigns scale_=1 for constant features if min==max,
    # but if min!=max and then becomes constant in a split, this could be an issue.
    # However, if a feature is constant, its coefficient in A_orig should ideally be 0.
    # For safety, replace potential zero or very small scales
    x_scale_vals[x_scale_vals < 1e-6] = 1.0
    y_scale_vals[y_scale_vals < 1e-6] = 1.0

    A_constr_model = A_orig / x_scale_vals.unsqueeze(0)
    B_constr_model = B_orig / y_scale_vals.unsqueeze(0)

    # Transform b: b_model = b_orig + A_orig @ X_min_orig + B_orig @ Y_min_orig
    # X_min_orig = x_scaler.data_min_
    # Y_min_orig = y_scaler.data_min_
    x_min_orig = torch.tensor(x_scaler.data_min_, dtype=torch.float32, device=device)
    y_min_orig = torch.tensor(y_scaler.data_min_, dtype=torch.float32, device=device)
    
    # Ensure correct dimensions for matmul if A_orig, B_orig are [1, N] and x_min_orig is [N]
    b_offset = torch.matmul(A_orig, x_min_orig.unsqueeze(-1)) + \
               torch.matmul(B_orig, y_min_orig.unsqueeze(-1))
    b_constr_model = b_orig - b_offset # The original formulation is Ax+By-b=0 -> A'x'+B'y' - b' = 0


    model = KKT_PPINN(
        config=model_config,
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        A=A_constr_model,
        B=B_constr_model,
        b=b_constr_model,
        epsilon=epsilon,
        probability_level=probability_level
    )
    model.to(device)

    # Create a trainer instance
    # The ModelTrainer needs to be adapted if its early stopping `patience`
    # is different from the scheduler's `patience`.
    # For now, assuming training_config.patience is for the scheduler.
    # We'll pass patience_early_stopping to the trainer if it supports it,
    # or the trainer needs to be modified.
    # For this example, the placeholder ModelTrainer uses training_config.patience for scheduler.
    # We'll rely on Optuna's pruning if training takes too long / loss is bad early.
    
    # Update training_config for this trial's patience values
    current_training_config = TrainingConfig(
        batch_size=training_config.batch_size,
        num_epochs=training_config.num_epochs, # Use base num_epochs
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        factor=training_config.factor,
        patience=training_config.patience, # This is for scheduler
        delta=training_config.delta,       # This is for scheduler (eps)
        device=device
    )
    # The ModelTrainer placeholder doesn't have explicit early stopping patience.
    # We can add it or rely on Optuna's pruner + fixed number of epochs.
    # For now, we use num_epochs from base config.
    
    model_trainer = trainer_class(model, current_training_config)
    
    # The train method in the placeholder returns best_val_loss as the third element
    _, _, val_loss = model_trainer.train(X_train, y_train, X_test, y_test, X_val, y_val, criterion)
    
    # Handle pruning
    trial.report(val_loss, step=model_trainer.config.num_epochs) # Report at the end
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_loss


def run_optimization():
    # Base Configurations
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


    print(f"Using device: {model_config.device}")

    # --- Data Loading and Preprocessing ---
    data = pd.read_csv('datasets/benchmark_CSTR.csv')
    x_df = data[['T x1','Ff_B x2','Ff_E x3']]
    y_df = data[['F_EB z1','F_B z2','F_E z3']]

    x_np = x_df.to_numpy()
    y_np = y_df.to_numpy()

    noise_level = 0.01
    y_noise = np.random.normal(0, 1, y_np.shape) * y_np * noise_level
    y_noisy_np = y_np + y_noise

    # Split data
    x_train_np, x_temp_np, y_train_np, y_temp_np = train_test_split(
        x_np, y_noisy_np, test_size=(1 - training_config.train_test_split), random_state=42
    )
    # test_val_split is the proportion of 'temp' data that becomes the test set
    x_val_np, x_test_np, y_val_np, y_test_np = train_test_split(
        x_temp_np, y_temp_np, test_size=training_config.test_val_split, random_state=42
    )

    # Scale data
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train_scaled_np = x_scaler.fit_transform(x_train_np)
    x_val_scaled_np = x_scaler.transform(x_val_np)
    x_test_scaled_np = x_scaler.transform(x_test_np)

    y_train_scaled_np = y_scaler.fit_transform(y_train_np)
    y_val_scaled_np = y_scaler.transform(y_val_np)
    y_test_scaled_np = y_scaler.transform(y_test_np)

    # Convert to tensors
    X_train = torch.tensor(x_train_scaled_np, dtype=torch.float32)
    X_val = torch.tensor(x_val_scaled_np, dtype=torch.float32)
    X_test = torch.tensor(x_test_scaled_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_scaled_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_scaled_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_scaled_np, dtype=torch.float32)

    # Original (unscaled) constraint definition: x2 - x3 - y2 + y3 = 0
    # A_orig * x_orig + B_orig * y_orig - b_orig = 0
    A_orig = torch.tensor([[0, 1, -1]], dtype=torch.float32)  # Coefficients for x: [T x1, Ff_B x2, Ff_E x3]
    B_orig = torch.tensor([[0, -1, 1]], dtype=torch.float32) # Coefficients for y: [F_EB z1, F_B z2, F_E z3]
    b_orig = torch.tensor([[0]], dtype=torch.float32)         # Right-hand side

    trainer_class = ModelTrainer # Make sure this is the correct ModelTrainer class
    criterion = GaussianMVNLL()  # Make sure this is the correct criterion

    # Create and run Optuna study
    # Added MedianPruner for early stopping of unpromising trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    
    study.optimize(lambda trial: objective(trial, training_config, model_config,
                                           X_train, y_train, X_test, y_test, X_val, y_val,
                                           A_orig, B_orig, b_orig, x_scaler, y_scaler,
                                           trainer_class, criterion),
                                           n_trials=100, timeout=None) # Adjust n_trials/timeout as needed

    # Print study results
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("\nBest trial:")
    trial = study.best_trial
    print("  Value (Best Validation Loss): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # You can also save results:
    # df_results = study.trials_dataframe()
    # df_results.to_csv("optuna_cstr_kkt_pinn_results.csv")


if __name__ == "__main__":
    # This might be required on Windows for multiprocessing with Optuna,
    # though Optuna's default in-memory storage is single-process unless you configure a distributed setup.
    # multiprocessing.freeze_support() # Usually needed if Optuna uses ProcessPoolExecutor, which might not be default.
    run_optimization()