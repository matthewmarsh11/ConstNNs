import argparse
import optuna
import torch
import numpy as np
from utils_new import *
from models.Gaussian_HPINN import KKT_PPINN
from models.mlp import MLP
from models.ec_nn import EC_NN
from mv_gaussian_nll import GaussianMVNLL
from batch_data_loader import load_batch_data
from base import *

np.random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data once globally
data = load_batch_data()
criterion = GaussianMVNLL()


def objective(trial, model_name: str):
    # --- Sample hyperparameters ---
    hidden_dim = trial.suggest_int("hidden_dim", 64, 1024)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # --- Rebuild model config ---
    model_config = MLPConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        activation='ReLU',
        device=device
    )

    training_config = data["training_config"]
    training_config.learning_rate = learning_rate
    training_config.weight_decay = weight_decay

    # --- Instantiate model ---
    if model_name == "kkt_ppinn":
        model = KKT_PPINN(
            config=model_config,
            input_dim=data["X_tensor"].shape[1],
            output_dim=data["y_tensor"].shape[1],
            A=data["scaled_A"],
            B=data["scaled_B"],
            b=data["scaled_b"],
            epsilon=0.1,
            probability_level=0.95,
        )
        X_train, X_test, X_val, y_train, y_test, y_val = data["X_train"], data["X_test"], data["X_val"], data["y_train"], data["y_test"], data["y_val"]
    elif model_name == "mlp":
        model = MLP(
            config=model_config,
            input_dim=data["X_tensor_unconst"].shape[1],
            output_dim=data["y_tensor"].shape[1],
        )
        X_train, X_test, X_val, y_train, y_test, y_val = data["X_train_unconst"], data["X_test_unconst"], data["X_val_unconst"], data["y_train"], data["y_test"], data["y_val"]
    elif model_name == "ec_nn":
        model = EC_NN(
            config=model_config,
            input_dim=data["X_tensor"].shape[1],
            output_dim=data["y_tensor"].shape[1],
            A=data["scaled_A"],
            B=data["scaled_B"],
            b=data["scaled_b"],
            dependent_ids=[0],
        )
        X_train, X_test, X_val, y_train, y_test, y_val = data["X_train"], data["X_test"], data["X_val"], data["y_train"], data["y_test"], data["y_val"]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # --- Training ---
    trainer = ModelTrainer(model, training_config)
    model, history, avg_val_loss = trainer.train(X_train, y_train, X_test, y_test, X_val, y_val, criterion)

    # --- Save best model ---
    trial.set_user_attr("model", model)
    trial.set_user_attr("val_loss", avg_val_loss)

    return avg_val_loss


def save_best_model(study, model_name):
    best_model = study.best_trial.user_attrs["model"]
    saver = ModelSaver()
    saver.save_full_model(best_model, f"models/optuna_{model_name}_best.pt")
    print(f"Best {model_name} saved with val_loss={study.best_value:.4f}")


def run_optimisation(model_name: str, n_trials: int = 30):
    study = optuna.create_study(direction="minimize", study_name=f"opt_{model_name}")
    study.optimize(lambda trial: objective(trial, model_name), n_trials=n_trials)

    save_best_model(study, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="kkt_ppinn", choices=["kkt_ppinn", "mlp", "ec_nn"],
                        help="Which model to optimise.")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials.")
    args = parser.parse_args()

    run_optimisation(args.model, args.trials)