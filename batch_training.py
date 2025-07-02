from utils_new import *
from models.Gaussian_HPINN import KKT_PPINN
from models.mlp import MLP
from models.ec_nn import EC_NN
from base import *
import torch
from mv_gaussian_nll import GaussianMVNLL
from batch_data_loader import load_batch_data

np.random.seed(42)
torch.manual_seed(42)

kkt_save = ModelSaver()
mlp_save = ModelSaver()
ec_save = ModelSaver()


def main():
    data = load_batch_data()

    training_config = data["training_config"]
    model_config = data["model_config"]

    X_tensor = data["X_tensor"]
    y_tensor = data["y_tensor"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    X_val = data["X_val"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    y_val = data["y_val"]

    # X_train_unconst = data["X_train_unconst"]
    # X_test_unconst = data["X_test_unconst"]
    # X_val_unconst = data["X_val_unconst"]
    # X_tensor_unconst = data["X_tensor_unconst"]

    scaled_A = data["scaled_A"]
    scaled_B = data["scaled_B"]
    scaled_b = data["scaled_b"]

    epsilon = np.sqrt(0.05)

    kkt_ppinn = KKT_PPINN(
        config=model_config,
        input_dim=X_tensor.shape[1],
        output_dim=y_tensor.shape[1],
        A=scaled_A,
        B=scaled_B,
        b=scaled_b,
        epsilon=epsilon,
        probability_level=0.95,
    )

    mlp = MLP(
        config=model_config,
        input_dim=X_tensor.shape[1],
        output_dim=y_tensor.shape[1]
    )

    ec_nn = EC_NN(
        config=model_config,
        input_dim=X_tensor.shape[1],
        output_dim=y_tensor.shape[1],
        A=scaled_A,
        B=scaled_B,
        b=scaled_b,
        dependent_ids=[0],
    )

    pinn = MLP(
        config=model_config,
        input_dim=X_tensor.shape[1],
        output_dim=y_tensor.shape[1]
    )

    criterion = GaussianMVNLL()

    kkt_trainer = ModelTrainer(kkt_ppinn, training_config)
    mlp_trainer = ModelTrainer(mlp, training_config)
    ec_trainer = ModelTrainer(ec_nn, training_config)
    pinn_trainer = ModelTrainer(pinn, training_config)

    kkt_ppinn, kkt_history, kkt_avg_loss = kkt_trainer.train(X_train, y_train, X_test, y_test, X_val, y_val, criterion)
    mlp, mlp_history, mlp_avg_loss = mlp_trainer.train(X_train, y_train, X_test, y_test, X_val, y_val, criterion)
    ec_nn, ec_history, ec_avg_loss = ec_trainer.train(X_train, y_train, X_test, y_test, X_val, y_val, criterion)
    # pinn, pinn_history, pinn_avg_loss = pinn_trainer.train(X_train, y_train, X_test, y_test, X_val, y_val, criterion)

    kkt_save.save_full_model(kkt_ppinn, 'models/batch_kkt_ppinn')
    mlp_save.save_full_model(mlp, 'models/batch_mlp')
    ec_save.save_full_model(ec_nn, 'models/batch_ec_nn')
    # pinn_trainer.save_full_model('models/batch_pinn')


if __name__ == "__main__":
    main()