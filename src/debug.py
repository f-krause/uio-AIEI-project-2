from src.load_data import get_data
from src.train import FederatedLearning
from src.config import Config

STACKED = True  # aggregate all households to one


def main():
    kwargs = {
        # Main mode
        "mode": "classification",  # prediction or classification

        # Model config
        "model": "LSTM",  # LSTM or RNN
        # "hidden_dim": 32,
        "num_layers": 1,
        "dropout": 0.0,

        # Training config
        "epochs": 300,
        "lr": 0.01,
        "batch_size": 128
    }

    config = Config(**kwargs)
    print(config)
    x_train, x_val, x_test, y_train, y_val, y_test = get_data(config)

    if STACKED:
        x_train, x_val, x_test = [arr.reshape(1, arr.shape[0] * arr.shape[1], arr.shape[2]) for arr in
                                  [x_train, x_val, x_test]]
        y_train, y_val, y_test = [arr.reshape(1, arr.shape[0] * arr.shape[1]) for arr in [y_train, y_val, y_test]]

    fl = FederatedLearning(config)
    fl.train(x_train, y_train, x_val, y_val)
    # fl.plot_training_loss()
    fl.evaluate(x_test, y_test)
