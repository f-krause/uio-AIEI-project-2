from load_data import get_data
from train import FederatedLearning
from config import Config


def main():
    kwargs = {
        # Main mode
        "mode": "prediction",  # prediction or classification

        # Model config
        "model": "LSTM",  # LSTM or RNN
        "input_dim": 1,
        "hidden_dim": 32,
        "output_dim": 1,
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
    fl = FederatedLearning(config)
    fl.train(x_train, y_train, x_val, y_val)
    fl.plot_training_loss()
    # fl.evaluate()  # TODO


if __name__ == "__main__":
    main()
