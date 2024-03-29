from src.load_data import get_data
from src.train import FederatedLearning
from src.config import Config

STACKED = True  # aggregate data of all households to one (i.e., traditional learning)


def main():
    # Specify parameters
    kwargs = {
        # Main mode
        "mode": "classification",  # prediction or classification

        # Model config
        "model": "RNN",  # LSTM or RNN
        # "hidden_dim": 32,
        "num_layers": 1,
        "dropout": 0.0,

        # Training config
        "epochs": 300,
        "lr": 0.01,
        "batch_size": 128
    }

    # Load as config
    config = Config(**kwargs)
    print(config)
    _, _, splits = get_data(config)
    x_train, x_val, x_test, y_train, y_val, y_test = splits

    if STACKED:
        # Aggregate data if specified
        x_train, x_val, x_test = [arr.reshape(1, arr.shape[0] * arr.shape[1], arr.shape[2]) for arr in
                                  [x_train, x_val, x_test]]
        y_train, y_val, y_test = [arr.reshape(1, arr.shape[0] * arr.shape[1]) for arr in [y_train, y_val, y_test]]

    # Train model
    fl = FederatedLearning(config)
    fl.train(x_train, y_train, x_val, y_val)

    ### ADAPT AS NEEDED BELOW ###

    # Plot image of train and validation loss
    # fl.plot_training_loss()

    # Plot image of train and validation accuracy
    # fl.plot_training_accuracy()

    # Print evaluation metrics
    fl.evaluation_metrics(x_test, y_test)

    # Plot confusion matrix
    print(fl.plot_confusion_matrix(x_test, y_test, return_matrix=True))
