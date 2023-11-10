import time
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.config import Config
from src.models import LSTMPred, RNNPred, LSTMClass, RNNClass


LABELS = ['AC', 'Dish washer', 'Washing Machine', 'Dryer', 'Water heater', 'TV', 'Microwave', 'Kettle',
          'Lighting', 'Refrigerator']


class FederatedLearning:
    def __init__(self, config: Config):
        self.config = config
        self.train_losses, self.val_losses = None, None
        self.train_accs, self.val_accs = None, None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        if self.config.mode.lower() == "prediction":
            if self.config.model.lower() == "lstm":
                self.model = LSTMPred(self.config)
            elif self.config.model.lower() == "rnn":
                self.model = RNNPred(self.config)
            self.criterion = torch.nn.MSELoss()  # mean-squared error for regression

        elif self.config.mode.lower() == "classification":
            if self.config.model.lower() == "lstm":
                self.model = LSTMClass(self.config)
            elif self.config.model.lower() == "rnn":
                self.model = RNNClass(self.config)
            self.criterion = torch.nn.CrossEntropyLoss()  # cross entropy loss for classification

        else:
            raise NotImplementedError("mode must be 'prediction' or 'classification'")

    def train(self, x_train, y_train, x_val, y_val):
        """Train the model on the given data.

        Args:
            x_train: training data of dimension (H: num households, N: number of train samples, L: sequence length)
            y_train: training labels of dimension (H: num households, N: number of train samples)
            x_val: validation data similar to x_train
            y_val: validation data similar to y_train
        """

        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        nr_samples = x_train.shape[1]
        nr_households = x_train.shape[0]
        if self.config.mode == "classification":
            y_train, y_val = y_train.long(), y_val.long()

        x_train_stacked = x_train.reshape(x_train.shape[0] * x_train.shape[1], x_train.shape[2])
        y_train_stacked = y_train.reshape(y_train.shape[0] * y_train.shape[1])

        # Train the model
        start_time = time.time()
        print("TRAIN:")
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            self.train_losses.append([epoch, 0.])

            for household_ix in range(nr_households):
                batch_ix = torch.randint(0, nr_samples, (self.config.batch_size,))
                x_batch = x_train[household_ix, batch_ix, ...]  # (B, L)
                y_batch = y_train[household_ix, batch_ix]  # (B,)

                outputs = self.model(x_batch)
                if self.config.mode == "prediction":
                    y_batch = y_batch.unsqueeze(-1)  # (B, 1)
                loss = self.criterion(outputs, y_batch)
                self.train_losses[-1][1] += loss.item()

                loss.backward()  # sum up gradients in parameters

            # average losses from different households
            self.train_losses[-1][1] = self.train_losses[-1][1] / nr_households
            for p in self.model.parameters():
                p.grad = p.grad / nr_households
            self.optimizer.step()

            # Validation step
            if epoch == 1 or epoch % self.config.eval_steps == 0:
                self.model.eval()
                with torch.no_grad():
                    # y-pred from (H, N, L) -> (H * N, L)
                    y_val_pred = self.model(x_val.reshape(-1, x_val.shape[-1]))
                    y_val = y_val.reshape(-1, 1)
                    if self.config.mode == "classification":
                        y_val = y_val.squeeze()
                        val_loss = self.criterion(y_val_pred, y_val)

                        val_acc = accuracy_score(y_val, y_val_pred.argmax(dim=1))
                        self.val_accs.append([epoch, val_acc])

                        y_train_pred = self.model(x_train_stacked.squeeze())
                        train_acc = accuracy_score(y_train_stacked, y_train_pred.argmax(dim=1))
                        self.train_accs.append([epoch, train_acc])
                        print("  Epoch: %d,  train loss: %1.4f, val loss: %1.4f, val acc: %1.4f" %
                              (epoch, loss.item(), val_loss.item(), val_acc))
                    else:
                        val_loss = self.criterion(y_val_pred, y_val)
                        print("  Epoch: %d,  train loss: %1.5f, val loss: %1.5f" %
                              (epoch, loss.item(), val_loss.item()))
                    self.val_losses.append([epoch, val_loss.item()])

        print("  Needed %1.2f minutes for training" % ((time.time() - start_time) / 60))

    def predict(self, x):
        """Get the prediction of the model for input x"""
        return self.model(x.reshape(-1, x.shape[-1]))

    def plot_training_loss(self):
        """Plot the training loss of training"""
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.plot(*zip(*self.train_losses), label="training loss")
        plt.plot(*zip(*self.val_losses), label="validation loss")
        plt.title("Train and Validation Loss while Training")
        plt.xlabel("epochs")
        if self.config.mode.lower() == "prediction":
            plt.ylabel("MSE")
        elif self.config.mode.lower() == "classification":
            plt.ylabel("cross entropy")
        plt.legend()
        plt.show()

    def plot_training_accuracy(self):
        """Plot the training and validation accuracy of training for classification tasks."""
        if self.config.mode == "prediction":
            return "Warning: no training accuracy for regression tasks!"

        plt.rcParams["figure.figsize"] = (10, 6)
        plt.plot(*zip(*self.train_accs), label="train accuracy")
        plt.plot(*zip(*self.val_accs), label="validation accuracy")
        plt.title("Train and Validation Accuracy while Training")
        plt.xlabel("epochs")
        if self.config.mode.lower() == "prediction":
            plt.ylabel("MSE")
        elif self.config.mode.lower() == "classification":
            plt.ylabel("cross entropy")
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, x, y, return_matrix=False):
        """Plot or return confusion matrix for classification task"""
        if self.config.mode == "prediction":
            return "Warning: no confusion matrix for regression tasks!"

        y = y.reshape(-1, 1).long().squeeze()
        y_pred = self.model(x.reshape(-1, x.shape[-1]))
        y_pred = y_pred.argmax(dim=1)
        cm = confusion_matrix(y, y_pred)

        if return_matrix:
            return cm
        else:
            plt.rcParams["figure.figsize"] = (8, 8)
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
            cm_display.plot()
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

    def evaluation_metrics(self, x, y):
        """Print loss and for classification a metrics overview """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x.reshape(-1, x.shape[-1]))
            y = y.reshape(-1, 1)
            print("METRICS:")
            if self.config.mode.lower() == "prediction":
                # create a mean model
                mean_model = torch.mean(y) * torch.ones_like(y)
                mean_mse = self.criterion(mean_model, y)
                print("  Mean model Test MSE: ", mean_mse.item())
                print("  Mean model Test RMSE:", mean_mse.sqrt().item())
                
                # print metrics of real model
                mse = self.criterion(y_pred, y)
                print("  Model Test MSE:      ", mse.item())
                print("  Model Test RMSE:     ", mse.sqrt().item())
                return y_pred

            elif self.config.mode.lower() == "classification":
                y = y.long().squeeze()
                test_loss = self.criterion(y_pred, y).item()
                print("  Test cross entropy:", round(test_loss, 4))

                # Metrics
                y_pred = y_pred.argmax(dim=1)
                print(classification_report(y, y_pred, target_names=LABELS))

    def plot_prediction_vs_label(self, x_full, y_full, train_idx, test_idx, val_idx=None, households: list = [0]):
        """
        Plot the prediction of the model given the previous week ground truth
        values as input.

        Args:
            x_full: Full input data
            y_full: Full target data
            train_idx: Indices of training data
            test_idx: Indices of test data
            val_idx: Indices of validation data (optional)
            households: List of indices of households to plot (0-49)
        """
        if self.config.mode == "classification":
            return "Warning: no prediction plot for regression tasks!"

        if not val_idx is None:
            torch.cat((train_idx, val_idx), dim=0)

        train_idx, test_idx = [arr.sort().values for arr in [train_idx, test_idx]]

        # Create a subplot for each household and plot both y_pred and y_pred_test
        plt.figure(figsize=(16, 4 * len(households)))
        for h in households:
            y_pred = self.predict(x_full[h][train_idx])
            y_pred_test = self.predict(x_full[h][test_idx])

            # Plot y_pred
            plt.subplot(len(households), 2, 2 * h + 1)
            plt.xlim(0, x_full.shape[1])
            plt.grid(True)
            plt.title("Daily Consumption Prediction for Household %d \n(train + val data)" % h)

            with torch.no_grad():
                plt.plot(train_idx + 6, y_full[h][train_idx], label="ground truth")
                plt.plot(train_idx + 6, y_pred, label="prediction")
            plt.legend()

            # Plot y_pred_test
            plt.subplot(len(households), 2, 2 * h + 2)
            plt.xlim(0, x_full.shape[1])
            plt.grid(True)
            plt.title("Daily Consumption Prediction for Household %d \n(test data)" % h)

            with torch.no_grad():
                plt.plot(test_idx + 6, y_full[h][test_idx], label="ground truth", marker='o', linestyle='-',
                         markersize=4)
                plt.plot(test_idx + 6, y_pred_test, label="prediction", marker='o', linestyle='-', markersize=4)
            plt.legend()

        plt.tight_layout()
