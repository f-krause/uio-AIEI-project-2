import time
import numpy as np
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

        if self.config.mode.lower() == "prediction":
            if self.config.model.lower() == "lstm":
                self.model = LSTMPred(self.config)
            elif self.config.model.lower() == "rnn":
                self.model = RNNPred(self.config)
            self.criterion = torch.nn.MSELoss()  # mean-squared error for regression
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        elif self.config.mode.lower() == "classification":
            if self.config.model.lower() == "lstm":
                self.model = LSTMClass(self.config)
            elif self.config.model.lower() == "rnn":
                self.model = RNNClass(self.config)
            self.criterion = torch.nn.CrossEntropyLoss()  # cross entropy loss for classification
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr)
        else:
            raise NotImplementedError("mode must be 'prediction' or 'classification'")

    def train(self, x_train, y_train, x_val, y_val):
        # x_train: (H: num households, N: number of train samples, L: sequence length)
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

        print("  Needed  %1.2f minutes for training" % ((time.time() - start_time) / 60))

    def predict(self, x):
        return self.model(x.reshape(-1, x.shape[-1]))

    def plot_training_loss(self):
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
        if self.config.mode == "classification":
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
        else:
            print("Warning: no training accuracy for regression tasks!")

    def plot_confusion_matrix(self, x, y, return_matrix=False):
        if self.config.mode == "classification":
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
        else:
            print("Warning: no confusion matrix for regression tasks!")

    def evaluation_metrics(self, x, y):
        self.model.eval()
        y_pred = self.model(x.reshape(-1, x.shape[-1]))
        y = y.reshape(-1, 1)
        print("METRICS:")
        if self.config.mode.lower() == "prediction":
            print("  Test MSE:", self.criterion(y, y_pred).item())

        elif self.config.mode.lower() == "classification":
            y = y.long().squeeze()
            test_loss = self.criterion(y_pred, y).item()
            print("  Test cross entropy:", round(test_loss, 4))

            # Metrics
            y_pred = y_pred.argmax(dim=1)
            print(classification_report(y, y_pred, target_names=LABELS))

        # TODO add plotting of results
