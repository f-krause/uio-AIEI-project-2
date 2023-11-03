import torch
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.config import Config
from src.models import LSTMPred, RNNPred, LSTMClass, RNNClass


class FederatedLearning:
    def __init__(self, config: Config):
        self.config = config
        self.train_losses, self.val_losses = None, None

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
        nr_samples = x_train.shape[1]
        nr_households = x_train.shape[0]
        if self.config.mode == "classification":
            y_train, y_val = y_train.long(), y_val.long()

        start_time = time.time()
        # Train the model
        for epoch in range(1, self.config.epochs+1):
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
            if epoch == 1 or epoch % self.config.eval_steps == 0:
                with torch.no_grad():
                    # y-pred from (H, N, L) -> (H * N, L)
                    y_pred = self.model(x_val.reshape(-1, x_val.shape[-1]))
                    y_val = y_val.reshape(-1, 1)
                    if self.config.mode == "classification":
                        y_val = y_val.squeeze()
                    val_loss = self.criterion(y_pred, y_val)
                    self.val_losses.append([epoch, val_loss.item()])

                print("Epoch: %d,  train loss: %1.5f, val loss: %1.5f" % (epoch, loss.item(), val_loss.item()))

        print("Needed  %1.2f minutes for training" % ((time.time() - start_time) / 60))

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

    def evaluate(self, x_test, y_test):
        self.model.eval()
        y_pred = self.model(x_test.reshape(-1, x_test.shape[-1]))
        y_test = y_test.reshape(-1, 1)
        if self.config.mode.lower() == "prediction":
            print("Test MSE:", self.criterion(y_test, y_pred).item())
        elif self.config.mode.lower() == "classification":
            y_test = y_test.long().squeeze()
            test_loss = self.criterion(y_pred, y_test).item()

            # Confusion matrix
            y_pred = y_pred.argmax(dim=1)
            labels = ['AC', 'Dish washer', 'Washing Machine', 'Dryer', 'Water heater', 'TV', 'Microwave', 'Kettle',
                      'Lighting', 'Refrigerator']
            print(confusion_matrix(y_test, y_pred, labels=labels))
            print("--------------------")

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=np.nan)
            recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=np.nan)
            f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=np.nan)

            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=np.nan)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=np.nan)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=np.nan)
            print("Test cross entropy:", round(test_loss, 4))
            print("Accuracy:          ", round(accuracy, 4))
            print("Precision (micro): ", round(precision_micro, 4))
            print("Recall (micro):    ", round(recall_micro, 4))
            print("F1 Score (micro):  ", round(f1_micro, 4))
            print("Precision (macro): ", round(precision_macro, 4))
            print("Recall (macro):    ", round(recall_macro, 4))
            print("F1 Score (macro):  ", round(f1_macro, 4))
