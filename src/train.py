from matplotlib import pyplot as plt
import torch
import time

from src.config import Config
from src.models import LSTMPred, RNNPred, LSTMClass, RNNClass


class FederatedLearning:
    def __init__(self, config: Config):
        print(config)
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

        start_time = time.time()
        # Train the model
        for epoch in range(self.config.epochs):
            self.optimizer.zero_grad()
            self.train_losses.append([epoch, 0.])

            for household_ix in range(nr_households):
                batch_ix = torch.randint(0, nr_samples, (self.config.batch_size,))
                xb = x_train[household_ix, batch_ix, ...]  # (B, L)
                yb = y_train[household_ix, batch_ix]  # (B,)

                outputs = self.model(xb)
                y = yb.unsqueeze(-1)  # (B, 1)
                loss = self.criterion(outputs, y)
                self.train_losses[-1][1] += loss.item()

                loss.backward()  # sum up gradients in parameters

            # average losses from different households
            self.train_losses[-1][1] = self.train_losses[-1][1] / nr_households
            for p in self.model.parameters():
                p.grad = p.grad / nr_households

            self.optimizer.step()
            if epoch % 100 == 0:
                with torch.no_grad():
                    # y_pred is of shape (H * N,)
                    y_pred = self.model(
                        x_val.reshape(-1, x_val.shape[-1])  # (H, N, L) -> (H * N, L)
                    )
                    y_gt = y_val.reshape(-1, 1)
                    val_loss = self.criterion(y_pred, y_gt)
                    self.val_losses.append([epoch, val_loss.item()])

                print("Epoch: %d,  train loss: %1.5f, val loss: %1.5f" % (epoch, loss.item(), val_loss.item()))

        print("Needed  %1.2f minutes for training" % ((time.time() - start_time) / 60))

    def plot_training_loss(self):
        plt.plot(*zip(*self.train_losses), label="training loss")
        plt.plot(*zip(*self.val_losses), label="validation loss")
        plt.legend()
        plt.show()

    def evaluate(self):
        pass
