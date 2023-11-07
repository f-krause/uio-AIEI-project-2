import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.config import Config
from src.models import LSTMPred, RNNPred, LSTMClass, RNNClass
from src.load_data import load_daily_consumptions


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

            # TODO add plotting of results
    
    def _get_daily_consumption_predictions(self):
        # load daily consumption data and put it into a tensor
        daily_cons = load_daily_consumptions()
        with torch.no_grad():
            daily_cons = torch.from_numpy(daily_cons).float()
        
        # get predictions from window values
        self.model.eval()
        with torch.no_grad():
            # create windows
            windows, ground_truths = [], []
            window_size = 7
            time_ix = torch.arange(window_size, daily_cons.shape[1] - 1)
            for i in time_ix:
                window = daily_cons[:, i-window_size:i]
                windows.append(window)
                ground_truths.append(daily_cons[:, i])
            windows = torch.stack(windows)
            windows = windows.transpose(1, 0) # move household dimension to front

            # predict samples for windows (for that, the first two dimensions of the 
            # window tensor need to be combined such that we get a matrix instead of a 
            # 3-tensor)
            predictions = self.model(windows.reshape(-1, window_size)).reshape(*windows.shape[:2])
            ground_truths = torch.stack(ground_truths)
            ground_truths = ground_truths.transpose(1, 0) # move household dimension to front
        
        return daily_cons, windows, predictions, ground_truths, time_ix

    def plot_daily_consumption_prediction(self, num_households = 5):
        """
        Plot the prediction of the model given the previous week ground truth
        values as input.

        Args:
            num_households: Number of households to plot
        """
        daily_cons, _, predictions, _, time_ix = self._get_daily_consumption_predictions()
        
        # plot predictions vs. groundtruths
        plt.figure(figsize=(6,3 * num_households))
        for h in range(num_households):
            plt.subplot(num_households, 1, h + 1)
            plt.xlim(0, daily_cons.shape[1])
            plt.grid(True)
            plt.title("daily consumption prediction for household %d" % h)

            # print ground truth and prediction
            plt.plot(daily_cons[h], label="ground truth")
            plt.plot(time_ix, predictions[h], label="prediction")

            plt.legend()
        plt.tight_layout()

    def plot_recursive_time_evolution(
        self,
        household_idx: int = 0,
        start_points: list[int] = None,
        num_steps: int = 20,
    ):
        """
        Create plots underneath each other that contain a recursive time
        evolution from given start points.

        Args:
            household_idx: Household for which the simulation should be ran.
            start_points: List of points to start the simulation from.
            num_steps: How many steps should be simulated.
        """

        if start_points is None:
            # use a default value, if there was none supplied
            start_points = [0, 100, 200]

        daily_cons, windows, _, _, time_ix = self._get_daily_consumption_predictions()

        # select the household
        daily_cons, windows = daily_cons[household_idx], windows[household_idx]

        # run simulation and plot results
        fig, axs = plt.subplots(nrows=len(start_points), figsize=(6, 2 * len(start_points)), sharex=True)
        for start_idx, ax in zip(start_points, axs):
            window = windows[start_idx]
            completion = [w.item() for w in window]
            with torch.no_grad():
                for _ in range(num_steps):
                    pred = self.model(torch.tensor(completion[-7:])[None,:])
                    completion.append(pred.item())
            ax.grid(True)
            ax.plot(torch.arange(len(window) - 1, len(completion)), completion[len(window) - 1:], label="prediction", c="k", marker="o")
            ax.plot(daily_cons[start_idx:start_idx+len(completion)], label="ground truth", ls=":", c="k")
            ax.axvline(len(window) - 1, c="k", ls="--", label="start of prediction")
            ax.set_title("start of prediction: day {}".format(time_ix[start_idx]))
        fig.legend(*ax.get_legend_handles_labels())
        fig.tight_layout()