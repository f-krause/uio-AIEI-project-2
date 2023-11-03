import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.config import Config


def get_data(config: Config):
    # x has the format (num households H, observations N, input dim L)
    if config.mode.lower() == "prediction":
        data = np.load("data/task1_data.npz")
    elif config.mode.lower() == "classification":
        data = np.load("data/task2_data.npz")
    else:
        raise NotImplementedError("mode must be 'prediction' or 'classification'")
    x = data["x"].transpose(1, 0, 2)
    y = data["y"].transpose(1, 0)
    x, y = [torch.from_numpy(arr).float() for arr in [x, y]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config.test_size, shuffle=config.shuffle,
                                                        random_state=config.seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=config.val_size,
                                                      shuffle=config.shuffle, random_state=config.seed)

    x_train, x_val, x_test = [arr.permute((1, 0, 2)) for arr in [x_train, x_val, x_test]]
    y_train, y_val, y_test = [arr.permute((1, 0)) for arr in [y_train, y_val, y_test]]

    # print([arr.shape for arr in [x_train, x_val, x_test, y_train, y_val, y_test]])
    return x_train, x_val, x_test, y_train, y_val, y_test
