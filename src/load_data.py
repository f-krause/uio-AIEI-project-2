import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.config import Config


def get_data(config: Config, return_index: bool = False):
    """Load data and split into train/validation/test sets

    Args:
        config: Config object with data loading parameters
        return_index: whether to return the indices of the train/validation/test sets
    """
    if config.mode.lower() == "prediction":
        data = np.load("data/task1_data.npz")
    elif config.mode.lower() == "classification":
        data = np.load("data/task2_data.npz")
    else:
        raise NotImplementedError("mode must be 'prediction' or 'classification'")

    # x has the format (num households H, observations N, input dim L)
    # move household dimension back for sampling
    x = data["x"].transpose(1, 0, 2)
    y = data["y"].transpose(1, 0)
    x, y = [torch.from_numpy(arr).float() for arr in [x, y]]

    # generate train/validation/test split indices with desired ratio
    full_ix = torch.arange(0, len(x))
    trval_ix, test_ix = train_test_split(full_ix, test_size=config.test_size,
                                         shuffle=config.shuffle,
                                         random_state=config.seed)
    train_ix, val_ix = train_test_split(trval_ix, test_size=config.val_size,
                                        shuffle=config.shuffle,
                                        random_state=config.seed)

    # generate sets from indices
    x_train, x_val, x_test = x[train_ix], x[val_ix], x[test_ix]
    y_train, y_val, y_test = y[train_ix], y[val_ix], y[test_ix]

    # move household dimension to front for consistency
    x, x_train, x_val, x_test = [arr.permute((1, 0, 2)) for arr in [x, x_train, x_val, x_test]]
    y, y_train, y_val, y_test = [arr.permute((1, 0)) for arr in [y, y_train, y_val, y_test]]

    # print([arr.shape for arr in [x_train, x_val, x_test, y_train, y_val, y_test]])
    splits = x_train, x_val, x_test, y_train, y_val, y_test
    indices = train_ix, val_ix, test_ix

    if return_index:
        return x, y, splits, indices
    else:
        return x, y, splits
