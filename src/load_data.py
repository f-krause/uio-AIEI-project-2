import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.config import Config


def get_data(config: Config, return_index: bool = False):
    # x has the format (num households H, observations N, input dim L)
    if config.mode.lower() == "prediction":
        data = np.load("data/task1_data.npz")
    elif config.mode.lower() == "classification":
        data = np.load("data/task2_data.npz")
    else:
        raise NotImplementedError("mode must be 'prediction' or 'classification'")

    # move household dimension back for sampling
    x = data["x"].transpose(1, 0, 2)
    y = data["y"].transpose(1, 0)
    x, y = [torch.from_numpy(arr).float() for arr in [x, y]]

    # generate train/validation/test split indices with desired ratio
    full_ix = torch.arange(0, len(x))
    trval_ix, te_ix = train_test_split(full_ix, test_size=config.test_size, 
                                       shuffle=config.shuffle, 
                                       random_state=config.seed)
    tr_ix, val_ix = train_test_split(trval_ix, test_size=config.val_size,
                                     shuffle=config.shuffle,
                                     random_state=config.seed)

    # generate sets from indices
    x_train, x_val, x_test = x[tr_ix], x[val_ix], x[te_ix]
    y_train, y_val, y_test = y[tr_ix], y[val_ix], y[te_ix]

    # move household dimension to front for consistency
    x, x_train, x_val, x_test = [arr.permute((1, 0, 2)) for arr in [x, x_train, x_val, x_test]]
    y, y_train, y_val, y_test = [arr.permute((1, 0)) for arr in [y, y_train, y_val, y_test]]

    # print([arr.shape for arr in [x_train, x_val, x_test, y_train, y_val, y_test]])
    splits = x_train, x_val, x_test, y_train, y_val, y_test
    if return_index:
        return splits, x, y, (tr_ix, val_ix, te_ix)
    else:
        return splits

def load_daily_consumptions(
    path="./data/daily_consumptions.npz",
):
    daily_cons = np.load(path)["arr_0"]
    assert daily_cons.ndim == 2
    return daily_cons