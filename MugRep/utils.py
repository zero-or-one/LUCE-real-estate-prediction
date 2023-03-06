import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import numpy as np
import os


''' Seed '''
def seed_everything(seed=13):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


''' Metrics '''
def score(y_predict, y_target):
    y_predict = y_predict.reshape(1, -1)
    y_target = y_target.reshape(1, -1)
    mse = mean_squared_error(y_predict, y_target)
    mae = mean_absolute_error(y_predict, y_target)
    if torch.is_tensor(y_target):
        y_target, y_predict = y_target.numpy(), y_predict.numpy()
    mape = np.mean(np.abs((y_target - y_predict) / np.maximum(np.ones(len(y_target)), \
        np.abs(y_target))))*100
    return mse, mae, mape, np.sqrt(mse)
