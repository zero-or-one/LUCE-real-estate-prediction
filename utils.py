import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np

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
    y_target, y_predict = y_target.numpy(), y_predict.numpy()
    mape = np.mean(np.abs((y_target - y_predict) / np.maximum(np.ones(len(y_target)), \
        np.abs(y_target))))*100
    return mse, mae, mape, np.sqrt(mse)

def pre_error(y_predict, y_target):
    y_predict = y_predict.reshape(1, -1)
    y_target = y_target.reshape(1, -1)
    y_minor = y_predict-y_target
    y_minor = np.fabs(y_minor)
    y_error = np.true_divide(y_minor, y_target)
    y_avg_error = np.mean(y_error)
    pred_acc = r2_score(y_target.reshape(-1), y_predict.reshape(-1))
    return y_avg_error, pred_acc