import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import os
from data import *


def score(y_predict, y_target):
    y_predict = y_predict.reshape(1, -1)
    y_target = y_target.reshape(1, -1)
    mse = mean_squared_error(y_predict, y_target)
    mae = mean_absolute_error(y_predict, y_target)
    return mse, mae, np.sqrt(mse)

# Error calculation
def pre_error(y_predict, y_target):
    y_predict = y_predict.reshape(1, -1)
    y_target = y_target.reshape(1, -1)
    y_minor = y_predict-y_target
    y_minor = np.fabs(y_minor)
    y_error = np.true_divide(y_minor, y_target)
    y_avg_error = np.mean(y_error)
    pred_acc = r2_score(y_target.reshape(-1), y_predict.reshape(-1))
    return y_avg_error, pred_acc

# Extract the current month from the parameter name
def get_layer(para_name):
    tmp = para_name.replace('glstm.','')
    nPos = tmp.index('.')
    num = int(tmp[:nPos])
    return num

# ID, listing price, transaction price, forecast price, listing forecast difference, transaction forecast difference
def price_str(val_predict, val_target, val_listprice):
    w_str = ''
    batch_size, label_size = val_predict.shape
    for k in range(batch_size):
        w_str += str(int(val_listprice[k,1]))+', '+str(val_listprice[k,0])+', '+str(val_target[k,0]) + \
                 ', ' + str(val_predict[k,0]) + ','  + str(abs(val_predict[k,0]-val_target[k,0])) + \
                 ', ' + str(abs(val_listprice[k,0]-val_target[k,0])) + '\n'
    return w_str

def prepare_data(config):
    if not os.path.exists(config.data_path + 'features.npy') \
            or not os.path.exists(config.data_path + 'labels.npy') \
            or not os.path.exists(config.data_path + 'train_index.npy') \
            or not os.path.exists(config.data_path + 'test_index.npy'):
        features, labels, train_index, test_index = \
            load_data(path=config.data_path, month_len=config.seq_len, house_size=config.house_size, dataset=config.dataset)
        print('Data is generated.')
    else:
        features = np.load(config.data_path + 'features.npy')
        labels = np.load(config.data_path + 'labels.npy')
        train_index = np.load(config.data_path + 'train_index.npy', allow_pickle=True)
        test_index = np.load(config.data_path + 'test_index.npy', allow_pickle=True)
        print('Data is loaded.')
    adj = [np.load(config.data_path + 'adjacency_house.npy'), np.load(config.data_path + 'adjacency_geo.npy')]
    # not working
    #adj = [np.load(config.data_path + 'adjacency_house.npz')['data'], np.load(config.data_path + 'adjacency_geo.npz')['data']]
    print('adj: ' + str(adj[0].shape))
    print('features: ' + str(features.shape))
    print('labels: ' + str(labels.shape))
    print('train_index: ' + str(train_index.shape))
    print('test_index: ' + str(test_index.shape))
    print('***********************************************************')
    return adj, features, labels, train_index, test_index