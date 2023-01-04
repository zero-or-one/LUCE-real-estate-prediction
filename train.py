# coding=utf-8
import numpy as np
import time
import os
import torch
import argparse
import torch.nn as nn
from models import *
from data import *
from utils import *
from logger import Logger
from config import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config" , type=str, default='default_configure')
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--visible_devices", type=str, default='0')       
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    torch.cuda.manual_seed(args.seed)
    #os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = eval(args.config)(device)
    device = torch.device(device)
    result_path = config.result_path
    logger = Logger(result_path, result_path + 'model_saved/', result_path + 'others/')
    logger.save_parameters(config)

    adj, features, labels, listprice, train_index, test_index = prepare_data(config)

    train_epoch = config.epoch
    seq_len = config.seq_len
    whole_house_size = features.shape[0]
    feature_size = features.shape[1]
    input_dim = features.shape[1]
    config.nfeat = input_dim

    # !!
    #adj = adj[:whole_house_size, :whole_house_size]

    train_index = range(8000)
    test_index = range(8000, 8596)

    features = torch.tensor(features).to(device).float()
    labels = torch.tensor(labels).to(device)
    listprice = torch.tensor(listprice).to(device)
    train_index = torch.LongTensor(train_index).to(device)
    test_index = torch.LongTensor(test_index).to(device)
    adj = torch.tensor(adj).to(device).float()
    # for now as adj is not prepared till end
    adj = adj.unsqueeze(0)
    print('features: ' + str(features.shape))
    print('labels: ' + str(labels.shape))
    print('listprice: ' + str(listprice.shape))
    print('train_index: ' + str(train_index.shape))
    print('test_index: ' + str(test_index.shape))

    model = eval(config.model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    loss_criterion = eval(config.loss)
    min_mae = 10000
    min_rmse = 10000
    max_pred_acc = 0
    min_train_loss = 10000
    w_str = ''
    # Tentatively, the training period of the model is the same every month
    for i in range(train_epoch):
        start_time = time.time()
        training_loss = []
        validation_losses = []
        model.train()
        optimizer.zero_grad()   # Gradient zeroing
        out_price = model(features, adj)  # forward() function in LSTMs, graph convolution operation
        print('out_price:' + str(out_price.shape))
        print('labels[train_index]:' + str(labels[train_index].shape))
        loss = loss_criterion(out_price[train_index, 0], labels[train_index, 0])  # loss calculation, pre and target
        loss.backward()     # backward propagation calculation
        optimizer.step()    # model parameter update
        training_loss.append(loss.detach().cpu().numpy())
        avg_training_loss = sum(training_loss) / len(training_loss)
        print("Epoch:{}  Training loss:{}".format(i, avg_training_loss))
        logger.log_training(i, avg_training_loss)

        # Evaluation of the trained model
        with torch.no_grad():
            model.eval()
            out_test_price = model(features)
            val_predict = out_test_price[test_index].detach().cpu().numpy()
            val_target = labels[test_index].cpu().numpy()
            val_listprice = listprice[test_index].cpu().numpy()
            mse, mae, rmse = score(val_predict, val_target)
            y_pre_error, pred_acc = pre_error(val_predict, val_target)
            if rmse < min_rmse:
                min_rmse = rmse
                min_mae = mae
                max_pred_acc = pred_acc
                output = val_predict
                logger.save_model(model, optimizer, i)
                w_str = price_str(val_predict, val_target, val_listprice)
        end_time = time.time()
        cost_time = end_time-start_time
        # print(w_str)
        print("Test MSE: {} MAE:{} RMSE: {} pre_error:{} cost_time:{}".format(mse,mae,rmse,y_pre_error,cost_time))
        logger.log_testing(i, mse, mae, rmse, y_pre_error, cost_time, w_str)

    print("MAE:{} RMSE: {} R2_score:{}\n".format(mae, rmse, max_pred_acc),flush=True)

