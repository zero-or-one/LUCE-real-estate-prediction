import argparse
from utils import *
from models import *
from data import *
from logger import Logger
from config import *
import numpy as np
import time
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import math



# month * house => month * batch_num * batch_size
def make_index_batch(train_index, batch_size):
    month_len, house_size = train_index.shape
    train_index = torch.LongTensor(train_index)
    index_list = []
    for i in range(month_len):
        index_month_list = []
        for j in range(math.ceil(house_size / batch_size)):
            batch_start = j * batch_size
            batch_end = (j + 1) * batch_size
            # print('batch_start: ' + str(batch_start))
            # print('batch_end: ' + str(batch_end))
            if batch_end > house_size:
                batch_end = house_size
                batch_start = batch_end - batch_size
            index_month_list.append(train_index[i, batch_start:batch_end])
        index_month_list = torch.stack(index_month_list, 0)
        index_list.append(index_month_list)
    index_batch = torch.stack(index_list, 0).permute(1, 0, 2)
    return index_batch


def make_Y_from_index(labels, train_index):
    batch_num, month_len, batch_size = train_index.size()
    Y_train_batch = []
    for i in range(batch_num):
        Y_train_batch.append(labels[train_index[i]])
    Y_train_batch = torch.stack(Y_train_batch, 0)
    return Y_train_batch


def main(config):
    result_path = config.result_path
    logger = Logger(result_path, result_path + 'model_saved/', result_path + 'others/')
    logger.save_parameters(config)

    train_epoch = config.epoch
    seq_len = config.seq_len
    gc1_out_dim = config.gc1_outdim
    lstm_input_dim = config.lstm_inputdim
    meta_size = config.meta_size
    batch_size = config.batch_size
    update_len = config.update_len
    device = torch.device(config.device)

    adj, features, labels, train_index, test_index = prepare_data(config)
    whole_house_size = features.shape[0]
    feature_size = features.shape[1]
    hidden_dim = feature_size # hidden_dim is consistent with the dimension of embedding
    all_month = train_index.shape[0]+1
    house_size = int(whole_house_size/all_month)
    
    # data batch processing
    train_index_batch = make_index_batch(train_index, batch_size)
    test_index_batch = make_index_batch(test_index, batch_size)
    print("test_index_batch: " + str(test_index_batch.shape))
    # tensorization
    train_index_batch = train_index_batch.to(device)
    print("train_index_batch: " + str(train_index_batch.shape))
    test_index_batch = test_index_batch.to(device)
    adj = torch.tensor(adj).to(device)
    features = torch.tensor(features).to(device)
    labels = torch.tensor(labels).to(device)

    #  model training
    for cur_month in range(1, config.seq_len+1):
        # A month corresponds to a model model, and parameters are updated in the model of this month; cur_month represents the last month of the current training
         # r_gcnLSTMs starts training from the first month of data input each time, and gradually expands the model to the length of cur_month
         # According to update_len, when cur_month exceeds update_len, only update the parameters of [cur_month-update_len: cur_month] month each time
        if cur_month <= update_len:
            model_lstm_len = cur_month
            train_index_p = train_index_batch[:, 0: cur_month, :]#.unsqueeze(1)
            test_index_p = test_index_batch[:, 0: cur_month, :]#.unsqueeze(1)
        else:
            model_lstm_len = update_len
            train_index_p = train_index_batch[:, cur_month - model_lstm_len: cur_month, :]#.unsqueeze(1)
            test_index_p = test_index_batch[:, cur_month - model_lstm_len: cur_month, :]#.unsqueeze(1)


        print('train_index_p: ' + str(train_index_p.shape))
        #print('test_index_p: ' + str(test_index_p.shape))
        Y_train_batch = make_Y_from_index(labels, train_index_p).to(device)
        Y_test_batch = make_Y_from_index(labels, test_index_p).to(device)
        batch_num = train_index_batch.shape[0]
        print('Y_train_batch: ' + str(Y_train_batch.shape))
        #print('Y_test_batch: ' + str(Y_test_batch.shape))

        # Given parameters, so that the data dimension after GCN and lstm does not change
        model = r_gcn2lv_1LSTMs(gcn_input_dim=feature_size, gc1_out_dim=gc1_out_dim, lstm_input_dim=feature_size,
                                hidden_dim=hidden_dim, label_out_dim=1,  meta_size=config.meta_size, all_month=all_month,
                                month_len=model_lstm_len, layers=config.layers, dropout=config.dropout).to(device)
        #model = nn.DataParallel(model)
        
        # pre-training model parameter loading
        if cur_month == 1:
            if config.pretrained_path:
                static_model = torch.load(config.lstm_checkpoint_path)
                model_dict = model.state_dict()
                # all existing parameters are inherited, including LSTM and GCN of each month
                state_dict = {'glstm.0.'+str(k): v for k, v in static_model.items() if 'glstm.0.'+str(k) in model_dict.keys()}
                #print(state_dict.keys())
                model_dict.update(state_dict)
                state_dict = {k: v for k, v in static_model.items() if k in model_dict.keys()}
                #print(state_dict.keys())
                model_dict.update(state_dict)
                model.load_state_dict(model_dict)
                print('pretrained model loaded!')

        elif 1 < cur_month <= update_len:     # current month is within the update range
            #parameter inheritance
            old_model = torch.load(config.result_path + 'model_saved/' + 'time' + str(cur_month - 1) + '.pkl')
            model_dict = model.state_dict()
            # all existing parameters are inherited, including LSTM and GCN of each month
            state_dict = {k: v for k, v in old_model.items() if k in model_dict.keys()}
            #print(state_dict.keys())
            model_dict.update(state_dict)

            # The previous one month for the GCN model and the previous one for the next month.
            new_dict = {k.replace('glstm.' + str(int(cur_month - 2)), 'glstm.' + str(int(cur_month - 1))): v for k, v in
                        old_model.items() if 'glstm.' + str(int(cur_month - 2)) in k}
            model_dict.update(new_dict)
            model.load_state_dict(model_dict)
            print('pretrained model from previous time loaded!')
    
        elif cur_month > update_len:  #  current month is out of the update range
            # parameter inheritance
            old_model = torch.load(config.result_path + 'model_saved/' + 'time' + str(cur_month - 1) + '.pkl')
            model_dict = model.state_dict()
            # all existing parameters are inherited, including LSTM and GCN of each month
            state_dict = {k: v for k, v in old_model.items() if k in model_dict.keys()}
            #print(state_dict.keys())
            model_dict.update(state_dict)
            # GCN dislocation inheritance of each month, that is, glstm.1 in old_model should be glstm.0 in model
            gcn_dict = {k.replace('glstm.' + str(get_layer(k)), 'glstm.' + str(get_layer(k) - 1)): v
                          for k, v in old_model.items() if 'glstm.' in k and get_layer(k) > 0}
            #print(gcn_dict.keys())
            model_dict.update(gcn_dict)
            model.load_state_dict(model_dict)
            print('old model from previous time loaded!')
        

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        loss_criterion = eval(config.loss)
        
        # set the training cycle of each month's model to be the same
        for i in range(train_epoch):
            start_time = time.time()
            for b in range(batch_num):
                training_loss = 0
                mae_list = 0
                rmse_list = 0
                mse_list = 0
                model.train()
                optimizer.zero_grad()
                #print(train_index_p[b].shape, features.shape, adj.shape)
                new_embedding, out_price = model(adj, features, train_index_p[b])
                new_embedding = Variable(new_embedding.data, requires_grad=True)
                loss = loss_criterion(out_price, Y_train_batch[b])
                loss.backward()  
                optimizer.step()
                training_loss += loss.item()

                # evaluate the model on the test set after training
                with torch.no_grad():
                    model.eval()
                    _, out_test_price = model(adj, features, test_index_p[0])
                    val_target = Y_test_batch[0].cpu().numpy()
                    val_predict = out_test_price.detach().cpu().numpy()
                    mse, mae, rmse = score(val_predict, val_target)
                    mse_list += mse
                    mae_list += mae
                    rmse_list += rmse
                    # we can't use the pre_error function because the val_target is not a list
            end_time = time.time()
            cost_time = end_time - start_time
            
            avg_training_loss = training_loss / batch_num
            logger.log_training(i, avg_training_loss)
            mse = mse_list / batch_num
            mae = mae_list / batch_num
            rmse = rmse_list / batch_num
            logger.log_testing(i, mse, mae, rmse, cost_time)
        torch.save(model.state_dict(), config.result_path + 'model_saved/' + 'time' + str(cur_month) + '.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config" , type=str, default='PrelifelongConfig')
    parser.add_argument("--visible_devices", type=str, default='0')       
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    torch.cuda.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device is", device)
    config = eval(args.config)(device)
    main(config)