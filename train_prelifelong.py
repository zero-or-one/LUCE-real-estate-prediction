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
    # !!!
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
    
    # remove the part included in pre-training
    #train_index = train_index[-6:]
    #test_index = test_index[-6:]
    '''
    data_len = features.shape[0]
    train_len = int(data_len * config.train_ratio)
    train_index = range(train_len)
    test_index = range(data_len-train_len, data_len)
    '''
    
    # data batch processing
    train_index_batch = make_index_batch(train_index, batch_size)
    print("train_index_batch: " + str(train_index_batch.shape))
    test_index_batch = make_index_batch(test_index, house_size)
    print("test_index_batch: " + str(test_index_batch.shape))
    # tensorization
    train_index_batch = train_index_batch.to(device)
    test_index_batch = test_index_batch.to(device)
    adj = torch.tensor(adj).to(device)
    features = torch.tensor(features).to(device)
    labels = torch.tensor(labels).to(device)

    #  model training
    for cur_month in range(1, 7):
        # A month corresponds to a model model, and parameters are updated in the model of this month; cur_month represents the last month of the current training
         # r_gcnLSTMs starts training from the first month of data input each time, and gradually expands the model to the length of cur_month
         # According to update_len, when cur_month exceeds update_len, only update the parameters of [cur_month-update_len: cur_month] month each time
        if cur_month <= update_len:
            model_lstm_len = cur_month
            train_index_p = train_index_batch[:, 0: cur_month, :]
            test_index_p = test_index_batch[:, 0: cur_month, :]
        else:
            model_lstm_len = update_len
            train_index_p = train_index_batch[:, cur_month - model_lstm_len: cur_month, :]
            test_index_p = test_index_batch[:, cur_month - model_lstm_len: cur_month, :]

        #print('train_index_p: ' + str(train_index_p.shape))
        #print('test_index_p: ' + str(test_index_p.shape))
        Y_train_batch = make_Y_from_index(labels, train_index_p).to(device)
        Y_test_batch = make_Y_from_index(labels, test_index_p).to(device)
        batch_num = train_index_batch.shape[0]
        #print('Y_train_batch: ' + str(Y_train_batch.shape))
        #print('Y_test_batch: ' + str(Y_test_batch.shape))

        # Given parameters, so that the data dimension after GCN and lstm does not change
        model = r_gcn2lv_1LSTMs(gcn_input_dim=feature_size, gc1_out_dim=gc1_out_dim, lstm_input_dim=feature_size,
                                hidden_dim=hidden_dim, label_out_dim=1,  meta_size=meta_size, all_month=all_month,
                                month_len=model_lstm_len, layers=args['layers'], dropout=args['dropout']).to(device)
        
        # Let's ignore this for now
        '''
        # pre-training model parameter loading
        if cur_month == 1:
            static_model = torch.load('model_saved_staticlstm/static.pkl')
            model_dict = model.state_dict()
            # 已有参数全部继承，包括LSTM和各月GCN all existing parameters are inherited, including LSTM and GCN of each month
            state_dict = {'glstm.0.'+str(k): v for k, v in static_model.items() if 'glstm.0.'+str(k) in model_dict.keys()}
            print(state_dict.keys())
            model_dict.update(state_dict)
            state_dict = {k: v for k, v in static_model.items() if k in model_dict.keys()}
            print(state_dict.keys())
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        elif 1 < cur_month <= update_len:     # current month is within the update range
            #parameter inheritance
            old_model = torch.load(model_file_path + 'month' + str(cur_month - 1) + '.pkl')
            model_dict = model.state_dict()
            # all existing parameters are inherited, including LSTM and GCN of each month
            state_dict = {k: v for k, v in old_model.items() if k in model_dict.keys()}
            print(state_dict.keys())
            model_dict.update(state_dict)

            # The previous one month for the GCN model and the previous one for the next month.
            new_dict = {k.replace('glstm.' + str(int(cur_month - 2)), 'glstm.' + str(int(cur_month - 1))): v for k, v in
                        old_model.items() if 'glstm.' + str(int(cur_month - 2)) in k}
            model_dict.update(new_dict)
            model.load_state_dict(model_dict)
    
        elif cur_month > update_len:  #  current month is out of the update range
            # parameter inheritance
            old_model = torch.load(model_file_path + 'month' + str(cur_month - 1) + '.pkl')
            model_dict = model.state_dict()
            # all existing parameters are inherited, including LSTM and GCN of each month
            state_dict = {k: v for k, v in old_model.items() if k in model_dict.keys()}
            print(state_dict.keys())
            model_dict.update(state_dict)
            # GCN dislocation inheritance of each month, that is, glstm.1 in old_model should be glstm.0 in model
            gcn_dict = {k.replace('glstm.' + str(get_layer(k)), 'glstm.' + str(get_layer(k) - 1)): v
                          for k, v in old_model.items() if 'glstm.' in k and get_layer(k) > 0}
            print(gcn_dict.keys())
            model_dict.update(gcn_dict)
            model.load_state_dict(model_dict)
        '''

        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        loss_criterion = nn.MSELoss()
        min_rmse = 10000
        
        # set the training cycle of each month's model to be the same
        for i in range(train_epoch*model_lstm_len):
            for b in range(batch_num):
                start_time = time.time()
                training_loss = []
                validation_losses = []
                model.train()
                optimizer.zero_grad()  # 梯度置零 gradient reset to zero
                new_embedding, out_price = model(adj, features, train_index_p[b])
                # new_embedding = Variable(new_embedding.data, requires_grad=True)
                # features = new_embedding.to(device)  # features is a global variable, inherited every time
                # The input is the index of multiple months
                #print('out_price: ' + str(out_price.shape))
                #print('Y_train_batch[b]: ' + str(Y_train_batch[b][cur_month-1:cur_month].shape))
                #with torch.no_grad():
                  # weights = np.tanh(np.arange(1,Y_train_batch[b].shape[0]+1) * (np.e / Y_train_batch[b].shape[0]))
                  #weights = np.arange(1,Y_train_batch[b].shape[0]+1)* (1 / Y_train_batch[b].shape[0])
                  #weights = torch.tensor(weights, dtype=torch.float32, device=device)
                # print(weights)
                #loss = (out_price-Y_train_batch[b])**2
                # T = T.view(model_lstm_len,batch_size)
                for k in range(weights.shape[0]):
                  loss[k] = loss[k]*weights[k]
                # print('loss shape', loss.shape)
                # exit()
                # loss = ((out_price-Y_train_batch[b])**2).permute(1,0,2).squeeze()*weights.permute(1,0)  # loss计算，pre与target
                # T = (out_price-Y_train_batch[b])**2
                # print('T:')
                # print(T.shape)
                # print(loss.shape)
                loss = loss.mean()
                loss.backward()  # loss backward propagation
                optimizer.step()  # model parameter update
                training_loss.append(loss.detach().cpu().numpy())
                avg_training_loss = sum(training_loss) / len(training_loss)
                print("Month:{}  Epoch:{}  Training loss:{}".format(cur_month, i, avg_training_loss))
                with open(result_file_path + 'month' + str(cur_month) + '_loss_error.txt', 'a+') as f:
                    f.write("Month:{}  Epoch:{}  Training loss:{}\n".format(cur_month, i, avg_training_loss))
                with open(other_file_path + 'train_loss.txt', 'a+') as f:
                    f.write("{}\n".format(avg_training_loss))

                # Evaluate the trained model on the test set
                with torch.no_grad():
                    model.eval()
                    _, out_test_price = model(adj, features, test_index_p[0])

                    val_target = Y_test_batch[0].cpu().numpy()
                    val_predict = out_test_price.detach().cpu().numpy()
                    '''
                    print('test_index_p[0][-1:]: '+str(test_index_p[0][-1:].shape))
                    print('val_predict: '+str(val_predict.shape))
                    print('val_target: '+str(val_target.shape))
                    '''
                    # print('val_target: '+str(val_target.shape))
                    mse, mae, rmse = score(val_predict, val_target)
                    y_pre_error = pre_error(val_predict, val_target)
                    if rmse < min_rmse:
                        min_rmse = rmse
                        output = val_predict
                        torch.save(model.state_dict(), model_file_path + 'month' + str(cur_month) + '.pkl')
                        # features = new_embedding.to(device)
                end_time = time.time()
                cost_time = end_time - start_time
                self.log_testing(cur_month, mse, mae, rmse, y_pre_error, cost_time)

        self.logger.log_monthly_price(w_str, cur_month)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config" , type=str, default='PrelifelongConfig')
    parser.add_argument("--visible_devices", type=str, default='0')       
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    torch.cuda.manual_seed(args.seed)
    #os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    config = eval(args.config)(device)
    main(config)