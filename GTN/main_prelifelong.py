import torch
import numpy as np
import torch.nn as nn
from model_gtn import GTN
from model_fastgtn import FastGTNs
from model_luce import LUCE
import pickle
import argparse
from torch_geometric.utils import add_self_loops
from sklearn.metrics import f1_score as sk_f1_score
from utils import init_seed, _norm
import copy
import pandas as pd
from sklearn.externals import joblib 


if __name__ == '__main__':
    init_seed(seed=777)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LUCE',
                        help='Model')
    parser.add_argument('--dataset', type=str,
                        help='Dataset', default='REALESTATE')
    parser.add_argument('--epoch', type=int, default=10000,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='hidden dimensions')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=int, default=0.1, help='learning rate decay')
    parser.add_argument('--lr_decay_step', type=int, default=1000, help='learning rate decay step')
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of GT/FastGT layers')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of runs')
    parser.add_argument("--channel_agg", type=str, default='concat')
    parser.add_argument("--remove_self_loops", action='store_true', help="remove_self_loops")
    # Configurations for FastGTNs
    parser.add_argument("--non_local", action='store_true', help="use non local operations")
    parser.add_argument("--non_local_weight", type=float, default=0, help="weight initialization for non local operations")
    parser.add_argument("--beta", type=float, default=0, help="beta (Identity matrix)")
    parser.add_argument('--K', type=int, default=1,
                        help='number of non-local negibors')
    parser.add_argument("--pre_train", action='store_true', help="pre-training FastGT layers")
    parser.add_argument('--num_FastGTN_layers', type=int, default=1,
                        help='number of FastGTN layers')
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    print(args)
    device = args.device

    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers

    A = []
    adj_matrix = np.load('dataset/{}.npy'.format("adjacency_house"))
    #for i in range(adj_matrix.shape[0]):
    edge_index = torch.from_numpy(np.vstack(adj_matrix.nonzero())).to(torch.long).to(device)
    # add target node
    edge_value = torch.from_numpy(adj_matrix[adj_matrix.nonzero()]).to(torch.float).to(device)
    #print(edge_index.shape, edge_value.shape)
    A.append((edge_index, edge_value))
    num_nodes = adj_matrix.shape[0]
    #exit()
    args.num_nodes = num_nodes
    # add self-loops and normalize if needed
    if args.model == 'FastGTN' and args.dataset != 'AIRPORT':
        edge_index, edge_value = add_self_loops(edge_index, edge_attr=edge_value, fill_value=1e-20, num_nodes=num_nodes)
        deg_inv_sqrt, deg_row, deg_col = _norm(edge_index.detach(), num_nodes, edge_value.detach())
        edge_value = deg_inv_sqrt[deg_row] * edge_value

    seq_len = 5*12
    update_len = 2*12
    house_size = 217

    node_features = pd.read_csv('dataset/{}.csv'.format('processed_data'))
    # initialize a model
    if args.model == 'GTN' or args.model == 'LUCE':
        model = GTN(num_edge=len(A),
                            num_channels=num_channels,
                            w_in = node_features.shape[1]-3,
                            w_out = node_dim,
                            num_layers=num_layers,
                            num_nodes=num_nodes,
                            args=args)        
    elif args.model == 'FastGTN':
        if args.pre_train and l == 1:
            pre_trained_fastGTNs = []
            for layer in range(args.num_FastGTN_layers):
                pre_trained_fastGTNs.append(copy.deepcopy(model.fastGTNs[layer].layers))
        while len(A) > num_edge_type:
            del A[-1]
        model = FastGTNs(num_edge_type=len(A),
                        w_in = node_features.shape[1],
                        num_nodes = args.batch_size,
                        args = args)
        if args.pre_train and l > 0:
            for layer in range(args.num_FastGTN_layers):
                model.fastGTNs[layer].layers = pre_trained_fastGTNs[layer]


    for cur_month in range(1, seq_len+1):
        # A month corresponds to a model model, and parameters are updated in the model of this month; cur_month represents the last month of the current training
         # r_gcnLSTMs starts training from the first month of data input each time, and gradually expands the model to the length of cur_month
         # According to update_len, when cur_month exceeds update_len, only update the parameters of [cur_month-update_len: cur_month] month each time
        node_features = pd.read_csv('dataset/{}.csv'.format('processed_data'))
        if cur_month <= update_len:
            model_lstm_len = cur_month
            node_features = node_features[node_features.month_year <= cur_month]
        else:
            model_lstm_len = update_len
            node_features = node_features[(node_features.month_year <= cur_month) & (node_features.month_year > cur_month-update_len)]
        node_features = node_features.values
        labels = np.expand_dims(node_features[:, -3], 1)
        node_features = node_features[:, :-3] + node_features[:, -1:]
        # split train/valid/test by 0.8/0.1/0.1 using indices
        nids = np.arange(0, node_features.shape[0])
        #np.random.shuffle(nids)
        # split nids to 80% train, 10% valid, 10% test
        nids = np.split(nids, [int(0.9*len(nids))])

        train_target = torch.from_numpy(labels[nids[0]]).type(torch.FloatTensor).to(device)
        valid_target = torch.from_numpy(labels[nids[1]]).type(torch.FloatTensor).to(device)

        num_edge_type = len(A)
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor).to(device)
        train_node_features = node_features[nids[0]]
        valid_node_features = node_features[nids[1]]

        
        # pre-training model parameter loading
        if cur_month == 1:
            if args.pretrained_path:
                static_model = torch.load(args.pretrained_path)
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
        

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        gamma = args.lr_decay
        step_size = args.lr_decay_step
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        model.to(device)
        #model = nn.DataParallel(model)
        loss = nn.L1Loss()
        Ws = []
        scaler = joblib.load(config.data_path + 'scaler.pkl')
        for epoch in range(epochs):
            # print('Epoch ',i)
            avg_train_loss = 0
            avg_valid_loss = 0
            avg_test_loss = 0
            avg_train_mse_error = 0
            avg_valid_mse_error = 0
            avg_test_mse_error = 0
            model.train()
            for batch in range(0, len(train_node_features), args.batch_size):
                batch_size = min(args.batch_size, len(train_node_features)-batch)
                num_batches = len(train_node_features)//batch_size
                optimizer.zero_grad()
                # take a batch of adjecency matrix
                a = []
                for i in range(len(A)):
                    a.append((A[i][0][:, batch:batch+batch_size], A[i][1][batch:batch+batch_size]))
                num_nodes = a[0][0].shape[1]
                #print(len(a), a[0][0].shape, a[0][1].shape)
                if args.model == 'FastGTN':
                    loss,train_mse,y_train,W = model(a, train_node_features[batch:batch+batch_size], train_target[batch:batch+batch_size], epoch=i)
                else:
                    loss,train_mse,y_train,W = model(a, train_node_features[batch:batch+batch_size], train_target[batch:batch+batch_size])
                loss.backward()
                optimizer.step()
                avg_train_loss += loss.detach().cpu().numpy() / num_batches
                avg_train_mse_error += train_mse / num_batches
            print('Epoch: {}\n Train - Loss: {}\n Train - MSE: {}\n'.format(epoch, avg_train_loss, avg_train_mse_error))
            scheduler.step()
            # validation
            model.eval()
            val_pred, val_tar = None, None
            for batch in range(0, len(valid_node_features), args.batch_size):
                batch_size = min(args.batch_size, len(valid_node_features)-batch)
                num_batches = len(valid_node_features)//batch_size
                # take a batch of adjecency matrix
                a = []
                for i in range(len(A)):
                    a.append((A[i][0][:, batch:batch+batch_size], A[i][1][batch:batch+batch_size]))
                with torch.no_grad():
                    if args.model == 'FastGTN':
                        val_loss, val_mse, y_valid,_ = model.forward(a, valid_node_features[batch:batch+batch_size], valid_target[batch:batch+batch_size], epoch=epoch)
                    else:
                        val_loss, val_mse, y_valid,_ = model.forward(a, valid_node_features[batch:batch+batch_size], valid_target[batch:batch+batch_size])
                if epoch % 9999 == 0:
                    y_target = valid_target[batch:batch+batch_size]
                    y_target = y_target.detach().cpu().numpy()
                    y_valid = y_valid.detach().cpu().numpy()

                    padding = np.zeros((val_predict.shape[0], val_predict.shape[1], 338))
                    val_predict = np.concatenate((padding, val_predict), axis=2)
                    val_target = np.concatenate((padding, val_target), axis=2)
                    # apply the inverse transform to each dimension
                    for j in range(val_predict.shape[0]):
                        val_predict[j] = scaler.inverse_transform(val_predict[j])
                        val_target[j] = scaler.inverse_transform(val_target[j])
                    val_predict = val_predict[:, :, -1]
                    val_target = val_target[:, :, -1]
                    # concatenate the val_pred and val_tar
                    if val_pred is None:
                        val_pred = val_predict
                        val_tar = val_target
                    else:
                        val_pred = np.concatenate((val_pred, val_predict), axis=0)
                        val_tar = np.concatenate((val_tar, val_target), axis=0)
            if val_pred is not None:
                np.save('./predictions/' + 'pred_time' + str(cur_month) + '_epoch' + str(i) + '.npy', val_pred)
                np.save('./predictions/' + 'target_time' + str(cur_month) + '_epoch' + str(i) + '.npy', val_tar)
                del val_pred, val_tar
                
                avg_valid_loss += val_loss.detach().cpu().numpy() / num_batches
                avg_valid_mse_error += val_mse / num_batches
            print('Epoch: {}\n Valid - Loss: {}\n Valid - MSE: {}\n'.format(epoch, avg_valid_loss, avg_valid_mse_error))
        # save the model
        result_path = './result/'
        torch.save(model.state_dict(), result_path + 'time' + str(cur_month) + '.pkl')