import torch
import numpy as np
import torch.nn as nn
from model_gtn import GTN
from model_fastgtn import FastGTNs
import pickle
import argparse
from torch_geometric.utils import add_self_loops
from sklearn.metrics import f1_score as sk_f1_score
from utils import init_seed, _norm
import copy
import pandas as pd

if __name__ == '__main__':
    init_seed(seed=777)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GTN',
                        help='Model')
    parser.add_argument('--dataset', type=str,
                        help='Dataset', default='REALESTATE')
    parser.add_argument('--epoch', type=int, default=30000,
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

    '''
    # build adjacency matrices for each edge type
    A = []
    for i,edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        # normalize each adjacency matrix
        if args.model == 'FastGTN' and args.dataset != 'AIRPORT':
            edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_attr=value_tmp, fill_value=1e-20, num_nodes=num_nodes)
            deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
            value_tmp = deg_inv_sqrt[deg_row] * value_tmp
        A.append((edge_tmp,value_tmp))
    edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
    A.append((edge_tmp,value_tmp))
    '''
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

    # add the diagonal entries for each node
    #diag_edge_index = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).to(torch.long)
    #diag_edge_value = torch.ones(num_nodes).to(torch.float)
    #A.append((diag_edge_index, diag_edge_value))

    node_features = pd.read_csv('dataset/{}.csv'.format('processed_data')).values
    labels = np.expand_dims(node_features[:, -3], 1)
    node_features = node_features[:, :-3] + node_features[:, -1:]
    # split train/valid/test by 0.8/0.1/0.1 using indices
    nids = np.arange(0, node_features.shape[0])
    #np.random.shuffle(nids)
    # split nids to 80% train, 10% valid, 10% test
    nids = np.split(nids, [int(0.9*len(nids))])

    train_target = torch.from_numpy(labels[nids[0]]).type(torch.FloatTensor).to(device)
    valid_target = torch.from_numpy(labels[nids[1]]).type(torch.FloatTensor).to(device)
    #test_target = torch.from_numpy(labels[nids[2]]).type(torch.FloatTensor).to(device)

    #train_node = torch.from_numpy(nids[0]).type(torch.LongTensor).to(device)
    #valid_node = torch.from_numpy(nids[1]).type(torch.LongTensor).to(device)
    #test_node = torch.from_numpy(nids[2]).type(torch.LongTensor).to(device)

    num_edge_type = len(A)
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor).to(device)
    train_node_features = node_features[nids[0]]
    valid_node_features = node_features[nids[1]]
    #test_node_features = node_features[nids[2]]

    runs = args.runs
    if args.pre_train:
        runs += 1
        pre_trained_fastGTNs = None
    for l in range(runs):
        # initialize a model
        if args.model == 'GTN':
            model = GTN(num_edge=len(A),
                                num_channels=num_channels,
                                w_in = node_features.shape[1],
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

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        gamma = args.lr_decay
        step_size = args.lr_decay_step
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        model.to(device)
        #model = nn.DataParallel(model)
        loss = nn.L1Loss()
        Ws = []
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
                avg_valid_loss += val_loss.detach().cpu().numpy() / num_batches
                avg_valid_mse_error += val_mse / num_batches
            print('Epoch: {}\n Valid - Loss: {}\n Valid - MSE: {}\n'.format(epoch, avg_valid_loss, avg_valid_mse_error))
            
            '''
            if i % 10 == 0:
                for batch in range(0, len(test_node_features), args.batch_size):
                    batch_size = min(args.batch_size, len(test_node_features)-batch)
                    num_batches = len(test_node_features)//batch_size
                    with torch.no_grad():
                        if args.model == 'FastGTN':
                            test_loss, y_test,W = model.forward(A, test_node_features[batch:batch+batch_size], test_target[batch:batch+batch_size], epoch=i)
                        else:
                            test_loss, y_test,W = model.forward(A, test_node_features[batch:batch+batch_size], test_target[batch:batch+batch_size])
                    avg_test_loss += test_loss.detach().cpu().numpy() / num_batches
                print('Epoch: {}\n Test - Loss: {}\n Test - MSE: {}\n'.format(epoch, avg_test_loss, avg_test_mse_error))
            '''