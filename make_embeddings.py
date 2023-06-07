import torch
import numpy as np
import torch.nn as nn
from model_gtn import GTN, GCN
from model_fastgtn import FastGTNs
from model_luce import LUCE
#from model_gtn2 import GTN_GAT
import pickle
import argparse
from torch_geometric.utils import add_self_loops
from sklearn.metrics import f1_score as sk_f1_score
from utils import init_seed, _norm
import copy
import pandas as pd
#from sklearn.externals import joblib 
import joblib
import os


if __name__ == '__main__':
    init_seed(seed=777)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LUCE',
                        help='Model')
    parser.add_argument('--dataset', type=str,
                        help='Dataset', default='REALESTATE')
    parser.add_argument('--epoch', type=int, default=3000,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=256,
                        help='hidden dimensions')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=int, default=0.1, help='learning rate decay')
    parser.add_argument('--lr_decay_step', type=int, default=1000, help='learning rate decay step')
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
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
    parser.add_argument('--pretrained_path', type=str, default='./result/time1.pkl')
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
    adj_matrix = np.load('data/{}.npy'.format("adjacency_luce"))
    adj_matrix = np.expand_dims(adj_matrix, axis=0)
    
    adj_matrix1 = np.load('data/{}.npy'.format("adj_goe"))
    adj_matrix1 = np.expand_dims(adj_matrix1, axis=0)
    #print(adj_matrix1.shape, adj_matrix.shape)
    adj_matrix = adj_matrix[:,:adj_matrix1.shape[1],:adj_matrix1.shape[2]]
    # concatenate two adjacency matrix
    adj_matrix = np.concatenate((adj_matrix, adj_matrix1), axis=0)
    

    num_nodes = adj_matrix.shape[1]
    #exit()
    args.num_nodes = num_nodes
    # add self-loops and normalize if needed
    if args.model == 'FastGTN' and args.dataset != 'AIRPORT':
        edge_index, edge_value = add_self_loops(edge_index, edge_attr=edge_value, fill_value=1e-20, num_nodes=num_nodes)
        deg_inv_sqrt, deg_row, deg_col = _norm(edge_index.detach(), num_nodes, edge_value.detach())
        edge_value = deg_inv_sqrt[deg_row] * edge_value
    seq_len = 1
    update_len = 1
    house_size = num_nodes
    result_path = './result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    node_features = np.load('data/{}.npy'.format("X_train"))

    # initialize a model
    if args.model == 'GTN':
        model = GTN(num_edge=2*num_nodes,
                            num_channels=num_channels,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            num_layers=num_layers,
                            num_nodes=num_nodes,
                            args=args)    
    elif args.model == 'LUCE':
        model = LUCE(num_edge=2*num_nodes,
                            num_channels=num_channels,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            num_layers=num_layers,
                            num_nodes=args.batch_size,
                            args=args)    
    elif args.model == 'GTN_GAT':
        model = GTN_GAT(num_edge=2*num_nodes,
                            num_channels=num_channels,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            num_layers=num_layers,
                            num_nodes=args.batch_size,
                            args=args)         
    elif args.model == 'GCN':
        model = GCN(in_channels=19,
                            out_channels=16,
                            num_nodes=args.batch_size,
                            args=args)

    # load pre-trained model
    chpt = torch.load(args.pretrained_path)
    model.load_state_dict(chpt)

    train_features = np.load('data/{}.npy'.format("X_train"))
    train_features = train_features
    valid_features = np.load('data/{}.npy'.format("X_test"))
    valid_features = valid_features

    train_labels = np.load('data/{}.npy'.format("y_train"))
    valid_labels = np.load('data/{}.npy'.format("y_test"))
    train_labels = train_labels
    valid_labels = valid_labels

    train_target = torch.from_numpy(train_labels).type(torch.FloatTensor).to(device)
    valid_target = torch.from_numpy(valid_labels).type(torch.FloatTensor).to(device)

    train_node_features = torch.from_numpy(train_features).type(torch.FloatTensor).to(device)
    valid_node_features = torch.from_numpy(valid_features).type(torch.FloatTensor).to(device)


    gamma = args.lr_decay
    step_size = args.lr_decay_step

    model.to(device)

    embs = None
    model.eval()
    for batch in range(0, len(train_node_features), args.batch_size):
        batch_size = min(args.batch_size, len(train_node_features)-batch)
        num_batches = len(train_node_features)//batch_size
        # take a batch of adjecency matrix
        a = []

        A = adj_matrix[:,batch:batch+batch_size, batch:batch+batch_size]
        edge_index = torch.from_numpy(np.vstack(A.nonzero())).to(torch.long)
        edge_weight = torch.from_numpy(A[A.nonzero()]).to(torch.float32)
        #print(edge_index.shape, edge_weight.shape)
        a.append((edge_index.to(device), edge_weight.to(device)))  
        num_nodes = edge_index.shape[1]         
        with torch.no_grad():
            emb, _ = model.make_embedding(a, train_node_features[batch:batch+batch_size], train_target[batch:batch+batch_size], num_nodes=num_nodes)
        if embs is None:
            embs = emb
        else:
            embs = torch.cat((embs, emb), dim=0)
    # save embeddings as train
    np.save('data/train_emb.npy', embs.cpu().numpy())
    embs = None
    # save embeddings as test
    for batch in range(0, len(valid_node_features), args.batch_size):
        batch_size = min(args.batch_size, len(valid_node_features)-batch)
        num_batches = len(valid_node_features)//batch_size
        # take a batch of adjecency matrix
        a = []

        A = adj_matrix[:,len(train_node_features)+batch:len(train_node_features)+batch+batch_size,len(train_node_features)+batch:len(train_node_features)+batch+batch_size]
        edge_index = torch.from_numpy(np.vstack(A.nonzero())).to(torch.long)
        edge_weight = torch.from_numpy(A[A.nonzero()]).to(torch.float32)
        #print(edge_index.shape, edge_weight.shape)
        a.append((edge_index.to(device), edge_weight.to(device)))  
        num_nodes = edge_index.shape[1]         
        with torch.no_grad():
            emb, _ = model.make_embedding(a, valid_node_features[batch:batch+batch_size], valid_target[batch:batch+batch_size], num_nodes=num_nodes)
        if embs is None:
            embs = emb
        else:
            embs = torch.cat((embs, emb), dim=0)
    # save embeddings as train
    np.save('data/test_emb.npy', embs.cpu().numpy())
        
        
