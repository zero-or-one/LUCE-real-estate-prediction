"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from train import train_epoch, evaluate_network
from logger import Logger

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from models import gnn_model
from data import RealEstateDGL 

"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, trainset, testset, params, net_params, result_path):

    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = trainset.dataset_name
    logger = Logger(result_path, result_path + 'model_saved/', result_path + 'others/')
    if net_params['lap_pos_enc']:
        st = time.time()
        print("[!] Adding Laplacian positional encoding.")
        trainset._add_laplacian_positional_encodings(net_params['pos_enc_dim'])
        testset._add_laplacian_positional_encodings(net_params['pos_enc_dim'])
        print('Time LapPE:',time.time()-st)
        
    if net_params['wl_pos_enc']:
        st = time.time()
        print("[!] Adding WL positional encoding.")
        trainset._add_wl_positional_encodings()
        testset._add_wl_positional_encodings()
        print('Time WL PE:',time.time()-st)
    
    if net_params['full_graph']:
        st = time.time()
        print("[!] Converting the given graphs to full graphs..")
        trainset._make_full_graph()
        testset._make_full_graph()
        print('Time taken to convert to full graphs:',time.time()-st)    

    logger.save_parameters(MODEL_NAME, params, net_params) 
    device = net_params['device']

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Training Graphs: ", len(trainset))
    print("Test Graphs: ", len(testset))

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=trainset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=trainset.collate)
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                epoch_train_loss, epoch_train_mae, epoch_train_mse, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                epoch_test_loss, epoch_test_mae, epoch_test_mse = evaluate_network(model, device, test_loader, epoch)
                
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss,
                              train_MAE=epoch_train_mae, test_MAE=epoch_test_mae,
                              train_MSE=epoch_train_mse, test_MSE=epoch_test_mse)
                cost_time = time.time()-start
                per_epoch_time.append(cost_time)

                # Saving checkpoint
                logger.log_training(epoch, epoch_train_loss)
                logger.log_testing(epoch, epoch_test_mse, epoch_test_mae, np.sqrt(epoch_test_mse), cost_time)
                if epoch % 300:
                    logger.save_model(model, epoch)
                scheduler.step(epoch_test_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    _, test_mae, test_mse = evaluate_network(model, device, test_loader, epoch)
    training_loss, train_mae, train_mse = evaluate_network(model, device, train_loader, epoch)
    logger.log_training(-1, training_loss)
    logger.log_testing(-1, test_mse, train_mae, np.sqrt(test_mse), 0)
    print("Test MAE: {:.4f}".format(test_mae))
    print("Test MSE: {:.4f}".format(test_mse))
    print("Train MAE: {:.4f}".format(train_mae))
    print("Train MSE: {:.4f}".format(train_mse))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    



def main():    
    """
        USER CONTROLS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json \
        file with training/model/data/param details", default='./config/base_config.json')
    parser.add_argument('--visible_gpus', help="Please give a list of visible GPUs", default='0')
    parser.add_argument('--model', help="Enter a model name if it's different from config", default=None)
    parser.add_argument('--dataset', help="Enter a dataset it's different from config", default=None)
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus

    with open(args.config) as f:
        config = json.load(f)
    if args.model:
        config['model'] = args.model
    if args.dataset:
        config['dataset'] = args.dataset
    
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    
    
    MODEL_NAME = config['model']
    DATASET_NAME = config['dataset']
    #df = pd.read_csv(config['data_dir'] +'/' + DATASET_NAME)
    X_train = np.load(config['data_dir'] + '/X_train.npy')
    X_test = np.load(config['data_dir'] + '/X_test.npy')
    y_train = np.load(config['data_dir'] + '/y_train.npy')
    y_test = np.load(config['data_dir'] + '/y_test.npy')
    trainset = RealEstateDGL(config['data_dir'], config['adjacency_list'], X_train, y_train, True)
    testset = RealEstateDGL(config['data_dir'], config['adjacency_list'], X_test, y_test, False)
    
    result_path = config['result_path']
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = config['params']['batch_size']
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    train_val_pipeline(MODEL_NAME, trainset, testset, config['params'], net_params, result_path)


main()    
