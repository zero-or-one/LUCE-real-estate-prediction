import os
from time import time
import dgl
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
#from sklearn.externals import joblib 
import joblib



def geo_adj(distance, id_list):
    adj = np.zeros((len(distance), len(distance)), dtype=np.float64)
    # put the distance value for each id and id_list instance
    for i in range(len(distance)):
        adj[i, id_list] = distance[i]
    # normalize the adjacency matrix
    adj = adj / adj.sum(axis=1, keepdims=True)
    return adj    

    

def create_euc_adj(distance, id_list, sigma):
    """
    sigma: the standard deviation of the Gaussian kernel
    the larger the sigma, the more similar the nodes are
    with respect to the distance
    the range of sigma is [0, 1]
    """
    adj = np.zeros((len(distance), len(distance)), dtype=np.float64)
    p = np.exp(-distance  * (sigma**2/2))
    # put the distance value for each id and id_list instance
    for i in range(len(distance)):
        adj[i, id_list] = p[i]
    # normalize the adjacency matrix
    adj = adj / adj.sum(axis=1, keepdims=True)
    return adj



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path" , type=str, default='./data/data.npz')
    parser.add_argument("--create_adj", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.4)
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True)
    # drop nan values
    #data = data[~np.isnan(data).any(axis=1)]
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    if args.create_adj:
        dist_geo = data['dist_geo']
        id_list_geo = data['idx_geo']
        dist_eucli = data['dist_eucli']
        id_list_eucli = data['idx_eucli']

        A_geo = geo_adj(dist_geo, id_list_geo)
        A_eucli = create_euc_adj(dist_eucli, id_list_eucli, args.sigma)
        # combine the adjacency matrix
        # add dimension at the beginning
        A_geo = A_geo[np.newaxis, :, :]
        A_eucli = A_eucli[np.newaxis, :, :]
        A = np.concatenate((A_geo, A_eucli), axis=0)
        
        print("The true shape of adjacency matrix for house meta path is {}".format(A.shape)) 

        np.save('./data/adjacency.npy', A)
        # It is better to save the adjacency matrix in sparse format, but it is not working
        #sparse.save_npz('./data/adjacency_house.npz', sparse.csr_matrix(Ah))
        #sparse.save_npz('./data/adjacency_geo.npz', sparse.csr_matrix(Ag))


    scaler = MinMaxScaler(feature_range=(-1, 1))

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #df.iloc[:, :-3] = scaler.fit_transform(df.iloc[:, :-3])
    # save the scaler
    joblib.dump(scaler, './data/scaler.pkl')
    # use different scaler for price
    scaler_price = MinMaxScaler(feature_range=(-1, 1))
    y_train = scaler_price.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_price.transform(y_test.reshape(-1, 1))
    # save the scaler
    joblib.dump(scaler_price, './data/scaler_price.pkl')

    # save the data
    np.save('./data/X_train.npy', X_train)
    np.save('./data/X_test.npy', X_test)
    np.save('./data/y_train.npy', y_train)
    np.save('./data/y_test.npy', y_test)
