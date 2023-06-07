import os
from time import time
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from sklearn.externals import joblib 
import joblib


def haversine_distance(lat1, lon1, lat2, lon2, radius=6371):
    """
    Calculate the Haversine distance between two points given their latitude and longitude coordinates.
    The result is returned in kilometers by default.
    """
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    a = np.sin(delta_lat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = radius * c
    return distance

def calculate_gaussian_similarity(data, sigma):
    n = len(data)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            distance = haversine_distance(data['lat'][i], data['long'][i], data['lat'][j], data['long'][j])
            similarity = np.exp(- (distance ** 2) / (2 * sigma ** 2))
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    # normalize the similarity matrix
    similarity_matrix = similarity_matrix / similarity_matrix.sum(axis=1, keepdims=True)
    return similarity_matrix


def geo_adj(distance, id_list):
    adj = np.zeros((len(distance), len(distance)), dtype=np.float64)
    #adj = np.zeros((5, 5), dtype=np.float64)
    # put the distance value for each id and id_list instance
    for i in range(len(distance)):
        adj[i, id_list] = distance[i]
        print(   adj[i, id_list], "adj i ")# i, id_list)
    # normalize the adjacency matrix
    adj = adj / adj.sum(axis=1, keepdims=True)
    print(adj)
    return adj    

    

def create_euc_adj(distance, id_list, sigma):
    """
    sigma: the standard deviation of the Gaussian kernel
    the larger the sigma, the more similar the nodes are
    with respect to the distance
    the range of sigma is [0, 1]
    """
    adj = np.zeros((len(distance), len(distance)), dtype=np.float64)
    #adj = np.zeros((5, 5), dtype=np.float64)
    p = np.exp(-distance**2 / (2 * sigma**2))
    #print(p.shape, "shape of p", len(distance))
    
    # put the distance value for each id and id_list instance
    for i in range(len(distance)):
        adj[i, id_list] = p[i]
        #print(adj[i, id_list], "p" , i, "check") 
        #sonia
    # normalize the adjacency matrix
    adj = adj / adj.sum(axis=1, keepdims=True)
#     print(adj[11][12])
#     print("*****************", adj)

    return adj



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path" , type=str, default='./data/kc.csv')
    parser.add_argument("--create_adj", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.4)
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    #print(data.shape)
    # remove nan
    data = data.dropna()
    #print(data.shape)
    # convert the date column to datetime object
    data['date'] = pd.to_datetime(data['date'])
    # leave only the year
    data['date'] = data['date'].dt.year
    # convert all the columns to float
    data = data.astype(float)
    # drop the id column
    data = data.drop('id', axis=1)
    if args.create_adj:
        print(data.columns)
        # create the adjacency matrix
        adj = calculate_gaussian_similarity(data, args.sigma)
        # save the adjacency matrix
        np.save('./data/adj_goe.npy', adj)
    # convert the dataframe to numpy array X and y
    y = data['price'].values
    X = data.drop('price', axis=1).values
    # apply log for each feature in X
    for i in range(X.shape[1]):
        X[:, i] = np.log(X[:, i] - (min(X[:, i]) - 1))
    y = np.log(y)
    # split the data into train and test
    # the first 80% of the data is train and the rest is test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    #scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #df.iloc[:, :-3] = scaler.fit_transform(df.iloc[:, :-3])
    # save the scaler
    joblib.dump(scaler, './data/scaler.pkl')
    # use different scaler for price
    scaler_price = MinMaxScaler(feature_range=(-1, 1))
    #scaler_price = StandardScaler()
    y_train = scaler_price.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_price.transform(y_test.reshape(-1, 1))
    # save the scaler
    joblib.dump(scaler_price, './data/scaler_price.pkl')

    # save the data
    np.save('./data/X_train.npy', X_train)
    np.save('./data/X_test.npy', X_test)
    np.save('./data/y_train.npy', y_train)
    np.save('./data/y_test.npy', y_test)
