import os
from time import time
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import joblib 


def construct_graph_from_df(df, G=None, hid='id'):
    '''
    df: dataframe containing the data
    G: networkx graph to which the data is to be added
    '''
    if G is None:
        G = nx.Graph()
    l = list(df.columns)
    done_house = []
    l.remove(hid)
    for _, row in df.iterrows():
        if row[hid] in done_house:
            continue
        G.add_node(row[hid])
        done_house.append(row[hid])
        for col in l:
            G.add_node(row[col])
            G.add_edge(row[hid], row[col])
    return G

# create adjacency matrix following the equaation in paper
# https://arxiv.org/abs/2008.05880
def create_adj(G, id_list):
    '''
    G: networkx graph for the meta path
    id_list: list of ids for which adjacency matrix is to be created
    '''
    adj = np.zeros((len(id_list), len(id_list)), dtype=np.float64)
    for i, h1 in enumerate(id_list):
        for j, h2 in enumerate(id_list):
            l1 = [t[1] for t in list(G.edges(h1))]
            l2 = [t[1] for t in list(G.edges(h2))]
            den = len(l1) + len(l2)
            num = 2 * len(list(set(l1) & set(l2)))
            similarity = num / den
            adj[i][j] = similarity
    return adj

def apply_PC(A):
    '''
    A: adjacency matrix
    '''
    pca = PCA(n_components=100)
    pca.fit(A)
    return pca.transform(A)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path" , type=str, default='./data/kc.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, index_col=False, encoding="utf8")
    df = df.dropna()
    
    # extract the from timestamp
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = pd.DatetimeIndex(df['date']).year
    #df['month'] = pd.DatetimeIndex(df['date']).month
    #df['day'] = pd.DatetimeIndex(df['date']).day
    
    
    drop_col = ['price', 'date']
    df = df.drop(drop_col, axis=1)
    df = df.reset_index(drop=True)
    # leave important features
    imp_features = ['id', 'grade', 'waterfront', 'sqft_living'] #, 'zipcode', 'view' 'year_old']
    df = df[imp_features]
    df = df.reset_index(drop=True)

    G = construct_graph_from_df(df)
    # Create adjacency matrix for each meta path
    A = create_adj(G, df.id.tolist())

    print("The true shape of adjacency matrix for house meta path is {}".format(A.shape)) 
    np.save('./data/adjacency_luce.npy', A)