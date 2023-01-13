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


def construct_graph_from_df(df, G=None, hid='house'):
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
    parser.add_argument("--data_path" , type=str, default='./data/dataset_realestate.csv')
    parser.add_argument("--create_adj", type=int, default=1)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, index_col=False)
    df = df.dropna()
    df['year'] = pd.DatetimeIndex(df['yyyymmdd']).year
    #df['month'] = pd.DatetimeIndex(df['yyyymmdd']).month

    drop_col = ['building_type', 'Unnamed: 0', 'compare', 'updated_at', 'yyyymmdd',
        'migrated_at', 'Unnamed: 0_y', 'construct_date']
    df = df.drop(drop_col, axis=1)
    df = df.reset_index(drop=True)

    # sort the dataframe by year and remove uniques
    df = df.sort_values(by='year')
    max_num_house = df.house.nunique()
    exists = []
    for i in df.index:
        if (df.loc[i, 'house'], df.loc[i, 'year']) in exists:
            df = df.drop(i, axis=0)
        else:
            exists.append((df.loc[i, 'house'], df.loc[i, 'year']))
    
    df = df.reset_index(drop=True)

    years = []
    for i in range(2006, 2023):
        years.append(len(df.loc[df['year']==i]))
    max_house_num = max(years)
    max_year = years.index(max_house_num) + 2006
    print("The maximum number of houses in a year is {}, year {}".format(max_house_num, max_year))

    for i in range(2006, 2023):
        if len(df.loc[df['year']==i]) < max_house_num:
            num = max_house_num-len(df.loc[df['year']==i])
            row = df.loc[df['year']==i].loc[[df.loc[df['year']==i].index[0]]]
            df = df.append([row] * num, ignore_index=True)
    df = df.sort_values(by='year')
    df = df.reset_index(drop=True)
    if args.create_adj:
        # create meta path and construct graph
        house_meta = ['house', 'area_index', 'households', 'pyeong_type', 'supply_area_rep',
            'supply_area', 'supply_pyeong_rep', 'supply_pyeong', 'private_area', 'private_pyeong',
            'private_area_rate', 'entrance_type_x', 'room_count', 'bathroom_count',
            'average_maintenance_cost', 'average_summer_maintenance_cost',
            'average_winter_maintenance_cost', 'danji_id_hash', 'danji_keyword', 'total_parking',
            'parking_households', 'entrance_type_y', 'heat_system', 'heat_fuel',
            'default_pyeong_type', 'floor']
        geo_meta = ['house', 'danji_id', 'danji_x', 'danji_id_hash', 'total_households',
            'dongs', 'bjd_code', 'sd', 'sgg', 'emd', 'lon_x', 'lat_y', 'construct_name']
        df_h = df[house_meta]
        df_g = df[geo_meta]
        Gh = construct_graph_from_df(df_h)
        Gg = construct_graph_from_df(df_g)
        # Create adjacency matrix for each meta path
        Ah = create_adj(Gh, df.house.tolist())
        Ag = create_adj(Gg, df.house.tolist())

        print("The true shape of adjacency matrix for house meta path is {}".format(Ah.shape)) 
        print("The true shape of adjacency matrix for geo meta path is {}".format(Ag.shape))
        # apply PCA to reduce the dimension of adjacency matrix
        # there is a bug again ...
        #Ah = apply_PC(Ah)
        #Ag = apply_PC(Ag)

        np.save('./data/adjacency_house.npy', Ah)
        np.save('./data/adjacency_geo.npy', Ag)
        # It is better to save the adjacency matrix in sparse format, but it is not working
        #sparse.save_npz('./data/adjacency_house.npz', sparse.csr_matrix(Ah))
        #sparse.save_npz('./data/adjacency_geo.npz', sparse.csr_matrix(Ag))

    # prepare data for training using one-hot encoding
    df = pd.get_dummies(df)
    # move the target column to the last
    end_col = ['price', 'house', 'year']
    df = df[[c for c in df if c not in end_col] + [c for c in end_col if c in df]]

    # save the data
    df.to_csv('./data/processed_data.csv', index=False)
