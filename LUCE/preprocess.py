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
from sklearn.externals import joblib 


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
    parser.add_argument("--fill_gaps", type=int, default=1)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, index_col=False)
    df = df.dropna()
    df['year'] = pd.DatetimeIndex(df['yyyymmdd']).year
    #df['month'] = pd.DatetimeIndex(df['yyyymmdd']).month

    drop_col = ['building_type', 'Unnamed: 0', 'compare', 'updated_at', 'yyyymmdd',
        'migrated_at', 'Unnamed: 0_y', 'danji_id', 'supply_area', 'pyeong_type', 'supply_pyeong_rep' ,
         'average_maintenance_cost','average_summer_maintenance_cost', 'average_winter_maintenance_cost', 
         'danji_id_hash']
    df = df.drop(drop_col, axis=1)
    df = df.reset_index(drop=True)

    # remove duplicate houses in each year   
    for i in list(set(df.year)):
        df_year = df[df['year'] == i]
        df_year = df_year.drop_duplicates(subset=['house'], keep='last')
        if i == list(set(df.year))[0]:
            df_new = df_year
        else:
            df_new = pd.concat((df_new, df_year))
    df = df_new
    all_houses = (set(df.house))
    # sort houses by year and house id
    df = df.sort_values(by=['year', 'house'])
    df = df.reset_index(drop=True)
    df_lstm = df.copy()
    if args.fill_gaps:
        '''
        # if house is not in the year, fill the missing value with price 0
        for i in list(set(df.year)):
            df_year = df[df['year'] == i]
            
            # find average price of the year
            avg_price = df_year.price.mean()
            
            houses = list(set(df_year.house))
            for h in all_houses:
                if h not in houses:
                    # find the average price of the house in other years
                    #avg_price = df[df['house'] == h].price.mean()
                    # find house from other years
                    row = df[df['house'] == h].iloc[0].copy()
                    row.price = avg_price
                    row.year = i
                    #print(row)
                    df_year = df_year.append(row, ignore_index=True)
            df_year = df_year.sort_values(by=['house'])
            df_year = df_year.reset_index(drop=True)
            if i == list(set(df.year))[0]:
                df_new = df_year
            else:
                df_new = pd.concat((df_new, df_year))
        '''
        # if house is not in the year, fill the missing value the next year's price
        for i in list(set(df.year)):
            df_year = df[df['year'] == i]
            houses = (set(df_year.house))
            for h in all_houses - houses:
                # find house from other years
                row = df[df['house'] == h].iloc[0].copy()
                row.year = i
                #print(row)
                df_year = df_year.append(row, ignore_index=True)
            df_year = df_year.sort_values(by=['house'])
            df_year = df_year.reset_index(drop=True)
            if i == list(set(df.year))[0]:
                df_new = df_year
            else:
                df_new = pd.concat((df_new, df_year))
            #print(len(df_year))
        df = df_new
    # sort houses by year and house id
    df = df.sort_values(by=['year', 'house'])
    #df_lstm = df.copy()
    # create meta path and construct graph
    if args.create_adj:
        house_meta = ['house', 'area_index', 'households',
            'supply_pyeong', 'private_area', 'private_pyeong',
            'private_area_rate', 'entrance_type_x', 'room_count', 'bathroom_count',
            'total_parking', 'parking_households', 'entrance_type_y', 
            'heat_system', 'heat_fuel', 'floor']
        geo_meta = ['house', 'total_households', 'dongs', 'bjd_code', 'sd', 
        'sgg', 'emd', 'lon_x', 'lat_y', 'construct_name']
        # we just use random year
        df_single = df[df['year']==2006]
        df_h = df_single[house_meta]
        #print(df_h.shape)
        df_g = df_single[geo_meta]
        Gh = construct_graph_from_df(df_h)
        Gg = construct_graph_from_df(df_g)
        # Create adjacency matrix for each meta path
        Ah = create_adj(Gh, df_single.house.tolist())
        Ag = create_adj(Gg, df_single.house.tolist())

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
    # scale the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, :-2] = scaler.fit_transform(df.iloc[:, :-2])
    # save the scaler
    joblib.dump(scaler, './data/scaler.pkl')

    # prepare data for training using one-hot encoding
    df_lstm = pd.get_dummies(df_lstm)
    # move the target column to the last
    end_col = ['price', 'house', 'year']
    df_lstm = df_lstm[[c for c in df_lstm if c not in end_col] + [c for c in end_col if c in df_lstm]]
    # scale the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_lstm.iloc[:, :-2] = scaler.fit_transform(df_lstm.iloc[:, :-2])
    # save the scaler
    joblib.dump(scaler, './data/scaler_lstm.pkl')

    # save the data
    df.to_csv('./data/processed_data_yearly.csv', index=False)
    df_lstm.to_csv('./data/processed_data_lstm.csv', index=False)
