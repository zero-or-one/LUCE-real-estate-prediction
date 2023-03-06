import os
from time import time
import dgl
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 


def distance(lon1, lat1, lon2, lat2):
    # lon1, lat1: longitude and latitude of the first point
    # lon2, lat2: longitude and latitude of the second point
    # return: orthodromic distance between the two points
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + \
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arcsin(np.sqrt(a))
    d = R * c
    return d

# It is incorrect with respect to community level. Change it later.
def create_adj(df, time_limit=3, distance_limit=1):
    '''
    df: dataframe containing the data
    '''
    l = len(df)
    adj_h = np.zeros((l, l), dtype=np.float64)
    adj_g = np.zeros((l, l), dtype=np.float64)
    for i in range(l):
        row1 = df.iloc[i]
        for j in range(i, l):
            row2 = df.iloc[j]
            if row1['year'] == row2['year'] and abs(row1['month'] - row2['month']) <= time_limit:
                adj_h[i][j] = 1
                dist = distance(row1['lon_x'], row1['lat_y'], row2['lon_x'], row2['lat_y'])
                if dist <= distance_limit:
                    adj_g[i][j] = 1                
    return adj_h, adj_g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path" , type=str, default='./data/dataset_realestate.csv')
    parser.add_argument("--create_adj", type=bool, default=True)
    args = parser.parse_args()
    df = pd.read_csv(args.data_path, index_col=False)
    df = df.dropna()
    df['year'] = pd.DatetimeIndex(df['yyyymmdd']).year
    df['month'] = pd.DatetimeIndex(df['yyyymmdd']).month

    drop_col = ['building_type', 'Unnamed: 0', 'compare', 'updated_at',
        'migrated_at', 'Unnamed: 0_y', 'danji_id', 'supply_area', 'pyeong_type', 'supply_pyeong_rep' ,
         'average_maintenance_cost','average_summer_maintenance_cost', 'average_winter_maintenance_cost', 
         'danji_id_hash', 'yyyymmdd']
    df = df.drop(drop_col, axis=1)


    if args.create_adj:
        house_meta = ['house', 'area_index', 'households', 'supply_area_rep',
            'supply_pyeong', 'private_area', 'private_pyeong',
            'private_area_rate', 'entrance_type_x', 'room_count', 'bathroom_count',
            'danji_keyword', 'total_parking',
            'parking_households', 'entrance_type_y', 'heat_system', 'heat_fuel',
            'default_pyeong_type', 'floor', 'price', 'year', 'month']
        geo_meta = ['house', 'danji_x', 'total_households', 'dongs', 'bjd_code', 
        'sd', 'sgg', 'emd', 'lon_x', 'lat_y', 'construct_name', 'price', 'year', 'month']
        
        df_h = df[house_meta]
        df_g = df[geo_meta]

        # Create adjacency matrix for each meta path
        Ah, Ag = create_adj(df)

        print("The true shape of adjacency matrix for house meta path is {}".format(Ah.shape)) 
        print("The true shape of adjacency matrix for geo meta path is {}".format(Ag.shape))

        np.save('./data/adjacency_house.npy', Ah)
        np.save('./data/adjacency_geo.npy', Ag)

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
    # save the data
    df.to_csv('./data/processed_data.csv', index=False)