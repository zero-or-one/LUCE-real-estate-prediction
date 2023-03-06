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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path" , type=str, default='./data/dataset_realestate.csv')
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
    df = df.drop_duplicates(subset=['house'], keep='last')
    df = df.reset_index()
    
    # prepare data for training using one-hot encoding
    df = pd.get_dummies(df)
    # move the target column to the last
    end_col = ['price', 'house', 'year']
    df = df[[c for c in df if c not in end_col] + [c for c in end_col if c in df]]
    # normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, :-2] = scaler.fit_transform(df.iloc[:, :-2])
    # save the scaler
    joblib.dump(scaler, './data/scaler.pkl')
    # save the data
    df.to_csv('./data/processed_data.csv', index=False)
