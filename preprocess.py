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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path" , type=str, default='./data/dataset_realestate.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, index_col=False)
    df = df.dropna()
    df['year'] = pd.DatetimeIndex(df['yyyymmdd']).year
    #df['month'] = pd.DatetimeIndex(df['yyyymmdd']).month

    drop_col = ['building_type', 'Unnamed: 0', 'compare', 'updated_at', 'yyyymmdd',
        'migrated_at', 'Unnamed: 0_y', 'construct_date', 'danji_id', 'danji_x', 'danji_id_hash', 'danji_keyword', 'construct_name', \
        'danji_keyword', 'construct_name']
    df = df.drop(drop_col, axis=1)
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['house'], keep='last')
    df = df.reset_index()
    
    # prepare data for training using one-hot encoding
    df = pd.get_dummies(df)
    # move the target column to the last
    end_col = ['price', 'house', 'year']
    df = df[[c for c in df if c not in end_col] + [c for c in end_col if c in df]]

    # save the data
    df.to_csv('./data/processed_data.csv', index=False)
