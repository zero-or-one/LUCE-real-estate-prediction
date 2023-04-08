import pandas
import numpy as np

df = pandas.read_csv('./data/kc_house_data.csv')
# sort by date
df = df.sort_values(by='date')
date_list = df['date']

import datetime

# convert date to month number
def convert_date_to_month(date):
    date = datetime.datetime.strptime(date, '%Y%m%dT%H%M%S')
    return date.month

date_list = date_list.apply(convert_date_to_month)
df['date'] = date_list
df = df.sort_values(by=['date', 'id'])

df = df.rename(columns={'date': 'month', 'id': 'house'})

import os
from time import time
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import joblib 
import time

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

all_houses = df['house']
# limit to only 200 most frequent houses
all_houses = all_houses.value_counts().index.tolist()
all_houses = all_houses[:1000]

# remove duplicate houses in each year   
for i in list(set(df.month)):
    df_year = df[df['month'] == i]
    df_year = df_year.drop_duplicates(subset=['house'], keep='last')
    # remove houses that are not in the top 200 most frequent houses
    df_year = df_year[df_year['house'].isin(all_houses)]
    if i == list(set(df.month))[0]:
        df_new = df_year
    else:
        df_new = pd.concat((df_new, df_year))
df = df_new
all_houses = (set(df.house))
# sort houses by year and house id
df = df.sort_values(by=['month', 'house'])
df = df.reset_index(drop=True)

for i in list(set(df.month)):
    df_year = df[df['month'] == i]
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
            row.month = i
            #print(row)
            df_year = df_year.append(row, ignore_index=True)
    df_year = df_year.sort_values(by=['house'])
    df_year = df_year.reset_index(drop=True)
    if i == list(set(df.month))[0]:
        df_new = df_year
    else:
        df_new = pd.concat((df_new, df_year))
df = df_new


df = df.sort_values(by=['month', 'house'])

# create meta path and construct graph

house_meta = ['house', 'bedrooms', 'bathrooms',
    'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'yr_built', 
    'yr_renovated']
geo_meta = ['house', 'zipcode', 'lat', 'long', 'sqft_living15', 
'sqft_lot15']
# we just use random year
df_single = df[df['month']==1]
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

np.save('./data/adjacency_house_yearly.npy', Ah)
np.save('./data/adjacency_geo_yearly.npy', Ag)



# prepare data for training using one-hot encoding
df = pd.get_dummies(df)
# move the target column to the last
end_col = ['price', 'house', 'month']
df = df[[c for c in df if c not in end_col] + [c for c in end_col if c in df]]
# scale the data
scaler = MinMaxScaler(feature_range=(-1, 1))
# apply scaler to numeric columns only

df.iloc[:, :-3] = scaler.fit_transform(df.iloc[:, :-3])
# save the scaler
joblib.dump(scaler, './data/scaler.pkl')
# use different scaler for price
scaler_price = MinMaxScaler(feature_range=(-1, 1))
df.iloc[:, -3] = scaler_price.fit_transform(df.iloc[:, -3].values.reshape(-1, 1))
# save the scaler
joblib.dump(scaler_price, './data/scaler_price.pkl')



# save the data
df.to_csv('./data/processed_data_yearly.csv', index=False)


