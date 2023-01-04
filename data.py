import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import networkx as nx


"""
Generate the input of the model from the various meta path npz files and feature txt files
Need to generate: composite adjacency matrix adj, feature matrix X, label Y; X and Y are divided into train and test
The size of X is: all houses * feature_size (the house will be divided into batch * Nodes during training)
The size of Y is: all house * label
Differentiate the training test set by index, ie train_index, test_index
The size is (month-1) * house_per_month * 1
"""

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path, month_len, house_size):
    # Ensure that the number of houses (house_size) in each month is equal, data_size = month_len * house_size
    print('Loading data...')

    #idx_features_labels = np.genfromtxt("{}n_feature.txt".format(path), dtype=np.float32)
    idx_features_labels = pd.read_csv("{}dummy_preproc.csv".format(path), dtype=np.float32).values

    feature_size = idx_features_labels.shape[1]     # feature dimension
    data_size = idx_features_labels.shape[0]        # Total number of houses for all months

    features = idx_features_labels[:, 0:feature_size-2]  # Features, remove listing price and transaction price
    listprice = idx_features_labels[:, -3]              # List price (temporarily unused during training)
    labels = idx_features_labels[:, -2]                 # final price
    listprice = listprice[:, np.newaxis]
    labels = labels[:, np.newaxis]                      # Column vector to column vector matrix


    # Add index to the last dimension of listprice
    index = np.array(range(0, data_size)).T
    index = index[:, np.newaxis]
    listprice = np.hstack((listprice, index))

    print('feature size: ' + str(features.shape))
    print('label size: ' + str(labels.shape))
    print('listprice size: ' + str(listprice.shape))

    df = pd.read_csv("{}dataset_realestate.csv".format(path))
    G = create_dummy_adj(df)
    adj = nx.to_numpy_matrix(G) # MYBE WRONG
    adj = adj + np.eye(adj.shape[0])  # Add self-loop
    #adj = normalize(adj, diag_lambda=1.0)  # Row-normalize

     # Create indexes for training and test sets
    index = range(0, data_size)
    train_index = []
    test_index = []
    for i in range(month_len - 1):
        train_index.append(index[i*house_size: (i+1)*house_size])
        test_index.append(index[(i+1)*house_size: (i+2)*house_size])
    train_index = np.array(train_index)
    test_index = np.array(test_index)

    np.save(path + 'adj.npy', adj)
    np.save(path + 'features.npy', features)
    np.save(path + 'labels.npy', labels)
    np.save(path + 'listprice.npy', listprice)
    np.save(path + 'train_index.npy', train_index)
    np.save(path + 'test_index.npy', test_index)

    return features, labels, listprice, train_index, test_index


def normalize(mx, diag_lambda):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx + diag_lambda * sp.diags(mx.diagonal())  # Diagonal enhancement
    return mx


def normalize_torch(adj, diag_lambda):
    rowsum = torch.sum(adj, dim=1)
    r_inv = torch.pow(rowsum, -1)
    r_inv = torch.flatten(r_inv)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    adj = r_mat_inv.mm(adj)
    adj = adj + diag_lambda * torch.diag(torch.diag(adj, 0))  # Diagonal enhancement
    return adj


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.changed to np.array"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    return np.array(sparse_tensor.to_dense())


def npz2array(metapath, filepath):
    data = sp.load_npz(filepath + metapath + ".npz")
    data = normalize(data, diag_lambda=5)
    data = sparse_mx_to_torch_sparse_tensor(data)
    print(metapath+": "+str(data.shape))
    return data

def create_dummy_adj(df, G=None):
    if G is None:
        G = nx.Graph()
    for i, row in df.iterrows():
        G.add_node(row['Unnamed: 0'])
        G.add_node(row['house'])
        G.add_node(row['danji_id'])
        G.add_node(row['danji_x'])
        G.add_node(row['area_index'])
        G.add_node(row['households'])
        G.add_node(row['pyeong_type'])
        G.add_node(row['supply_area_rep'])
        G.add_node(row['supply_area'])
        G.add_node(row['supply_pyeong_rep'])
        G.add_node(row['supply_pyeong'])
        G.add_node(row['private_area'])
        G.add_node(row['private_pyeong'])
        G.add_node(row['private_area_rate'])
        G.add_node(row['entrance_type_x'])
        G.add_node(row['room_count'])
        G.add_node(row['bathroom_count'])
        G.add_node(row['average_maintenance_cost'])
        G.add_node(row['average_summer_maintenance_cost'])
        G.add_node(row['average_winter_maintenance_cost'])
        #G.add_node(row['updated_at'])
        #G.add_node(row['migrated_at'])
        #G.add_node(row['danji_id_hash'])
        #G.add_node(row['construct_date'])
        G.add_node(row['total_households'])
        
        
        G.add_edge(row['Unnamed: 0'], row['house'])
        G.add_edge(row['Unnamed: 0'], row['danji_id'])
        G.add_edge(row['Unnamed: 0'], row['danji_x'])
        G.add_edge(row['Unnamed: 0'], row['area_index'])
        G.add_edge(row['Unnamed: 0'], row['households'])
        G.add_edge(row['Unnamed: 0'], row['pyeong_type'])
        G.add_edge(row['Unnamed: 0'], row['supply_area_rep'])
        G.add_edge(row['Unnamed: 0'], row['supply_area'])
        G.add_edge(row['Unnamed: 0'], row['supply_pyeong_rep'])
        G.add_edge(row['Unnamed: 0'], row['supply_pyeong'])
        G.add_edge(row['Unnamed: 0'], row['private_area'])
        G.add_edge(row['Unnamed: 0'], row['private_pyeong'])
        G.add_edge(row['Unnamed: 0'], row['private_area_rate'])
        G.add_edge(row['Unnamed: 0'], row['room_count'])
        G.add_edge(row['Unnamed: 0'], row['bathroom_count'])
        G.add_edge(row['Unnamed: 0'], row['average_maintenance_cost'])
        G.add_edge(row['Unnamed: 0'], row['average_summer_maintenance_cost'])
        G.add_edge(row['Unnamed: 0'], row['average_winter_maintenance_cost'])
        G.add_edge(row['Unnamed: 0'], row['bathroom_count'])
        G.add_edge(row['Unnamed: 0'], row['total_households'])        
        
    return G

