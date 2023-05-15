import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np
import csv
import dgl
from dgl.data.utils import load_graphs
from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib
import pandas as pd

class RealEstateDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, adjacency_names, X, y, train=True):
        """For now, we only support the full dataset as training data."""
        self.data_dir = data_dir
        self.dataset_name = 'RealEstate'
        self.labels = y
        self.df = X
        self.train = train
        # limit the number of samples to 5000
        if self.train:
            self.df = self.df[:5000]
            self.labels = self.labels[:5000]
        else:
            self.df = self.df[-5000:]
            self.labels = self.labels[-5000:]
        self.number_of_nodes = self.df.shape[0]
        # check if saved graph exists
        if os.path.exists(os.path.join(data_dir, 'graph.bin')):
            self.graph, _ = load_graphs(os.path.join(data_dir, 'graph.bin'))
            self.meta_size = self.graph.edata['weight_1'].shape[1]
        else:
            self._prepare_graphs(adjacency_names, self.df)
    
    def _prepare_graphs(self, adjacency_names, node_features):
        graph = None
        i = 0
        '''
        for name in adjacency_names:
            A = np.load(os.path.join(self.data_dir, name))
            # for graph consisteny
            A = np.where(A == 0, 1e-5, A)
            #G = nx.from_numpy_array(A)
            #G = G.to_directed()
            #g = dgl.from_networkx(G, edge_attrs=['weight'])
            # please work
            src, dst = np.nonzero(A)
            g = dgl.graph((src, dst))
            g.edata['weight'] = torch.tensor(A[src, dst])
            g.ndata['feats'] = torch.tensor(node_features)
            if graph is None:
                graph = g
            else:
                i += 1
                graph.edata['weight_{}'.format(i)] = g.edata['weight']
        self.graph = graph
        self.meta_size = i + 1
        '''
        '''
        self.graph_list = []
        name = adjacency_names[0]
        A = np.load(os.path.join(self.data_dir, name))
        self.number_of_nodes = A.shape[0]
        src, dst = np.nonzero(A)
        g = dgl.graph((src, dst))
        g.edata['weight'] = torch.tensor(A[src, dst])
        g.ndata['feats'] = torch.tensor(node_features)

        minimum_weight = 0.8
        # Iterate over each node in the graph and create subgraphs
        for i in range(g.number_of_nodes()):
            # Get the incoming edges and their weights for the current node
            in_edges = g.in_edges(i)
            weights = g.edata['weight'][i*self.number_of_nodes:(i+1)*self.number_of_nodes]
            
            # Create a mask for edges below the threshold
            mask = weights >= minimum_weight
            
            if mask.shape[0] != in_edges[0].shape[0]:
                # solve this problem later
                continue
            #print(mask.shape, in_edges[0].shape, mask_.shape)
            # choose the indices of the edges that are below the threshold
            src = in_edges[0][mask]
            dst = in_edges[1][mask]
            # rename src items from 1 to len(src) 
            src_ = torch.arange(0, len(src))

            # create the graph with these edges
            subg = dgl.DGLGraph((src_, dst))
            subg.edata['weight'] = weights[mask]
            skip_list = [17, 381, 407, 537, 364, 603, 594, 654, 600, 620]
            print(subg.edata['weight'].shape, g.ndata['feats'][src].shape)
            if subg.edata['weight'].shape[0] != g.ndata['feats'][src].shape[0]\
                or g.ndata['feats'][src].shape[0] in skip_list:
                # solve this problem later
                continue
                #break
            subg.ndata['feats'] = g.ndata['feats'][src]
            # Append the subgraph to the list
            self.graph_list.append(subg)
            '''
        self.graph_list = []
        name = 'adjacency.npy'
        A = np.load(os.path.join(self.data_dir, name))
        A = A[0]
        if self.train:
            A = A[:self.df.shape[0], :self.df.shape[0]]
        else:
            A = A[-self.df.shape[0]:, -self.df.shape[0]:]
        limit = node_features.shape[0]
        A = A[:limit, :limit]
        self.number_of_nodes = A.shape[0]
        # experimntal parameters
        minimum_weight = 0.1
        #feature_limit = 100
        for i in range(self.number_of_nodes):
            src, dst = np.nonzero(A)
            # Get the incoming edges and their weights for the current node given A
            chosen = np.where(src == i)
            src = src[chosen]
            dst = dst[chosen]
            weight = A[src, dst]
            mask = weight >= minimum_weight
            src = src[mask]
            dst = dst[mask]
            weight = weight[mask]
            # limit to smaller graphs
            #scr = src[:limit]
            #dst = dst[:limit]
            #weight = weight[:limit]
            # rename dst items from 1 to len(dst)
            src_ = torch.arange(0, len(src))
            dst_ = torch.arange(0, len(dst))
            g = dgl.DGLGraph((src_, dst_))
            #print(A.shape, src.shape, dst.shape, src_.shape, dst_.shape)
            g.edata['weight'] = torch.tensor(A[src_, dst_])
            #print(g.edata['weight'].shape, node_features[dst].shape, node_features[src_].shape)
            #print(A.shape, node_features.shape)
            g.ndata['feats'] = torch.tensor(node_features[dst])#[:,:feature_limit])
            #print("src", src.shape,"feats", g.ndata['feats'].shape)
            self.graph_list.append(g)
            #if i == 5:
            #   break
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_list)
    
    def __getitem__(self, idx):
        """Get the idx-th example from the dataset."""
        '''
        weight = self.graph.edata['weight'][idx*self.number_of_nodes:(idx+1)*self.number_of_nodes].unsqueeze(0)
        for i in range(1,self.meta_size):
            weight_add = self.graph.edata['weight_{}'.format(i)][idx*self.number_of_nodes:(idx+1)*self.number_of_nodes].unsqueeze(0)
            weight = torch.cat((weight, weight_add), dim=0)
        item = self.graph.ndata['feats'][idx], weight,
        return item, self.labels[idx]
        '''
        return self.graph_list[idx], self.labels[idx]

    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        batched_graph = dgl.batch(graphs)       
        return batched_graph, labels

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True
        self.graph_list = [self_loop(g) for g in self.graph_list]

    def _make_full_graph(self):
        # function for converting graphs to full graphs
        # this function will be called only if full_graph flag is True
        self.graph_list = [make_full_graph(g) for g in self.graph_list]    
    
    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.graph_list = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.graph_list]

    def _add_wl_positional_encodings(self):
        # WL positional encoding from Graph-Bert, Zhang et al 2020.
        self.graph_list = [wl_positional_encoding(g) for g in self.graph_list]

        


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in RealEstateDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def make_full_graph(g):
    """
        Converting the given graph to fully connected
        This function just makes full connections
        removes available edge features 
    """

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    full_g.ndata['feat'] = g.ndata['feat']
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges()).long()
    
    try:
        full_g.ndata['lap_pos_enc'] = g.ndata['lap_pos_enc']
    except:
        pass

    try:
        full_g.ndata['wl_pos_enc'] = g.ndata['wl_pos_enc']
    except:
        pass    
    
    return full_g



def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return g

def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1


    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g

