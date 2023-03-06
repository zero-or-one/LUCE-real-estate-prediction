import torch
import torch.nn as nn


class GraphAttentionLayer(Module):
    """
    Graph attention layer
    This is a single graph attention layer.
    A GAT is made up of multiple such layers.
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        """
        in_features: the number of input features per node
        out_features: the number of output features per node
        n_heads: the number of attention heads
        is_concat: whether the multi-head results should be concatenated or averaged
        dropoutL the dropout probability
        leaky_relu_negative_slope: the negative slope for leaky relu activation
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.linear = nn.Linear(in_features, self.n_hidden n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden 2, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        h is the input node embeddings of shape [n_nodes, in_features].
        adj_mat: the adjacency matrix of shape [n_nodes, n_nodes, n_heads].
        We use shape [n_nodes, n_nodes, 1]since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        adj_mat[i][j]: Trueif there is an edge from node ito node j.
        """

        # Number of nodes
        n_nodes = h.shape[0]
        # The initial transformation,
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)

        # ##Calculate attention score
        g_repeat = g.repeat(n_nodes, 1, 1)
        # g_repeat_interleavegets
        # where each node embedding is repeated n_nodestimes.
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        # Now we concatenate to get
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        # Reshape so that g_concat[i, j]: $\overrightarrow{g_i} \Vert \overrightarrow{g_j}$
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 self.n_hidden)

        # Calculate
        # $$e_{ij} = \text{LeakyReLU} \Big(
        # \mathbf{a}^\top \Big[
        # \overrightarrow{g_i} \Vert \overrightarrow{g_j}
        # \Big] \Big)$$
        # e: of shape [n_nodes, n_nodes, n_heads, 1]
        e = self.activation(self.attn(g_concat))
        # Remove the last dimension of size 1
        e = e.squeeze(-1)

        # The adjacency matrix should have shape
        # [n_nodes, n_nodes, n_heads]or[n_nodes, n_nodes, 1]
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        # Mask $e_{ij}$ based on adjacency matrix.
        # $e_{ij}$ is set to $- \infty$ if there is no edge from $i$ to $j$.
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        # We then normalize attention scores (or coefficients)
        a = self.softmax(e)

        # Apply dropout regularization
        a = self.dropout(a)

        # Calculate final output for each head
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads self.n_hidden)
        # Take the mean of the heads
        else:
            return attn_res.mean(dim=1)


class GAT(Module):
    """
    Graph Attention Network (GAT)
    https://arxiv.org/abs/1710.10903
    This graph attention network has two [graph attention layers](index.html).
    """

    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        """
        in_features: the number of features per node
        n_hidden: the number of features in the first graph attention layer
        n_classes: the number of classes
        n_heads: the number of heads in the graph attention layers
        dropout: the dropout probability
        """
        super().__init__()

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
        # Activation function after first graph attention layer
        self.activation = nn.ELU()
        # Final graph attention layer where we average the heads
        self.output = GraphAttentionLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
        x: the features vectors of shape [n_nodes, in_features]
        adj_mat: the adjacency matrix of the form
         [n_nodes, n_nodes, n_heads]or [n_nodes, n_nodes, 1]
        """
        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.dropout(x)
        # Output layer (without activation) for logits
        return self.output(x, adj_mat)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)