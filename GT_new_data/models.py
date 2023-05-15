import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math



from graph_transformer_net import GraphTransformerNet


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformerNet,
        'GT_LSTM': GT_LSTM,
        'GTPrelifelong': GTPrelifelong,
    }
        
    return models[MODEL_NAME](net_params)

class GTPrelifelong(nn.Module):
    def __init__(self, net_params):
        super(GTPrelifelong, self).__init__()
        self.hidden_dim = net_params['out_dim']
        self.month_len = net_params['month_len']
        self.house_size = net_params['house_size']
        self.gt_list = []
        #for i in range(self.month_len):
        #    self.gt_list.append(GraphTransformerNet(net_params, False))
        #self.gt = nn.ModuleList(self.gt_list)
        self.gt = GraphTransformerNet(net_params, False)
        self.lstm = nn.LSTM(input_size=net_params['feature_size'], hidden_size=self.hidden_dim, num_layers=net_params['num_layers'], batch_first=True)
        self.linear_price = nn.Linear(self.hidden_dim, 1)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):
        """
        :param x: Nodes * input_dim
        :param adj: meta_size * Nodes * Nodes
        :param y_index: last_month(1) * batch_size
        :return: Global features and the price of the last month
        """
        # print('y_index: ' + str(y_index.shape))
        # print('month_len: '+ str(month_len))
        house_size = self.house_size
        #g_emb = self.gt[i](g, h, e, h_lap_pos_enc, h_wl_pos_enc) 
        g_emb = self.gt(g, h, e, h_lap_pos_enc, h_wl_pos_enc)
        seq_list = []
        #print(g_emb)
        #print(g_emb.shape)
        #for i in range(self.month_len):
        #    seq_list.append(
        #        g_emb.index_select(0, torch.LongTensor(range(i * house_size, (i + 1) * house_size)).to(h.device)))  # 0 by row, 1 by column
        #sequence = torch.stack(seq_list, 0)
        if g_emb.shape[0] % (self.month_len-1) == 0:
            sequence = g_emb.view(-1, self.month_len-1, self.hidden_dim)
        else:
            sequence = g_emb.view(1, -1, self.hidden_dim)
        out, hidden = self.lstm(sequence)
        out_allmonth_t = out.reshape(-1, self.hidden_dim)
        out_price_t = self.linear_price(out_allmonth_t)
        # Take out the label of the house where the transaction occurred, and use it as a signal for backpropagation
            # It depends entirely on the length of y_index, which months are included in y_index, and the label of which months is taken
        #label_list = []
        #for i in range(self.month_len):
        #    label_list.append(out_price_t.index_select(0, y_index[i]))
        #out_price = torch.stack(label_list, 0)  
        out_allmonth = out_allmonth_t
        out_price = out_price_t
        return self.LeakyReLU(out_price)

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss



class GT_LSTM(nn.Module):
    def __init__(self, net_params):
        super(GT_LSTM, self).__init__()
        self.hidden_dim = net_params['out_dim']
        self.month_len = net_params['month_len']
        self.house_size = net_params['house_size']
        self.gt_list = []
        #for i in range(self.month_len):
        #    self.gt_list.append(GraphTransformerNet(net_params, False))
        #self.gt = nn.ModuleList(self.gt_list)
        self.gt = GraphTransformerNet(net_params, False)
        self.lstm = nn.LSTM(input_size=net_params['feature_size'], hidden_size=self.hidden_dim, num_layers=net_params['num_layers'], batch_first=True)
        self.linear_price = nn.Linear(self.hidden_dim, 1)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):
        """
        :param x: Nodes * input_dim
        :param adj: meta_size * Nodes * Nodes
        :param y_index: last_month(1) * batch_size
        :return: Global features and the price of the last month
        """
        # print('y_index: ' + str(y_index.shape))
        # print('month_len: '+ str(month_len))
        house_size = self.house_size
        #g_emb = self.gt[i](g, h, e, h_lap_pos_enc, h_wl_pos_enc) 
        g_emb = self.gt(g, h, e, h_lap_pos_enc, h_wl_pos_enc)
        seq_list = []
        #print(g_emb)
        #print(g_emb.shape)
        #for i in range(self.month_len):
        #    seq_list.append(
        #        g_emb.index_select(0, torch.LongTensor(range(i * house_size, (i + 1) * house_size)).to(h.device)))  # 0 by row, 1 by column
        #sequence = torch.stack(seq_list, 0)
        if g_emb.shape[0] % (self.month_len-1) == 0:
            sequence = g_emb.view(-1, self.month_len-1, self.hidden_dim)
        else:
            sequence = g_emb.view(1, -1, self.hidden_dim)
        out, hidden = self.lstm(sequence)
        out_allmonth_t = out.reshape(-1, self.hidden_dim)
        out_price_t = self.linear_price(out_allmonth_t)
        # Take out the label of the house where the transaction occurred, and use it as a signal for backpropagation
            # It depends entirely on the length of y_index, which months are included in y_index, and the label of which months is taken
        #label_list = []
        #for i in range(self.month_len):
        #    label_list.append(out_price_t.index_select(0, y_index[i]))
        #out_price = torch.stack(label_list, 0)  
        out_allmonth = out_allmonth_t
        out_price = out_price_t
        return self.LeakyReLU(out_price)

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss



class LSTM_static(nn.Module):
    def __init__(self, config):
        super(LSTM_static, self).__init__()
        self.house_size = config.house_size
        self.nfeat = config.nfeat
        self.lstm = nn.LSTM(input_size=self.nfeat, hidden_size=self.nfeat, num_layers=config.num_layers, bidirectional=config.bidirectional)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = config.dropout
        self.linear_price = nn.Linear(2*self.nfeat, 1)

    def forward(self, x):
        shape = x.shape[0]
        house_size = self.house_size
        seq_len = int(shape/house_size)
        # Construct time series
        seq_list = []
        for i in range(seq_len):
            seq_list.append(
                x.index_select(0, torch.LongTensor(range(i * house_size, (i + 1) * house_size)).to(x.device)))  # 0 by row, 1 by column
        sequence = torch.stack(seq_list, 0)  # month_len, batch_size, lstm_input_dim
        # LSTM training on all embeddings generated by GCN
        #print(sequence.shape)
        out, hidden = self.lstm(sequence)  # out:(month_len, batch_size, hidden_size)
        #print(out.shape, self.nfeat)
        out = out.view(shape, self.nfeat)
        x = self.linear_price(out)
        return x 

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.output_dim = 1
        self.dimension = self.num_layers * 2 if config.bidirectional else self.num_layers
        self.lstm = nn.LSTM(config.input_dim, config.hidden_dim, config.num_layers, batch_first=True,\
         dropout=config.dropout, bidirectional=config.bidirectional)
        fc_input_dim = self.hidden_dim * 2 if config.bidirectional else self.hidden_dim
        self.fc = nn.Linear(fc_input_dim, self.output_dim)

    def forward(self, x):
        x = x.unsqueeze(0)
        
        h0 = torch.zeros(self.dimension, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.dimension, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out) 

        return out.squeeze(0)