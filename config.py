
'''
Model parameter configuration
'''

class DefaultConfig:
    def __init__(self, device):
        self.device = device
        self.data_path = './data/'
        self.dataset = 'processed_data.csv'
        self.result_path = 'result/'
        self.lstm_inputdim = 500
        self.input_size = 514
        self.gc1_outdim = 400
        self.gc2_outdim = 400
        self.layers = 1
        self.dropout = 0.2
        self.epoch = 800
        self.batch_size = 1024 #350
        self.seq_len = 17  # the number of years in the data
        self.house_size = 120 # number of houses per year
        self.meta_size = 2
        self.update_len = 4
        self.train_ratio = 0.9
        self.lr = 1e-3
        self.weight_decay = 5e-4
        self.loss = 'nn.MSELoss()'

class PrelifelongConfig(DefaultConfig):
    def __init__(self, device):
        super().__init__(device)
        self.result_path = 'result_prelifelong/'
        self.model = 'r_gcn2lv_1LSTMs(config)'
        self.pretrained_path = None

class LSTMConfig(DefaultConfig):
    def __init__(self, device):
        super().__init__(device)
        self.result_path = 'result_lstm/'
        self.model = 'LSTM_static(config)'

class GCNLSTMConfig(DefaultConfig):
    def __init__(self, device):
        super().__init__(device)
        self.result_path = 'result_gcn_lstm/'
        self.model = 'GCNlstm_static(config)'

class GCNConfig(DefaultConfig):
    def __init__(self, device):
        super().__init__(device)
        self.result_path = 'result_gcn/'
        self.model = 'GCN2lv_static(config)'

class TGCNConfig(DefaultConfig):
    def __init__(self, device):
        super().__init__(device)
        self.result_path = 'result_tgcn/'
        self.model = 'T_GCN(config)'