
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
        self.epoch = 100
        self.batch_size = 350
        self.seq_len = 5*12  # the number of years in the data
        self.house_size = 217 # number of houses per year
        self.meta_size = 2
        self.update_len = 4
        self.train_ratio = 0.9
        self.lr = 1e-3
        self.weight_decay = 5e-4
        self.loss = 'nn.MSELoss()'
        self.save_period = 300
        self.num_layers = 3
        self.bidirectional = True
        self.concat = False


class PrelifelongConfig(DefaultConfig):
    def __init__(self, device):
        super().__init__(device)
        self.yearly = True
        if self.yearly:
            self.result_path = 'result_prelifelong_yearly/'
            self.dataset = 'processed_data_yearly.csv'
            self.model = 'r_gcn2lv_1LSTMs(config)'
            self.pretrained_path = None
            self.concat = False
            self.seq_len = 17
            self.update_len = 6
            self.house_size = 184
            self.layers = 1
            self.lr = 1e-3
            self.epoch = 30000
        else:
            self.result_path = 'result_prelifelong_monthly/'
            self.dataset = 'processed_data_monthly.csv'
            self.model = 'r_gcn2lv_1LSTMs(config)'
            self.pretrained_path = None
            self.concat = True
            self.seq_len = 5*12
            self.update_len = 4*12
            self.house_size = 217
            self.epoch = 15000
        

class GCNlstm_staticConfig(DefaultConfig):
    def __init__(self, device):
        super().__init__(device)
        self.result_path = 'result_gcnlstm_static/'
        self.model = 'GCNlstm_static(config)'
        self.lr = 1e-4
        self.epoch = 20000



class LSTMConfig(DefaultConfig):
    def __init__(self, device):
        super().__init__(device)
        self.result_path = 'result_lstm/'
        self.model = 'LSTM(config)'
        self.dataset = 'processed_data_lstm.csv'
        self.hidden_dim = 256
        self.num_layers = 3
        self.lr = 1e-4
        self.input_dim = 338
        self.bidirectional = True


class GCNConfig(DefaultConfig):
    def __init__(self, device):
        super().__init__(device)
        self.result_path = 'result_gcn/'
        self.model = 'GCN2lv_static(config)'
        self.lr = 1e-4
