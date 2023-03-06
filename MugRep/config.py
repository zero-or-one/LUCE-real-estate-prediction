
'''
Model parameter configuration
'''

class DefaultConfig:
    def __init__(self, device):
        self.device = device
        self.data_path = './data/'
        self.dataset = 'processed_data.csv'
        self.result_path = 'result/'

        self.batch_size = 512 #350
        self.meta_size = 2
        self.update_len = 4
        self.train_ratio = 0.9
        self.lr = 1e-3
        self.weight_decay = 5e-4

        self.distance_limit = 0.1
        self.time_limit = 3 # 3 months
