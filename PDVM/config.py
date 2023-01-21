class DefaultConfig:
    def __init__(self, device):
        self.device = device
        # Path parameters
        self.data_path = '../data/'
        self.dataset = 'processed_data.csv'
        self.ckpt_path = 'checkpoint/'

        # Data parameters
        self.K = 7 # number of nearest neighbors, preferably odd
        self.year = 'all' # maximum amount of samples in 2016
        self.distance_limit = 2 # maximum distance between two points

        # Model parameters
        self.input_dim = 303
        self.init_hidden_dim = 256
        self.hidden_dim = 128
        self.num_layers = 2
        self.num_fc_layers = 1
        self.fc_hidden_dim = 128
        self.dropout = 0.1

        # Training parameters
        self.batch_size = 128
        self.epoch_num = 5000
        self.train_ratio = 0.9
        self.lr = 1
        self.weight_decay = 1e-10
