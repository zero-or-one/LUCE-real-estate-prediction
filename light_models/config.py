class Config:
    def __init__(self, device):
        self.device = device
        # Path parameters
        self.data_path = 'data/'
        self.dataset = 'processed_data.csv'
        self.ckpt_path = 'checkpoint/'
        self.model_path = 'model/'
        self.model_name = 'model.pkl'

        # Data parameters
        self.year = 'all' # maximum amount of samples in 2016
        self.number_of_features = 50 # number of features to use
        
        # Model parameters
        self.input_dim = 340
        self.dropout = 0.1

        # Training parameters
        self.batch_size = 256
        self.epoch_num = 5000
        self.train_ratio = 0.9
        self.lr = 1e-3
        self.weight_decay = 1e-10


class DNNConfig(Config):
    def __init__(self, device):
        super().__init__(device)
        self.ckpt_path = 'checkpoint_DNN/'
        
        # Model parameters
        self.hidden_dims = [128, 64, 1] # hidden layers

        # Training parameters
        self.lr = 1e1

class HAConfig(Config):
    def __init__(self, device):
        super().__init__(device)
        self.ckpt_path = 'checkpoint_HA/'
        self.model_path = 'model_HA/'
        self.model_name = 'HA.pkl'
        self.n_estimators = 1000
        self.max_depth = 5
        self.kernel = 'rbf'
        self.gamma = 0.1
        self.C = 10
        self.epsilon = 0.1
                
class SVMConfig(Config):
    def __init__(self, device):
        super().__init__(device)
        self.ckpt_path = 'checkpoint_SVM/'
        self.model_path = 'model_SVM/'
        self.model_name = 'SVM.pkl'
        self.kernel = 'rbf'
        self.gamma = 0.1
        self.C = 10
        self.epsilon = 0.1

class GBRTConfig(Config):
    def __init__(self, device):
        super().__init__(device)
        self.ckpt_path = 'checkpoint_GBRT/'
        self.model_path = 'model_GBRT/'
        self.model_name = 'GBRT.pkl'
        self.n_estimators = 1000
        self.max_depth = 5


class PDVMConfig(Config):
    def __init__(self, device):
        self.ckpt_path = 'checkpoint_PDVM/'

        # Data parameters
        self.K = 5 # number of nearest neighbors, preferably odd
        self.year = 'all' # maximum amount of samples in 2016
        self.distance_limit = 2 # maximum distance between two points

        # Model parameters
        self.input_dim = 338
        self.init_hidden_dim = 256
        self.hidden_dim = 128
        self.num_layers = 2
        self.num_fc_layers = 1
        self.fc_hidden_dim = 128
        self.dropout = 0.1

        # Training parameters
        self.lr = 1e-3
