import torch
import torch.nn as nn


class B_LSTM(nn.Module):

    def __init__(self, config):
        super(B_LSTM, self).__init__()
        self.input_dim = config.input_dim
        self.init_hidden_dim = config.init_hidden_dim
        self.hidden_dim = config.hidden_dim
        self.batch_size = config.batch_size
        self.num_layers = config.num_layers
        self.num_fc_layers = config.num_fc_layers
        self.fc_hidden_dim = config.fc_hidden_dim
        self.p = config.dropout
        self.drop = nn.Dropout(self.p)
        self.K = config.K

        self.init_linear = nn.Linear(self.input_dim, self.init_hidden_dim)
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.init_hidden_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)

        # Define the output layer
        if self.num_fc_layers == 1:
            self.linear = nn.Linear(self.hidden_dim*2, 1)
        elif self.num_fc_layers == 2:
            self.linear = nn.Sequential(
                nn.Linear(self.hidden_dim*2, self.fc_hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.p),
                nn.Linear(self.fc_hidden_dim, 1)
            )
        else:
            raise ValueError('num_fc_layers must be 1 or 2')

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size ,hidden_dim]
        # shape of hidden and context: [batch_size, num_layers, hidden_dim].
        input = self.drop(self.init_linear(input))
        lstm_out, (hidden, context) = self.lstm(input)
        y_pred = self.drop(self.linear(lstm_out)).squeeze()
        return y_pred[:, self.K//2+1]