import torch
import torch.nn as nn
from .orth_norm import OrthNorm

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, fc1_size, output_size, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1_size = fc1_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.rnn1 = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 4, fc1_size)
        self.fc2 = nn.Linear(fc1_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm((hidden_size * 2, ))
        self.bn = nn.BatchNorm1d(hidden_size * 4)

    def extract_features(self, sequence, rnn1, rnn2, layer_norm):
        output, (final_h1, _) = rnn1(sequence)
        normed_output = layer_norm(output)
        _, (final_h2, _) = rnn2(normed_output)
        return final_h1, final_h2

    def rnn_flow(self, x):
        batch_size = x.size(0) ## batch first
        # ef_reps = torch.cat([sentences, visual, acoustic], 2)
        h1, h2 = self.extract_features(x, self.rnn1, self.rnn2, self.layer_norm)
        h = torch.cat((h1, h2), dim=2).contiguous().view(batch_size, -1)
        return self.bn(h)

    def forward(self, x):
        h = self.rnn_flow(x)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.relu(h)
        o = self.fc2(h)
        return o

class FcClassifier(nn.Module):
    def __init__(self, input_dim, layers, output_dim, activation=nn.ReLU, dropout=0.1):
        ''' Fully Connect classifier
            Parameters:
            --------------------------
            input_dim: input feature dim
            layers: [x1, x2, x3] will create 3 layers with x1, x2, x3 hidden nodes respectively.
            output_dim: output feature dim
            activation: activation function
            dropout: dropout rate
        '''
        super().__init__()
        self.fc1 = nn.Linear(input_dim, layers[0])
        self.bn1 = nn.BatchNorm1d(layers[0])
        self.relu1 = activation()
        self.mid_layers = []
        self.dropout = nn.Dropout(dropout)
        for i in range(1, len(layers)):
            self.mid_layers.append(nn.Linear(layers[i-1], layers[i]))
            if i == len(layers)-1:
                self.mid_layers.append(nn.BatchNorm1d(layers[i]))
            self.mid_layers.append(activation())
        ## if len(layer) == 2 the loop with fail
        if len(layers) == 2:
            self.mid_layers = nn.Sequential(
                nn.Linear(layers[0], layers[1]),
                nn.BatchNorm1d(layers[1]),
                activation(),
            )
        ## make mid_layers to a whole module
        self.mid_layers = nn.Sequential(*self.mid_layers)
        # self.orth_norm = OrthNorm()
        self.layers_norm = nn.LayerNorm(layers[-1])
        self.fc_out = nn.Linear(layers[-1], output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        # feat = self.layers_norm(self.mid_layers(x))
        feat = self.mid_layers(x)
        out = self.fc_out(feat)
        return out, feat