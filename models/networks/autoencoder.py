import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class AcousticAutoencoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1582, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1582),
            # nn.ReLU(True),
        )
 
    def forward(self, x):
        latent_vector = self.encoder(x)
        reconstructed = self.decoder(latent_vector)
        return reconstructed, latent_vector


class LSTMAutoencoder(nn.Module):
    ''' Conditioned LSTM autoencoder
    '''
    def __init__(self, opt):
        super().__init__()
        self.input_size = opt.input_size
        self.hidden_size = opt.hidden_size
        self.embedding_size = opt.embedding_size
        self.false_teacher_rate = opt.false_teacher_rate # use the label instead of the output of previous time step
        super().__init__()
        self.encoder = nn.LSTMCell(self.input_size, self.hidden_size)
        self.enc_fc = nn.Linear(self.hidden_size, self.embedding_size)
        self.decoder = nn.LSTMCell(self.hidden_size + self.input_size, self.input_size)
        self.dec_fc = nn.Linear(self.embedding_size, self.hidden_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        ''' x.size() = [batch, timestamp, dim]
        '''
        # timestamp_size = x.size(1)
        # inverse_range = range(timestamp_size-1, -1, -1)
        # inverse_x = x[:, inverse_range, :]
        outputs = []
        o_t_enc = torch.zeros(x.size(0), self.hidden_size).cuda()
        h_t_enc = torch.zeros(x.size(0), self.hidden_size).cuda()
        o_t_dec = torch.zeros(x.size(0), self.input_size).cuda()
        h_t_dec = torch.zeros(x.size(0), self.input_size).cuda()

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.squeeze(1)
            o_t_enc, h_t_enc = self.encoder(input_t, (o_t_enc, h_t_enc))

        embd = self.relu(self.enc_fc(h_t_enc))
        dec_first_hidden = self.relu(self.dec_fc(embd))
        dec_first_zeros = torch.zeros(x.size(0), self.input_size).cuda()
        dec_input = torch.cat((dec_first_hidden, dec_first_zeros), dim=1)

        for i in range(x.size(1)):
            o_t_dec, h_t_dec = self.decoder(dec_input, (o_t_dec, h_t_dec))
            if self.training and random.random() < self.false_teacher_rate:
                dec_input = torch.cat((dec_first_hidden, x[:, -i-1, :]), dim=1)
            else:
                dec_input = torch.cat((dec_first_hidden, h_t_dec), dim=1)
            outputs.append(h_t_dec)
        
        outputs.reverse()
        outputs = torch.stack(outputs, 1)
        # print(outputs.shape)
        return outputs, embd

if __name__ == '__main__':
    import tools
    m = LSTMAutoencoder(20, 15, 10)
    m = tools.init_net(m).cuda()
    a = torch.ones(4, 12, 20).cuda()
    b, _ = m(a)
    print(b.size())