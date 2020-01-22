import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.transformerEncoder import TransformerEncoder
from models.transformer.transformerDecoder import TransformerDecoder

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        x = x.view(-1, num_features)
        return x

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    def forward(self, x):
        return x.view(-1, 4, 4, 171)

class lexical_autoencoder(nn.Module):
    
    def __init__(self):
        super(lexical_autoencoder, self).__init__()

        self.encoder = TransformerEncoder(1024,4,2)
        self.decoder = TransformerDecoder(1024,4,2)

        self.Conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 4, 4, stride=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(171*4*4,512),
            nn.Linear(512, 128)
        )

        self.deConv = nn.Sequential(
            
            nn.Linear(128, 512),
            nn.Linear(512, 171*4*4),
            Reshape(),
            nn.ConvTranspose2d(4, 64, 4, stride=3, padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1), 
            nn.ReLU(True),
        )

    def forward(self, x):
        x = x.reshape(-1, 22, 1024)
        x = self.encoder(x).reshape(-1, 1, 22, 1024)
        latent_vector = self.Conv(x)

        x_hat = self.deConv(latent_vector).squeeze(1)

        x = x.squeeze(1)
        reconstructed = self.decoder(x[:, range(x.size(1)-1, -1, -1), :], x_hat)
        reconstructed = reconstructed[:, range(reconstructed.size(1)-1, -1, -1), :]

        return latent_vector, reconstructed
