import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer.transformerEncoder import TransformerEncoder
from .transformer.transformerDecoder import TransformerDecoder

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
        return x.view(-1, 32, 2, 29)

class visual_autoencoder(nn.Module):
    
    def __init__(self):
        super(visual_autoencoder, self).__init__()

        self.encoder = TransformerEncoder(342,3,2)
        self.decoder = TransformerDecoder(342,3,2)

        self.Conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(16, 64, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, stride=3, padding=1),  
            nn.ReLU(True),
            Flatten(),
            nn.Linear(29*2*32,128)
        )
        self.deConv = nn.Sequential(
            nn.Linear(128,29*2*32),
            Reshape(),
            nn.ConvTranspose2d(32, 64, 3, stride=3, padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 16, 5, stride=2, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), 
            nn.ReLU(True),
            
        )

    def forward(self, x):
        x = x.reshape(-1, 18, 342)
        x = self.encoder(x).reshape(-1, 1, 18, 342)
        latent_vector = self.Conv(x)

        x_hat = self.deConv(latent_vector).squeeze(1)

        x = x.squeeze(1)
        reconstructed = self.decoder(x[:, range(x.size(1)-1, -1, -1), :], x_hat)
        reconstructed = reconstructed[:, range(reconstructed.size(1)-1, -1, -1), :]
        

        return latent_vector, reconstructed
