import torch
import torch.nn as nn
import torch.nn.functional as F

class AcousticAutoencoder(nn.Module):
    def __init__(self, opt):
        super(AcousticAutoencoder, self).__init__()
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



if __name__ == "__main__":
    net = acoustic_autoencoder()
    print(net)