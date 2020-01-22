'''
@Author: your name
@Date: 2019-12-08 19:50:13
@LastEditTime : 2019-12-30 12:10:11
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \SSER\acoustic_model.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class acoustic_autoencoder(nn.Module):
    def __init__(self):
        super(acoustic_autoencoder, self).__init__()

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
            nn.ReLU(True),
        )


 
    def forward(self, x):

        latent_vector = self.encoder(x)
        reconstructed = self.decoder(latent_vector)
        return latent_vector, reconstructed

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features