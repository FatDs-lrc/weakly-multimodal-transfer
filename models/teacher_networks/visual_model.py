'''
@Author: your name
@Date: 2019-12-08 19:50:14
@LastEditTime : 2020-01-03 22:00:41
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \SSER\visual_model.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet.resnet import resnet34


class visual_autoencoder(nn.Module):
    def __init__(self):
        super(visual_autoencoder, self).__init__()

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, stride=3, padding=1), 
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  
        #     nn.Conv2d(16, 32, 3, stride=2, padding=1),  
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=1), 
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(32, 16, 3, stride=2),  
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1), 
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 1, 2, stride=2, padding=0), 
        #     nn.Tanh()
        # )
        # self.fc1 = nn.Linear(224*4,128)
        # self.fc2 = nn.Linear(128,224*4)


        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(16, 64, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, stride=3, padding=1),  
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 3, stride=3, padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 16, 5, stride=2, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), 
            nn.ReLU(True)
        )
        self.fc1 = nn.Linear(29*2*32,128)
        self.fc2 = nn.Linear(128,29*2*32)

        print(self)

 
    def forward(self, x):
        x = x.reshape(-1,1,18,342)
        x = self.encoder(x)
        x = x.view(-1, self.num_flat_features(x))
        latent_vector = F.relu(self.fc1(x))

        x_hat = F.relu(self.fc2(latent_vector))
        x_hat = x_hat.view(-1, 32, 2, 29)
        # x_hat = x_hat.view(-1, 8, 1, 28)
        reconstructed = self.decoder(x_hat)
        reconstructed = reconstructed.reshape(-1,1,18,342)
        return latent_vector, reconstructed

        # latent_vector = self.encoder(x)
        # return latent_vector, None

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



if __name__ == "__main__":
    net = visual_autoencoder()
    print(net)