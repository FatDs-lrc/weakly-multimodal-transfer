import torch
import torch.nn as nn
import os
from .acoustic_model import acoustic_autoencoder
from .visual_model import visual_autoencoder
from .visual_trans import visual_autoencoder as vt
from .lexical_model import lexical_autoencoder
from .classifier import classifier


class TeacherModel(nn.Module):
    ''' Import multimodal teacher model as one class '''
    def __init__(self, pretrained_path, fusion_dim=256):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.A = acoustic_autoencoder()
        self.L = lexical_autoencoder()
        self.C = classifier('discrete', fusion_dim)
        # load from saved file on disk
        self.A.load_state_dict(torch.load(os.path.join(self.pretrained_path, 'A.pt')))
        self.L.load_state_dict(torch.load(os.path.join(self.pretrained_path, 'L.pt')))
        self.C.load_state_dict(torch.load(os.path.join(self.pretrained_path, 'C.pt')))
        # set no grad
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        # print network
        print("Teacher model init from {}".format(self.pretrained_path))
        print(self)
        
    def forward(self, x_a, x_l):
        latent_A, _ = self.A(x_a)
        latent_L, _ = self.L(x_l)
        fusion = torch.cat([latent_A, latent_L], 1)
        pred = self.C(fusion)
        return pred


class TeacherModel_AVL(nn.Module):
    ''' Import multimodal teacher model as one class '''
    def __init__(self, pretrained_path, fusion_dim=384):
        super().__init__()
        self.pretrained_path = pretrained_path
        print('Teacher Loaded from :', self.pretrained_path)
        self.A = acoustic_autoencoder()
        self.L = lexical_autoencoder()
        self.V = vt()
        self.C = classifier('discrete', fusion_dim)
        # load from saved file on disk
        self.A.load_state_dict(torch.load(os.path.join(self.pretrained_path, 'A.pt')))
        self.L.load_state_dict(torch.load(os.path.join(self.pretrained_path, 'L.pt')))
        self.V.load_state_dict(torch.load(os.path.join(self.pretrained_path, 'V.pt')))
        self.C.load_state_dict(torch.load(os.path.join(self.pretrained_path, 'C.pt')))
        # set no grad
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        # print network
        print("Teacher model init from {}".format(self.pretrained_path))
        
    def forward(self, x_a, x_l, x_v):
        latent_A, _ = self.A(x_a)
        latent_L, _ = self.L(x_l)
        latent_V, _ = self.V(x_v)
        fusion = torch.cat([latent_A, latent_L, latent_V], 1)
        pred = self.C(fusion)
        return pred