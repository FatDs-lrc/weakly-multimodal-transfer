import torch
import torch.nn as nn
from models.networks.tools import MultiLayerFeatureExtractor
from models.networks.self_modules.fc_encoder import FcEncoder


class TestDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 32))
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
    
    def forward(self, x):
        x1 = self.fc1(x)
        x1_relu = self.relu1(x1)
        x2 = self.fc2(x1_relu)
        x2_relu = self.relu2(x2)
        x3 = self.fc3(x2_relu)
        x3_relu = self.relu1(x3)
        x4 = self.fc4(x3_relu)
        x4_relu = self.relu4(x4)
        x5 = self.fc5(x4_relu)
        x5_relu = self.relu5(x5)
        return x1, x2, x3, x4, x5, x4_relu

class test(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = TestDNN()
    
    def forward(self, x):
        return self.fc(x)

# a = test()
# print(a)
# e = MultiLayerFeatureExtractor(a, 'fc.fc1,fc.fc2,fc.fc3[2],fc.relu4', cuda=False)
# input_data = torch.ones([1, 256]).float()
# output = a(input_data)
# x1, x2, x3 = output[:3]
# x4_relu= output[-1]
# extract1, extract2, extract3, e_relu4 = e.extract()
# print((x1==extract1).any())
# print((x2==extract2).any())
# print((x3==extract3).any())
# print((e_relu4==x4_relu).any())
def load_from_opt_record(file_path):
    lines = open(file_path).readlines()
    opt = {}
    for line in lines:
        if not ':' in line:
            continue
        key, value = line.split(':')[:2]
        key = key.strip()
        #value = value.split('\t')[0].strip()
        print(line)
        print(key)
        print(value.split('[')[0].strip())
        input()

# load_from_opt_record('checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion128,128run2/train_opt.txt')

import torch.nn as nn
import torch
import numpy as np


layer1=nn.Softmax(dim=-1)
layer2=nn.LogSoftmax(dim=-1)
 
input=np.asarray([2,3])
input=torch.Tensor(input)
 
output1=layer1(input)
output2=layer2(input)
print('output1:',output1)
print('log output1', torch.log(output1))
print('output2:',output2)
