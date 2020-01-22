""" Make graph using knn method """
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from iemocap_whole_dataset import get_dataloader
import sys
sys.path.append('../')
from models.networks.lstm_autoencoder import LSTMAutoencoder

def get_embedding_list(model, dataloader, modality='lexical'):
    total_fts = []
    for data in dataloader:
        _, ft = model(data[modality].cuda())
        ft = ft.detach().cpu().numpy()
        total_fts.append(ft)
    total_fts = np.concatenate(total_fts)
    return total_fts

def make_knn_graph(fts, n=100, sigma=1):
    node_num = len(fts)
    adjacent = np.zeros([node_num, node_num])
    for i in range(node_num):
        print(i)
        adj_row = list(map(lambda x: np.exp(-(np.linalg.norm(x-fts[i]) / (2 * sigma))), fts))
        sorted_adj_row = sorted(adj_row, reverse=True)
        rank_n = sorted_adj_row[n]
        for j in range(len(adj_row)):
            if adj_row[j] <= rank_n:
                adj_row[j] = 0
        adj_row = np.array(adj_row)
        adjacent[i, :] = adj_row
    save_path = f'adjacent_matrix_knn{n}_sigma{sigma}.npy'
    np.save(f'./assets/{save_path}', adjacent)

class fake_opt:
    hidden_size = 512
    input_size = 1024
    embedding_size = 256
    false_teacher_rate = 0

if __name__ == '__main__':
    opt = fake_opt()
    modality = sys.argv[1]
    save_path = 'assets/{}_embeds.npy'.format(modality)
    dataloader = get_dataloader()
    model = LSTMAutoencoder(opt)
    model.load_state_dict(torch.load('../checkpoints/autoencoder_lstm_lexical/53_net_AE.pth'))
    model.eval().cuda()
    embeds = get_embedding_list(model, dataloader, modality=modality)
    print(embeds.shape)
    np.save(save_path, embeds)
    make_knn_graph(embeds, 100, 1)
    