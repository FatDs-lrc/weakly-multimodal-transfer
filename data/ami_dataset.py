import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy

class AmiDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        acoustic_ft_type = 'IS10'
        visual_ft = "denseface"
        lexical_ft = 'text'
        unsup_norm_data = '/data2/ljj/sser_discrete_data/ami/feature/{}/fts_norm.npy'
        unsup_data = '/data2/ljj/sser_discrete_data/ami/feature/{}/fts.npy'
        self.acoustic_data = np.load(unsup_norm_data.format(acoustic_ft_type))
        self.visual_data = np.load(unsup_norm_data.format(visual_ft))
        self.lexical_data = np.load(unsup_data.format(lexical_ft))
        # mask for visual feature
        self.v_mask = copy.deepcopy(self.visual_data)
        self.v_mask[self.v_mask != 0] = 1
        # mask for acoustic feature
        self.a_mask = copy.deepcopy(self.acoustic_data)
        self.a_mask[self.a_mask != 0] = 1
        # mask for text feature
        self.l_mask = copy.deepcopy(self.lexical_data)
        self.l_mask[self.l_mask != 0] = 1
    
    def __getitem__(self, index):
        acoustic = torch.from_numpy(self.acoustic_data[index])
        visual = torch.from_numpy(self.visual_data[index])
        lexical = torch.from_numpy(self.lexical_data[index])
        a_mask = torch.from_numpy(self.a_mask[index])
        v_mask = torch.from_numpy(self.v_mask[index])
        l_mask = torch.from_numpy(self.l_mask[index])
        return {
            'acoustic': acoustic, 
            'visual': visual, 
            'lexical': lexical, 
            'a_mask': a_mask,
            'v_mask': v_mask,
            'l_mask': l_mask
        }
    
    def __len__(self):
        return len(self.acoustic_data)

if __name__ == '__main__':
    a = AmiDataset()
    print(next(iter(a)))