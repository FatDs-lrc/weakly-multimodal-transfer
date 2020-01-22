import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy
import os

class MixDataset(BaseDataset):
    def __init__(self, opt, is_train=None):
        super().__init__(opt)
        acoustic_data_ami, visual_data_ami, lexical_data_ami = self.read_ami()
        acoustic_data_iemo, visual_data_iemo, lexical_data_iemo = self.read_iemocap()
        self.acoustic_data = np.concatenate([acoustic_data_ami, acoustic_data_iemo], 0)
        self.visual_data = np.concatenate([visual_data_ami, visual_data_iemo], 0)
        self.lexical_data = np.concatenate([lexical_data_ami, lexical_data_iemo], 0)
        # for trn val split
        trn_num = int(0.8 * len(self.acoustic_data))
        if is_train is not None and is_train:
            self.acoustic_data = self.acoustic_data[:trn_num]
            self.visual_data = self.visual_data[:trn_num]
            self.lexical_data = self.lexical_data[:trn_num]
        elif is_train is not None:
            self.acoustic_data = self.acoustic_data[trn_num:]
            self.visual_data = self.visual_data[trn_num:]
            self.lexical_data = self.lexical_data[trn_num:]
        # mask for visual feature
        self.v_mask = copy.deepcopy(self.visual_data)
        self.v_mask[self.v_mask != 0] = 1
        # mask for acoustic feature
        self.a_mask = copy.deepcopy(self.acoustic_data)
        self.a_mask[self.a_mask != 0] = 1
        # mask for text feature
        self.l_mask = copy.deepcopy(self.lexical_data)
        self.l_mask[self.l_mask != 0] = 1
    
    def read_ami(self):
        acoustic_ft_type = 'IS10'
        visual_ft = "denseface"
        lexical_ft = 'text'
        unsup_norm_data = '/data2/ljj/sser_discrete_data/ami/feature/{}/fts_norm.npy'
        unsup_data = '/data2/ljj/sser_discrete_data/ami/feature/{}/fts.npy'
        acoustic_data_ami = np.load(unsup_norm_data.format(acoustic_ft_type))
        visual_data_ami = np.load(unsup_norm_data.format(visual_ft))
        lexical_data_ami = np.load(unsup_data.format(lexical_ft))
        return acoustic_data_ami, visual_data_ami, lexical_data_ami

    def read_iemocap(self):
        acoustic_ft_type = 'IS10'
        visual_ft = "denseface"
        lexical_ft = 'text'
        data_path = "/data2/ljj/sser_discrete_data/iemocap/feature/{}/crossVal{}/"
        label_path = "/data2/ljj/sser_discrete_data/iemocap/target/crossVal{}/"
        cvNo = 1
        # train
        acoustic_data_trn = np.load(data_path.format(acoustic_ft_type, cvNo) + "trn_norm.npy")
        visual_data_trn = np.load(data_path.format(visual_ft, cvNo) + "trn_norm.npy")
        lexical_data_trn = np.load(data_path.format(lexical_ft, cvNo) + "trn_fts.npy")
        # val
        set_name = 'val'
        acoustic_data_val = np.load(data_path.format(acoustic_ft_type, cvNo) + set_name + "_norm.npy")
        visual_data_val = np.load(data_path.format(visual_ft, cvNo) + set_name + "_norm.npy")
        lexical_data_val = np.load(data_path.format(lexical_ft, cvNo) + set_name + "_fts.npy")
        # total
        acoustic_data = np.concatenate([acoustic_data_trn, acoustic_data_val], 0)
        visual_data = np.concatenate([visual_data_trn, visual_data_val], 0)
        lexical_data = np.concatenate([lexical_data_trn, lexical_data_val], 0)
        return acoustic_data, visual_data, lexical_data
            
    
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
            'l_mask': l_mask,
        }
    
    def __len__(self):
        return len(self.acoustic_data)

if __name__ == '__main__':
    a = MixDataset()
    print(len(a))
    print(next(iter(a)))