import os
import copy
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .base_dataset import BaseDataset



class IemocapSeqDataset(BaseDataset):
    ''' This dataset is for reproduction work of 
        "Learning Alignment for Multimodal Emotion Recognition from Speech"
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--A_ft_type', type=str, default='lld_34', help='lld_34 extracted from pyAudioAnalysis, 34 dim lld feature')
        parser.add_argument('--L_ft_type', type=str, default='glove', help='glove is from glove.840B.300.txt')
        parser.add_argument('--data_path', type=str, default='/data2/ljj/sser_discrete_data/iemocap/feature/{}/crossVal{}/')
        parser.add_argument('--label_path', type=str, default='/data2/ljj/sser_discrete_data/iemocap/target/crossVal{}/')
        return parser

    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        # acoustic_ft_type = opt.A_ft_type
        # lexical_ft_type = opt.L_feat_type
        # data_path = opt.data_path
        # label_path = opt.label_path
        acoustic_ft_type = 'lld_34'
        lexical_ft_type = 'glove'
        data_path = '/data2/ljj/sser_discrete_data/iemocap/feature/{}/crossVal{}/'
        label_path = '/data2/ljj/sser_discrete_data/iemocap/target/crossVal{}/'
        # whether to use manual collate_fn instead of default collate_fn
        self.manual_collate_fn = True 
        self.acoustic_data = np.load(data_path.format(acoustic_ft_type, cvNo) + f"{set_name}_norm.npy", allow_pickle=True)
        self.lexical_data = np.load(data_path.format(lexical_ft_type, cvNo) + f"{set_name}_fts.npy", allow_pickle=True)
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_labels.npy")
        self.label = np.argmax(self.label, axis=1)
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
        acoustic = torch.from_numpy(self.acoustic_data[index])
        lexical = torch.from_numpy(self.lexical_data[index])
        label = torch.Tensor(1).fill_(torch.tensor(self.label[index]))
        index = torch.Tensor(1).fill_(torch.tensor(index))
        length_a = torch.Tensor(1).fill_(torch.tensor(acoustic.size(0)))
        length_l = torch.Tensor(1).fill_(torch.tensor(lexical.size(0)))

        return {
            'acoustic': acoustic, 
            'lexical': lexical, 
            'label': label,
            'index': index,
            'length_a': length_a,
            'length_l': length_l
        }
    
    def __len__(self):
        return len(self.label)
    
    def collate_fn(self, batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        A = pad_sequence([sample['acoustic'] for sample in batch], padding_value=torch.tensor(0.0))
        L = pad_sequence([sample['lexical'] for sample in batch], padding_value=torch.tensor(0.0))
        label = torch.cat([sample['label'] for sample in batch])
        index = torch.cat([sample['index'] for sample in batch])
        length_a = torch.cat([sample['length_a'] for sample in batch])
        length_l = torch.cat([sample['length_l'] for sample in batch])
        max_len_a = int(torch.max(length_a).item())
        max_len_l = int(torch.max(length_l).item())
        mask = torch.zeros([length_a.size(0), max_len_l, max_len_a]).float()
        for i in range(len(batch)):
            mask[i, :length_l[i].int().item(), :length_a[i].int().item()] = 1.0

        return {
            'acoustic': A, 
            'lexical': L, 
            'label': label,
            'index': index,
            'length_a': length_a,
            'length_l': length_l,
            'mask': mask
        }

if __name__ == '__main__':
    class test:
        cvNo = 0
    
    opt = test()
    a = IemocapSeqDataset(opt, 'trn')
    from torch.utils.data import DataLoader
    _iter = DataLoader(a, batch_size=4, shuffle=True, collate_fn=a.collate_fn)
    kk = next(iter(_iter))
    for k,v in kk.items():
        print(k, v.size())
   