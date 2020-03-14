import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy


class IemocapDataset(BaseDataset):
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        # acoustic_ft_type = 'IS10'
        acoustic_ft_type = opt.acoustic_ft_type
        visual_ft = "denseface"
        lexical_ft = 'text'
        data_path = "/data2/ljj/sser_discrete_data/iemocap/feature/{}/crossVal{}/"
        label_path = "/data2/ljj/sser_discrete_data/iemocap/target/crossVal{}/"

        self.acoustic_data = np.load(data_path.format(acoustic_ft_type, cvNo) + f"{set_name}_norm.npy")
        self.visual_data = np.load(data_path.format(visual_ft, cvNo) + f"{set_name}_norm.npy")
        self.lexical_data = np.load(data_path.format(lexical_ft, cvNo) + f"{set_name}_fts.npy")
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_labels.npy")
        self.label = np.argmax(self.label, axis=1)

        # mask for visual feature
        self.v_mask = copy.deepcopy(self.visual_data)
        self.v_mask[self.v_mask != 0] = 1
        # mask for acoustic feature
        self.a_mask = copy.deepcopy(self.acoustic_data)
        self.a_mask[self.a_mask != 0] = 1
        # mask for text feature
        self.l_mask = copy.deepcopy(self.lexical_data)
        self.l_mask[self.l_mask != 0] = 1
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
        acoustic = torch.from_numpy(self.acoustic_data[index])
        visual = torch.from_numpy(self.visual_data[index])
        lexical = torch.from_numpy(self.lexical_data[index])
        a_mask = torch.from_numpy(self.a_mask[index])
        v_mask = torch.from_numpy(self.v_mask[index])
        l_mask = torch.from_numpy(self.l_mask[index])
        label = torch.tensor(self.label[index])
        index = torch.tensor(index)

        return {
            'acoustic': acoustic, 
            'visual': visual, 
            'lexical': lexical, 
            'a_mask': a_mask,
            'v_mask': v_mask,
            'l_mask': l_mask,
            'label': label,
            'index': index,
        }
    
    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
    
    opt = test()
    a = IemocapDataset(opt)
    print(next(iter(a)))