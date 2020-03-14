import os
import time
import json
import numpy as np
from tqdm import tqdm
from data import create_dataset, create_trn_val_tst_dataset
from models.utils.config import OptConfig
from models.early_fusion_multi_model import EarlyFusionMultiModel
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract(model, val_iter, save_path, phase='val', modality='A'):
    model.eval()
    total_feat = []
    total_label = []
    
    for i, data in tqdm(enumerate(val_iter)):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        feat = getattr(model, 'embd_{}'.format(modality)).detach().cpu().numpy()
        label = data['label']
        total_feat.append(feat)
        total_label.append(label)
    
    # calculate metrics
    total_feat = np.vstack(total_feat)
    total_label = np.concatenate(total_label)
    
    # save model_feat
    np.save(os.path.join(save_path, '{}.npy'.format(phase)), total_feat)
    np.save(os.path.join(save_path, '{}_label.npy'.format(phase)), total_label)

def load_from_opt_record(file_path):
    opt_content = json.load(open(file_path, 'r'))
    opt = OptConfig()
    opt.load(opt_content)
    return opt


if __name__ == '__main__':
    teacher_path = 'checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run1/'
    opt_path = os.path.join(teacher_path, 'train_opt.conf')
    opt = load_from_opt_record(opt_path)
    opt.isTrain = False                             # teacher model should be in test mode
    opt.gpu_ids = [0]
    opt.serial_batches = True
    for cv in range(1, 11):
        opt.cvNo = cv
        teacher_path_cv = os.path.join(teacher_path, str(cv))
        dataset, val_dataset, tst_dataset = create_trn_val_tst_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = EarlyFusionMultiModel(opt)
        model.cuda()
        model.load_networks_cv(teacher_path_cv)

        save_root = 'analysis/teacher_feats/A/' + str(cv)
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        extract(model, dataset, save_root, phase='trn')
        extract(model, val_dataset, save_root, phase='val')
        extract(model, tst_dataset, save_root, phase='tst')
        
