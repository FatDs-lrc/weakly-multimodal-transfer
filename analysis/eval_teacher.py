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

def eval(model, val_iter, save_dir, phase='test'):
    model.eval()
    total_pred = []
    total_label = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)
    
    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)
    model.train()
    
    # save test results
    np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
    np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

    return acc, uar, f1, cm

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
    opt.dataset_mode = 'iemocap_10fold'
    setattr(opt, 'miss_num', 2)
    for cv in range(1, 11):
        opt.cvNo = cv
        teacher_path_cv = os.path.join(teacher_path, str(cv))
        dataset, val_dataset, tst_dataset = create_trn_val_tst_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = EarlyFusionMultiModel(opt)
        model.cuda()
        model.load_networks_cv(teacher_path_cv)

        save_root = 'checkpoints/teacher_raw_results/' + str(cv)
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        # eval(model, dataset, save_root, phase='trn')
        eval(model, val_dataset, save_root, phase='val')
        eval(model, tst_dataset, save_root, phase='test')
        
