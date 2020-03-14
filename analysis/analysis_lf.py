import os,sys 
sys.path.append('../')
import time
import torch
import numpy as np
from opts.train_opts import TrainOptions
from data import create_dataset, create_trn_val_tst_dataset
from models import create_model
import json
from models.utils.config import OptConfig
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score


def analysis(models, val_iter):
    total_number = 0
    total_correct = 0
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        batch_size = data['label'].size(0)

        for idx in range(4):
            models[idx].set_input(data)        
            models[idx].test()

        pred_1 = models[0].pred.argmax(dim=1).detach().cpu().numpy()
        pred_2 = models[1].pred.argmax(dim=1).detach().cpu().numpy()
        pred_3 = models[2].pred.argmax(dim=1).detach().cpu().numpy()
        pred_4 = models[3].pred.argmax(dim=1).detach().cpu().numpy()
        correct = (pred_1 == pred_2) * (pred_1 == pred_3) * (pred_1==pred_4) * \
                  (pred_2 == pred_3) * (pred_2 == pred_4) *  \
                  (pred_3 == pred_4)
        
        correct = correct.sum()
        total_correct += correct
        total_number += batch_size
    
    # print("correct: {} total_num:{} rate: {:.4f}".format(total_correct, total_number, total_correct/total_number))
    return '\t'.join([str(total_correct), str(total_number), str(total_correct/total_number)])


def eval(models, val_iter, cv, is_save=False, phase='test'):
    total_pred = []
    total_label = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        final_pred = []
        for idx in range(4):
            models[idx].set_input(data)         # unpack data from dataset and apply preprocessing
            models[idx].test()
            _pred = models[idx].pred
            final_pred.append(_pred)
        
        pred = torch.stack(final_pred)
        pred = torch.mean(pred, dim=0).argmax(dim=1).detach().cpu().numpy()
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
    
    # save test results
    if is_save:
        save_dir = 'checkpoints/test_enssemble/' + str(cv)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

    return acc, uar, f1, cm


if __name__ == '__main__':
    # opt = TrainOptions().parse()                        # get training options
    # opt.isTrain = False                                 # set isTrain = False
    model_name = 'ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run{}'
    total_val = []
    total_tst = []
    for cv in range(1, 11):
        models = []
        for run_idx in range(1,5):
            cur_model_name = model_name.format(run_idx)
            opt_info = json.load(
                open('checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run1/train_opt.conf')
            )
            opt = OptConfig()
            opt.load(opt_info)
            opt.isTrain = False
            opt.gpu_ids = [0]
            opt.cvNo = cv
            model = create_model(opt)      # create a model given opt.model and other options
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.load_networks_cv('checkpoints/{}/{}'.format(cur_model_name, cv))
            model.cuda()
            model.eval()
            models.append(model)

        opt.cvNo = cv
        dataset, val_dataset, tst_dataset = create_trn_val_tst_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        print('The number of training samples = %d' % dataset_size)
        total_val.append(analysis(models, val_dataset))
        total_tst.append(analysis(models, tst_dataset))
        eval(models, val_dataset, cv, is_save=True, phase='val')
        eval(models, tst_dataset, cv, is_save=True, phase='test')
    
    for v, t in zip(total_val, total_tst):
        print(v + '\t' + t)
    