import os
import time
import numpy as np
from opts.train_opts import TestOptions
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

if __name__ == '__main__':
    opt = TestOptions().parse()    # get training options
    total_label = []
    total_pred = []
    root = os.path.join(opt.log_dir, opt.name)
    for cv in range(5, 10):
        cv_dir = os.path.join(root, cv)
        cur_cv_pred = np.load(os.path.join(cv_dir, 'test_pred.npy'))
        cur_cv_label = np.load(os.path.join(cv_dir, 'test_label.npy'))
        total_label.append(cur_cv_label)
        total_pred.append(cur_cv_pred)
    
    total_label = np.concatenate(total_label)
    total_pred = np.concatenate(total_pred)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)
    print(opt.name)
    print('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
    print('\n{}'.format(cm))