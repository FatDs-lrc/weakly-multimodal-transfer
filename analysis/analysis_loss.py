import os, glob
import sys
import numpy as np
import matplotlib.pyplot as plt


def get_all_losses_from_log(log_file):
    lines = open(log_file).readlines()
    loss = {}
    for line in lines:
        if not line.startswith('2020'): continue
        content = line.split(' - INFO - ')[1]
        if not content.startswith('Cur epoch '): continue
        epoch_num = int(content.split()[2])
        losses = content.split()[4:]
        epoch_loss = {}
        for _loss in losses:
            loss_name = _loss.split(':')[0]
            loss_value = float(_loss.split(':')[1])
            epoch_loss[loss_name] = loss_value
        
        if loss.get(epoch_num) == None:
            loss[epoch_num] = [epoch_loss]
        else:
            loss[epoch_num].append(epoch_loss)

    # manage loss
    loss_names = sorted(list(loss[1][0].keys()))
    total = []
    for epoch in loss.keys():
        # loss[epoch] = [{ce:xx, kd:xxx}, {}, {}]
        epoch_loss = {}
        for loss_name in loss_names:
            epoch_loss[loss_name] = []
            for step in loss[epoch]:
                epoch_loss[loss_name].append(step[loss_name])
            epoch_loss[loss_name] = np.array(epoch_loss[loss_name])
            epoch_loss[loss_name] = np.mean(epoch_loss[loss_name])
        
        total.append(epoch_loss)
    
    total_data = []
    for loss_name in loss_names:
        all_epoch_loss_data = list(map(lambda x: x[loss_name], total))
        total_data.append(all_epoch_loss_data)
    
    total_data = np.array(total_data)
    return loss_names, total_data

def plot(loss_names, loss_data, out_name):
    num_losses = loss_data.shape[0]
    plt.cla()
    colors = ['r', 'b', 'g', 'yellow']
    plt.title('plot loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    for i, loss in enumerate(loss_names):
        x = np.array(range(1, 101))
        y = loss_data[i]
        plt.plot(x, y, colors[i], label=loss)

    group_labels = list(map(lambda x: str(x), x))
    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    # plt.show()
    plt.savefig(out_name)

def check_log_file(log_file):
    lines = open(log_file).readlines()
    loss = {}
    for line in lines:
        if not line.startswith('2020'): continue
        content = line.split(' - INFO - ')[1]
        if not content.startswith('Cur epoch '): continue
        epoch_num = int(content.split()[2])
        if epoch_num == 100:
            return True
    
    return False

def get_log_path(path):
    log_files = glob.glob(os.path.join(path, '*.log'))
    log_files = list(filter(lambda x: check_log_file(x), log_files))
    log_files = sorted(log_files)
    return log_files[0]

root = 'logs'
expr = 'A_ts_Adnn512,256,128_lossBCE_kd1.0_temp2.0_ce0.5_mmd0.0_run1'
save_dir = 'analysis/' + expr
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for cv in range(1, 11):
    log_path = get_log_path(os.path.join(root, expr, str(cv)))
    loss_names, total_data = get_all_losses_from_log(log_path)
    plot(loss_names, total_data, os.path.join(save_dir, str(cv) + '.png'))