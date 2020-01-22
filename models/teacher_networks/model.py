'''
@Author: your name
@Date: 2019-12-08 19:50:13
@LastEditTime : 2020-01-11 01:14:34
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \SSER\model.py
'''
from graph.acoustic_model import acoustic_autoencoder
from graph.visual_model import visual_autoencoder
from graph.lexical_model import lexical_autoencoder
from graph.classifier import classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from loss_component.MMD_loss import MMD_loss
import os
from torch.nn import init
from loss_component.CE_loss import CE_loss



class SSER_model(nn.Module):

    def __init__(self, learning_rate, devices, target, path=None):
        super(SSER_model, self).__init__()

        self.A = init_net(acoustic_autoencoder()).cuda() 
        self.V = init_net(visual_autoencoder()).cuda() 
        self.L = init_net(lexical_autoencoder()).cuda()
        self.C = init_net(classifier(target, 128*2)).cuda() 
        self.MSE = nn.MSELoss().cuda()
        self.MMD_loss = MMD_loss().cuda()
        self.CE_loss = CE_loss().cuda()
        
        # Load pretrain model
        if path:
            print('restore')
            self.A.load_state_dict(torch.load(os.path.join(path,"A.pt")))
            # self.V.load_state_dict(torch.load(os.path.join(path,"V.pt")))
            self.L.load_state_dict(torch.load(os.path.join(path,"L.pt")))
            self.C.load_state_dict(torch.load(os.path.join(path,"C.pt")))
    
    def forward(self, sup_A, sup_V, sup_L,
                unsup_A, unsup_V, unsup_L,
                sup_a_mask, unsup_a_mask,  
                sup_v_mask, unsup_v_mask, 
                sup_l_mask, unsup_l_mask, 
                label, unsup_label,
                sup_unpaired_A, sup_unpaired_V, sup_unpaired_L,
                unsup_unpaired_A, unsup_unpaired_V, unsup_unpaired_L,
                setup="ss", dimension=None, epoch=10):

        # ==========supervised part==============

        sup_A = Variable(sup_A).float().cuda()

        sup_V = Variable(sup_V.reshape(-1,1,18,342)).float().cuda()
        # sup_L = Variable(sup_L).float().cuda()

        sup_unpaired_A = Variable(sup_unpaired_A).float().cuda()
        sup_unpaired_V = Variable(sup_unpaired_V).float().cuda()
        # sup_unpaired_L = Variable(sup_unpaired_L).float().cuda()

        latent_A, reconstruct_A = self.A(sup_A)
        latent_V, reconstruct_V = self.V(sup_V)
        # latent_L, reconstruct_L = self.L(sup_L)
        

        # fusion = torch.cat([latent_A, latent_V, latent_L], 1)
        fusion = torch.cat([latent_A, latent_V], 1)
        pred = self.C(fusion)

        # reconstruct_V = reconstruct_V.mul(sup_v_mask.reshape(-1,1,18,342).cuda())

        if dimension != None:
            ce_loss = self.CE_loss(pred, Variable(label[:,dimension]).float().cuda())
        else:
            ce_loss = self.CE_loss(pred, Variable(label).float().cuda())

        loss =  ce_loss
        reconstruct_V = reconstruct_V.mul(sup_v_mask.reshape(-1, 1, 18, 342).cuda())
        reconstruct_loss = self.MSE(reconstruct_A, sup_A).cuda() + self.MSE(reconstruct_V, sup_V).cuda()

        loss += 0.2*reconstruct_loss
    
        # MMd = 0.1 * self.MMD_loss(latent_A, latent_V)
        MMd = 0.1 * self.MMD_loss(latent_A, latent_V).cuda()
        loss += MMd

        unpaired_latent_A, _ = self.A(sup_unpaired_A)
        unpaired_latent_V, _ = self.V(sup_unpaired_V)
        # unpaired_latent_L, _ = self.L(sup_unpaired_L)

        MMd = 0.1 * (-1) * self.MMD_loss(unpaired_latent_A, unpaired_latent_V)
        # MMd += 0.1 * (-1) * self.MMD_loss(unpaired_latent_A, unpaired_latent_L).cuda()
        loss += MMd

        if setup == "fs" or epoch < 10:
            return loss

        elif setup == "ss":
            # ==========unsupervised part==============

            unsup_A = Variable(unsup_A).float().cuda()
            unsup_V = Variable(unsup_V.reshape(-1,1,18,342)).float().cuda()
            unsup_L = Variable(unsup_L.reshape(-1,1,22,1024)).float().cuda()

            unsup_unpaired_A = Variable(unsup_unpaired_A).float().cuda()
            unsup_unpaired_V = Variable(unsup_unpaired_V).float().cuda()
            # unsup_unpaired_L = Variable(unsup_unpaired_L).float().cuda()

            unsup_latent_A, unsup_reconstruct_A = self.A(unsup_A)
            unsup_latent_V, unsup_reconstruct_V = self.V(unsup_V)
            # unsup_latent_L, unsup_reconstruct_L = self.L(unsup_L)

            # fusion = torch.cat([unsup_latent_A, unsup_latent_V, unsup_latent_L], 1)
            fusion = torch.cat([unsup_latent_A, unsup_latent_V], 1)
            unsup_pred = self.C(fusion)

            # if dimension != None:
            #     ce_loss = self.CE_loss(unsup_pred, Variable(unsup_label[:,dimension]).float().cuda())
            # else:
            #     ce_loss = self.CE_loss(unsup_pred, Variable(unsup_label).float().cuda())

            # loss += 0.2 * ce_loss    

            unsup_reconstruct_V = unsup_reconstruct_V.mul(unsup_v_mask.reshape(-1,1,18,342).cuda())
            # unsup_reconstruct_L = unsup_reconstruct_L.reshape(-1,1,22,1024).cuda()

            reconstruct_loss = self.MSE(unsup_reconstruct_A, unsup_A) + self.MSE(unsup_reconstruct_V, unsup_V) #+ self.MSE(unsup_reconstruct_L, unsup_L)
            
            loss += 0.2 * reconstruct_loss
            
            mmd = 0.1 * self.MMD_loss(unsup_latent_A, unsup_latent_V)
            # mmd = 0.1 * self.MMD_loss(unsup_latent_A, unsup_latent_L)
            
            loss += mmd

            unsup_unpaired_latent_A, _ = self.A(unsup_unpaired_A)
            unsup_unpaired_latent_V, _ = self.V(unsup_unpaired_V)
            # unsup_unpaired_latent_L, _ = self.L(unsup_unpaired_L)

            mmd = 0.1 * (-1) * self.MMD_loss(unsup_unpaired_latent_A, unsup_unpaired_latent_V)
            # mmd += 0.1 * (-1) * self.MMD_loss(unsup_unpaired_latent_A, unsup_unpaired_latent_L)
            loss += mmd

            return loss

        '''
        elif setup == "all":

            unsup_A = Variable(unsup_A).float().cuda()
            unsup_V = Variable(unsup_V).float().cuda()
            unsup_L = Variable(unsup_L).float().cuda()
            unsup_unpaired_A = Variable(unsup_unpaired_A).float().cuda()
            unsup_unpaired_V = Variable(unsup_unpaired_V).float().cuda()

            unsup_latent_A, unsup_reconstruct_A = self.A(unsup_A)
            unsup_latent_V, unsup_reconstruct_V = self.V(unsup_V)
            unsup_latent_L, unsup_reconstruct_L = self.L(unsup_L)

            unsup_reconstruct_V = unsup_reconstruct_V.mul(unsup_v_mask.reshape(-1,1,18,342).cuda())

            fusion = torch.cat([unsup_latent_A, unsup_latent_V, unsup_latent_L], 1)
            unsup_pred = self.C(fusion)

            if dimension != None:
                loss += self.CE_loss(unsup_pred, Variable(unsup_label[:,dimension]).float().cuda())
            else:
                loss += self.CE_loss(unsup_pred, Variable(unsup_label).float().cuda())

            unsup_re_loss = self.MSE(unsup_reconstruct_A, unsup_A)  + self.MSE(unsup_reconstruct_V, unsup_V) #+ self.MSE(reconstruct_L, sup_L)
            loss += 0.2 * unsup_re_loss

            MMd = 0.1 * self.MMD_loss(unsup_latent_A, unsup_latent_V)
            loss += MMd

            unsup_unpaired_latent_A, _ = self.A(unsup_unpaired_A)
            unsup_unpaired_latent_V, _ = self.V(unsup_unpaired_V)

            MMd = (-1) * 0.1 * self.MMD_loss(unsup_unpaired_latent_A, unsup_unpaired_latent_V)
            loss += MMd

            return loss
        '''
    
    def val_or_test(self, sup_A, sup_V, sup_L):
        sup_A = Variable(sup_A).float().cuda()
        sup_V = Variable(sup_V).float().cuda()
        sup_L = Variable(sup_L).float().cuda()

        latent_A, _ = self.A(sup_A)
        latent_V, _ = self.V(sup_V)
        latent_L, _ = self.L(sup_L)

        # fusion = torch.cat([latent_A, latent_V, latent_L], 1)
        fusion = torch.cat([latent_A, latent_V], 1)
        pred = self.C(fusion)

        return pred

    # CE_loss for fuzzy label
    
    def save(self, path):
        # save model
        torch.save(self.A.state_dict(), os.path.join(path,"A.pt"))
        torch.save(self.V.state_dict(), os.path.join(path,"V.pt"))
        torch.save(self.L.state_dict(), os.path.join(path,"L.pt"))
        torch.save(self.C.state_dict(), os.path.join(path,"C.pt"))

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    net.cuda()
    init_weights(net, init_type, init_gain=init_gain)
    return net

