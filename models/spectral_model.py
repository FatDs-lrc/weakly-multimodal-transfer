import os, glob
import torch
import numpy as np
import torch.nn.functional as F
from .base_model import BaseModel
from .teacher_networks import TeacherModel, TeacherModel_AVL
from .networks.classifier import LSTMClassifier, FcClassifier, EF_model
from .networks.tools import init_net
from .networks.spectral_loss import SpectralLoss, OrthPenalty
from .networks.soft_center_loss import SoftCenterLoss

class SpectralModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--mid_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--input_dim_a', type=int, default=1582, help='input dim for acoustic domain')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='input dim for lexical domain')
        parser.add_argument('--hidden_size', type=int, default=512, help='hidden size for lstm')
        parser.add_argument('--lstm_fc1_size', type=int, default=256, help='fc size for lstm_classifier')
        parser.add_argument('--fusion_size', type=int, default=128, help='fusion latent vector size in fusion model')
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--teacher_path', type=str, default='/data2/ljj/SSER_model/setup_ss_batchsize_128_lr_0.001_tgt_discrete_mark_IS10+bert+decoder+mmd+0.3+negative+valid_ami/')
        # parser.add_argument('--kd_weight', type=float, default=1.0, help='weight of kd loss')
        parser.add_argument('--pretrained_dir', type=str, default='checkpoints/spec_pretrained_dir')
        parser.add_argument('--kd_temp', type=float, default=3.0, help='knowledge distilling temperature')
        parser.add_argument('--kd_start_epoch', type=int, default=10, help='knowledge distilling start epoch')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--modality', type=str, help='which modality to use for student model')
        parser.add_argument('--adjacent_path', type=str, help='path to adjacent matrix file')
        parser.add_argument('--orth_weight', type=float, help='weight of orthogonal penalty')
        parser.add_argument('--spec_weight', type=float, help='weight of spectral loss')
        parser.add_argument('--center_weight', type=float, default=0.0, help='weight of soft center weight')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.teacher_path = os.path.join(opt.teacher_path, f'cvNo{opt.cvNo}')
        self.teacher_model = TeacherModel_AVL(self.teacher_path, fusion_dim=384).cuda()
        self.loss_names = ['KD', 'spec', 'orth', 'center']
        self.modality = opt.modality
        # acoustic student model
        if self.modality == 'acoustic':
            self.model_names = ['A'] 
            layers = list(map(lambda x: int(x), opt.mid_layers.split(',')))
            self.netA = init_net(FcClassifier(opt.input_dim_a, layers, opt.output_dim), 
                                opt.init_type, opt.init_gain, gpu_ids=opt.gpu_ids)

        # visual student model
        elif self.modality == 'lexical':
            self.model_names = ['L']
            self.netL = init_net(LSTMClassifier(opt.input_dim_v, opt.hidden_size, 
                                    opt.lstm_fc1_size, opt.output_dim, opt.dropout_rate),
                                    opt.init_type, opt.init_gain, gpu_ids=opt.gpu_ids)
            # input_size, hidden_size, fc1_size, output_size, dropout_rate
            
        elif self.modality == 'A+L':
            self.model_names = ['A_L']
            layers = list(map(lambda x: int(x), opt.mid_layers.split(',')))
            fc_classifier = FcClassifier(opt.input_dim_a, layers, opt.output_dim) 
            lstm_classifier = LSTMClassifier(opt.input_dim_l, opt.hidden_size, 
                    opt.lstm_fc1_size, opt.output_dim, opt.dropout_rate)
            
            self.netA_L = init_net(EF_model(fc_classifier, lstm_classifier, layers[-1], opt.lstm_fc1_size, 
                                    opt.fusion_size, opt.output_dim, opt.dropout_rate),
                                    opt.init_type, opt.init_gain, gpu_ids=opt.gpu_ids)
            # fc_classifier, lstm_classifier, out_dim_a, out_dim_v, fusion_size, num_class
        if self.isTrain:
            self.adjacent = np.load(opt.adjacent_path)
            self.criterion_kd = torch.nn.BCELoss()
            self.criterion_spec = SpectralLoss(self.adjacent)
            self.criterion_orth = OrthPenalty()
            self.criterion_center = SoftCenterLoss(4, layers[-1] if self.modality=='acoustic' 
                                                        else opt.fusion_size)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(
                        getattr(self, 'net'+self.model_names[0]).parameters(), 
                        lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_clf = torch.optim.Adam(self.netA.module.fc_out.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # print(self.netA.module.[-1])
            # input()
            self.optimizers.append(self.optimizer)
            self.temp = opt.kd_temp
            self.spec_weight = opt.spec_weight
            self.orth_weight = opt.orth_weight
            self.kd_start_epoch = opt.kd_start_epoch
            self.center_weight = opt.center_weight
        
        # init parameter from pretrained model
        if self.isTrain and opt.pretrained_dir != "None":
            print(os.path.join(opt.pretrained_dir, str(opt.cvNo), '*_net_{}'.format(self.model_names[0])))
            model_path = glob.glob(
                os.path.join(opt.pretrained_dir, str(opt.cvNo), '*_net_{}.pth'.format(self.model_names[0])))[0]
            getattr(self, 'net'+self.model_names[0]).module.load_state_dict(torch.load(model_path))
            print("Model init from {}".format(model_path))

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.acoustic = input['acoustic'].float().cuda()
        self.lexical = input['lexical'].float().cuda()
        self.visual = input['visual'].float().cuda()
        self.label = input['label'].cuda()
        self.index = input['index'].cuda()
        # self.label = self.label.argmax(dim=1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            with torch.no_grad():
                self.teacher_logits = self.teacher_model(self.acoustic, self.lexical, self.visual)
                self.teacher_pred = F.softmax(self.teacher_logits / self.temp, dim=-1)
        # for acoustic modality
        if self.modality == 'acoustic':
            self.student_logits, self.student_feat = self.netA(self.acoustic)
        # for lexical modality
        elif self.modality == 'lexical':
            self.student_logits, self.student_feat = self.netL(self.lexical)
        elif self.modality == 'A+L':
            self.student_logits, self.student_feat = self.netA_L(self.acoustic, self.lexical)
            

        self.student_pred = F.softmax(self.student_logits / self.temp, dim=1)

    def backward(self, kd_start):
        """Calculate the loss """
        if kd_start:
            self.loss_KD = self.criterion_kd(self.student_pred, self.teacher_pred)
        else:
            self.loss_KD = torch.tensor(0).cuda()

        self.loss_spec = self.spec_weight * self.criterion_spec(self.student_feat, self.index)
        self.loss_orth = self.orth_weight * self.criterion_orth(self.student_feat)
        self.loss_center = self.center_weight * self.criterion_center(self.student_feat, self.teacher_pred)
        loss = self.loss_KD + self.loss_spec + self.loss_orth + self.loss_center
        loss.backward()
        torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+self.model_names[0]).parameters(), 0.1)

    def backward_spec(self):
        """Calculate the loss for reconstruct feature"""
        # if kd_start:
        #     self.loss_KD = self.criterion_kd(self.student_pred, self.teacher_pred)
        # else:
        #     self.loss_KD = torch.tensor(0).cuda()
        
        self.loss_spec = self.spec_weight * self.criterion_spec(self.student_feat, self.index)
        self.loss_orth = self.orth_weight * self.criterion_orth(self.student_feat)
        self.loss_KD = torch.tensor(0).cuda() 
        # loss = self.loss_KD + self.loss_spec # + self.loss_orth
        loss = self.loss_spec + self.loss_orth
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+self.model_names[0]).parameters(), 0.1)
    
    def backward_clf(self):
        """Calculate the loss for reconstruct feature"""
        self.loss_KD = self.criterion_kd(self.student_pred, self.teacher_pred)
        self.loss_KD.backward()
        torch.nn.utils.clip_grad_norm(self.netA.parameters(), 1)
        self.loss_spec = torch.tensor(0).cuda() 
        self.loss_orth = torch.tensor(0).cuda() 

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad() 
        self.backward(epoch>=self.kd_start_epoch)
        self.optimizer.step()
        # # backward
        # if epoch <= self.kd_start_epoch:
        #     self.optimizer.zero_grad()  
        #     self.backward_spec()
        #     self.optimizer.step()
        # else:
        # # backward clf
        #     self.optimizer_clf.zero_grad()
        #     self.backward_clf()      
        #     self.optimizer_clf.step()