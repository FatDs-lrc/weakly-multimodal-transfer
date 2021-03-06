import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .teacher_networks import TeacherModel
from .networks.classifier import LSTMClassifier, FcClassifier
from .networks.tools import init_net

class TeacherStudentModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--mid_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--input_dim', type=int, default=1582, help='output dim')
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--teacher_path', type=str, default='/data2/ljj/SSER_model/setup_ss_batchsize_128_lr_0.001_tgt_discrete_mark_IS10+bert+decoder+mmd+0.3+negative+valid_ami/')
        parser.add_argument('--kd_weight', type=float, default=1.0, help='weight of kd loss')
        parser.add_argument('--kd_temp', type=float, default=3.0)
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--modality', type=str, help='which modality to use for student model')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.teacher_path = os.path.join(opt.teacher_path, f'cvNo{opt.cvNo}')
        self.teacher_model = TeacherModel(self.teacher_path).cuda()
        self.loss_names = ['CE', 'KD']
        self.modality = opt.modality
        # acoustic student model
        if self.modality == 'acoustic':
            self.model_names = ['A'] 
            layers = list(map(lambda x: int(x), opt.mid_layers.split(',')))
            self.netA = init_net(FcClassifier(opt.input_dim, layers, opt.output_dim), 
                                opt.init_type, opt.init_gain, gpu_ids=opt.gpu_ids)

        # visual student model
        elif self.modality == 'lexical':
            pass

        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_kd = torch.nn.BCELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netA.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.temp = opt.kd_temp
            self.kd_weight = opt.kd_weight
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.acoustic = input['acoustic'].float().to(self.device)
        self.lexical = input['lexical'].float().to(self.device)
        self.label = input['label'].to(self.device)
        # self.label = self.label.argmax(dim=1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            with torch.no_grad():
                self.teacher_logits = self.teacher_model(self.acoustic, self.lexical)
                self.teacher_pred = F.softmax(self.teacher_logits / self.temp, dim=-1)
        
        if self.modality == 'acoustic':
            self.student_logits, _ = self.netA(self.acoustic)
        elif self.modality == 'lexical':
            self.student_logits = self.netL(self.lexical)

        self.student_pred = F.softmax(self.student_logits / self.temp, dim=1)

   
    def backward(self):
        """Calculate the loss for reconstruct feature"""
        self.loss_CE = self.criterion_ce(self.student_pred, self.label)
        self.loss_KD = self.kd_weight * self.criterion_kd(self.student_pred, self.teacher_pred)
        # loss = self.loss_CE + self.loss_KD
        loss = self.loss_KD
        loss.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
