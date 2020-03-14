import torch
import os
import json
import torch.nn.functional as F
from .base_model import BaseModel
from .early_fusion_multi_model import EarlyFusionMultiModel
from .networks.self_modules.fc_encoder import FcEncoder
from .networks.self_modules.lstm_encoder import LSTMEncoder
from .networks.self_modules.textcnn_encoder import TextCNN
from .networks.classifier import FcClassifier
from .networks.tools import init_net, MultiLayerFeatureExtractor
from .networks.self_modules.loss import MMD_loss
from .utils.config import OptConfig

'''
测试teacher模型的输出特征, 直接用teacher model的单模态输入特征输入DNN分类
'''
class TeacherTestModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--mid_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--teacher_path', type=str, default='None')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--modality', type=str, choices=['A', 'V', 'L'], help='which modality to use for model')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.teacher_path = os.path.join(opt.teacher_path, str(opt.cvNo))
        self.teacher_config_path = os.path.join(opt.teacher_path, 'train_opt.conf')
        self.teacher_config = self.load_from_opt_record(self.teacher_config_path)
        self.teacher_config.isTrain = False                             # teacher model should be in test mode
        self.teacher_config.gpu_ids = opt.gpu_ids                       # set gpu to the same
        self.teacher_model = EarlyFusionMultiModel(self.teacher_config)
        self.teacher_model.cuda()
        self.teacher_model.load_networks_cv(self.teacher_path)

        self.loss_names = ['CE']
        self.modality = opt.modality
        self.model_names = ['C']
        layers = list(map(lambda x: int(x), opt.mid_layers.split(',')))
        self.netC = FcClassifier(128, layers, opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
            
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            
        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.acoustic = input['acoustic'].float().cuda()
        self.lexical = input['lexical'].float().cuda()
        self.visual = input['visual'].float().cuda()
        self.length_l = input['l_length'].long().cuda()
        self.length_v = input['v_length'].long().cuda()
        self.label = input['label'].cuda()
        self.input = input

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            with torch.no_grad():
                self.teacher_model.set_input(self.input)
                self.teacher_model.test()
                self.teacher_feat = getattr(self.teacher_model, 'embd_{}'.format(self.modality)).detach()
        
        self.logits, _ = self.netC(self.teacher_feat)
        self.pred = F.softmax(self.logits, dim=-1)
   
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        self.loss_CE.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 0.1)


    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt
