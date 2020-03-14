import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .teacher_networks import TeacherModel, TeacherModel_AVL
from .networks.classifier import GruEncoder, FcEncoder, SimpleClassifier, TextCNN, LSTMEncoder
from .networks.tools import init_net, MidLayerFeatureExtractor
from .networks.mmd_loss import MMD_loss

class TSModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--mid_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--input_dim_a', type=int, default=1582, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--hidden_size', default=256, type=int, help='rnn hidden layer')
        parser.add_argument('--fc1_size', default=128, type=int, help='rnn embedding size')
        parser.add_argument('--embd_dim', type=int, default=128, help='fusion model fusion size')
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--teacher_path', type=str, default='None')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--kd_weight', type=float, default=1.0, help='weight of kd loss')
        parser.add_argument('--mmd_weight', type=float, default=1e-3, help='weight of mmd loss')
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
        # self.teacher_path = os.path.join(opt.teacher_path, f'cvNo{opt.cvNo}')
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        trn_cv_number = (opt.cvNo + 1) // 2 - 1
        self.teacher_path = os.path.join(opt.teacher_path, f'cvNo{trn_cv_number}')
        self.teacher_model = TeacherModel_AVL(self.teacher_path).cuda()
        self.loss_names = ['CE', 'KD', ]
        self.modality = opt.modality
        self.model_names = ['C']
        self.netC = SimpleClassifier(opt.embd_dim, opt.output_dim, opt.dropout_rate)
        # acoustic model
        if 'A' in self.modality:
            self.model_names.append('A')
            layers = list(map(lambda x: int(x), opt.mid_layers.split(',')))
            self.netA = FcEncoder(opt.input_dim_a, layers, opt.dropout_rate)
            
        # lexical model
        if 'L' in self.modality:
            self.model_names.append('L')
            self.netL = GruEncoder(opt.input_dim_l, opt.hidden_size, opt.fc1_size)
            
        # visual model
        if 'V' in self.modality:
            self.model_names.append('V')
            self.netV = GruEncoder(opt.input_dim_v, opt.hidden_size, opt.fc1_size)
            
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_kd = torch.nn.BCELoss()
            self.criterion_mmd = MMD_loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.temp = opt.kd_temp
            self.ce_weight = opt.ce_weight
            self.kd_weight = opt.kd_weight
            self.mmd_layer = self.teacher_model.C.net[3]
            self.mmd_weight = opt.mmd_weight
            self.teacher_mid_extractor = MidLayerFeatureExtractor(self.mmd_layer)
        
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
        self.length_l = input['l_length'].long().cuda()
        self.length_v = input['v_length'].long().cuda()
        self.label = input['label'].cuda()
        # self.label = self.label.argmax(dim=1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            with torch.no_grad():
                self.teacher_logits = self.teacher_model(self.acoustic, self.lexical, self.visual)
                self.teacher_pred = F.softmax(self.teacher_logits / self.temp, dim=-1)
                self.teacher_mid_feat = self.teacher_mid_extractor.extract()
        
        final_embd = []
        if 'A' in self.modality:
            # self.student_logits, self.student_feat = self.netA(self.acoustic)
            self.student_feat_A = self.netA(self.acoustic)
            final_embd.append(self.student_feat_A)

        if 'L' in self.modality:
            self.student_feat_L = self.netL(self.lexical, self.length_l)
            final_embd.append(self.student_feat_L)
        
        if 'V' in self.modality:
            self.student_feat_V = self.netV(self.visual, self.length_v)
            final_embd.append(self.student_feat_V)
        
        # early fusion
        self.student_feat = torch.cat(final_embd, dim=-1)
        self.student_logits = self.netC(self.student_feat)
        self.student_pred = F.softmax(self.student_logits / self.temp, dim=-1)
   
    def backward(self):
        """Calculate the loss for back propagation"""
        loss = torch.as_tensor(0.0)
        if len(self.gpu_ids) >0:
            loss = loss.cuda()

        if self.ce_weight > 0:
            self.loss_CE = self.ce_weight * self.criterion_ce(self.student_pred, self.label)
            loss += self.loss_CE

        if self.kd_weight > 0:
            self.loss_KD = self.kd_weight * self.criterion_kd(self.student_pred, self.teacher_pred)
            loss += self.loss_KD

        if self.mmd_weight >0:
            self.loss_mmd = self.mmd_weight * self.criterion_mmd(self.teacher_mid_feat, self.student_feat)
            loss += self.mmd_weight

        loss.backward()
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
