import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.didi_attention import DiDiAttention
from .networks.lstm_encoder import BiLSTMEncoder
from .networks.classifier import MaxPoolFc
from .networks.tools import init_net

class DiDiModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--hidden_size', type=int, default=100, help='hidden state of lstm model')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--input_a', type=int, default=34, help='Acoustic input size')
        parser.add_argument('--input_l', type=int, default=300, help='lexical input size')
        parser.add_argument('--multihead_num', type=int, default=5, help='multi-head attention num')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.loss_names = ['CE']
        # build model

        self.model_names = ['A', 'L', 'Att', 'C'] 
        self.netA = BiLSTMEncoder(opt.input_a, opt.hidden_size, opt.dropout_rate)
        self.netL = BiLSTMEncoder(opt.input_l, opt.hidden_size, opt.dropout_rate)
        self.netAtt = DiDiAttention(opt.hidden_size *2, opt.hidden_size *2, opt.multihead_num)
        self.netC = MaxPoolFc(opt.hidden_size *2, opt.output_dim)
        # visual model
       

        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            parameters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
           
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
        self.length_a = input['length_a'].long().cuda()
        self.length_l = input['length_l'].long().cuda()
        self.label = input['label'].long().cuda()
        self.mask = input['mask'].float().cuda()
        # self.label = self.label.argmax(dim=1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.sequence_a = self.netA(self.acoustic, self.length_a)
        self.sequence_l = self.netL(self.lexical, self.length_l)
        self.fusion = self.netAtt(self.sequence_a, self.sequence_l, self.length_a, self.length_l)
        self.logits = self.netC(self.fusion)
        self.pred = F.softmax(self.logits, dim=-1)
   
    def backward(self):
        """Calculate the loss for back propagation"""
        torch.autograd.set_detect_anomaly(True)
        self.loss_CE = self.criterion_ce(self.pred, self.label)
        self.loss_CE.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
