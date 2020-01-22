import torch
import itertools
from .base_model import BaseModel
from .networks.lstm_autoencoder import LSTMAutoencoder
from .networks.fc_autoencoder import AcousticAutoencoder
from .networks.tools import init_net


class AutoencoderModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_size', type=int, default=1024, help='input size of the lstm model')
        parser.add_argument('--hidden_size', type=int, default=512, help='hidden size of the lstm model')
        parser.add_argument('--embedding_size', type=int, default=256, help='embedding size of the lstm model')
        parser.add_argument('--modality', type=str, default='acoustic', help='which modality to use choose from [acoustic, visual, lexical]')
        parser.add_argument('--false_teacher_rate', type=float, default='0.5', help='for lstm autoencoder in training')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['reconstruct']
        self.model_names = ['AE'] 
        self.modality = opt.modality
        if self.modality in ['lexical', 'visual']:
            self.netAE = init_net(LSTMAutoencoder(opt), opt.init_type, opt.init_gain, gpu_ids=opt.gpu_ids)
        else:
            self.netAE = init_net(AcousticAutoencoder(opt), opt.init_type, opt.init_gain, gpu_ids=opt.gpu_ids)
        
        if self.isTrain:
            self.criterion = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netAE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.input_data = input[self.modality].float().to(self.device)
        self.input_mask = input[self.modality[0] + '_mask'].float().to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.recon, self.embd = self.netAE(self.input_data)

   
    def backward(self):
        """Calculate the loss for reconstruct feature"""
        if self.modality == 'acoustic':              # no mask for acoustic IS10 feature
            self.loss_reconstruct = self.criterion(self.recon, self.input_data)
        else:
            self.loss_reconstruct = self.criterion(self.recon * self.input_mask, self.input_data)
        self.loss_reconstruct.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step()       
