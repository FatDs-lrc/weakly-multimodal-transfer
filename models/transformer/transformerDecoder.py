from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models.transformer.decoder_module import *
from models.transformer.position_embedding import SinusoidalPositionalEmbedding


class TransformerDecoder(nn.Module):
    def __init__(self, emb_dim, num_heads, layers, dropout=0.0):
        '''Attention is all you nedd. https://arxiv.org/abs/1706.03762
        Args:
            hp: Hyper Parameters
            enc_voc: vocabulary size of encoder language
            dec_voc: vacabulary size of decoder language
        '''
        super(TransformerDecoder, self).__init__()

        self.num_heads = num_heads
        self.layers = layers

        # decoder
        # self.dec_positional_encoding = SinusoidalPositionalEmbedding(embed_dim)

        self.dec_dropout = nn.Dropout(dropout)
        for i in range(layers):
            self.__setattr__('dec_self_attention_%d' % i,
                             multihead_attention(num_units=emb_dim,
                                                 num_heads=num_heads,
                                                 dropout_rate=dropout,
                                                 causality=True))
            self.__setattr__('dec_vanilla_attention_%d' % i,
                             multihead_attention(num_units=emb_dim,
                                                 num_heads=num_heads,
                                                 dropout_rate=dropout,
                                                 causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(emb_dim,
                                                                    [4 * emb_dim,
                                                                     emb_dim]))
        # self.logits_layer = nn.Linear(self.hp.hidden_units, 1024)


    def forward(self, y, enc):
        # define decoder inputs
        decoder_inputs = y[:, :-1, :] # 2:<S>
        
        # Positional Encoding
        # decoder_inputs += self.embed_positions(decoder_inputs.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        #Dropout
        dec = self.dec_dropout(decoder_inputs)

        # Blocks
        for i in range(self.layers):
            # self-attention
            dec = self.__getattr__('dec_self_attention_%d' % i)(dec, dec, dec)
            # vanilla attention
            dec = self.__getattr__('dec_vanilla_attention_%d' % i)(dec, enc, enc)
            # feed forward
            dec = self.__getattr__('dec_feed_forward_%d' % i)(dec)

        # Final linear projection
        # self.logits = self.logits_layer(self.dec)

        # concat <end>
        return torch.cat( (y[:, :1, :] , dec[:,:21,:]), dim=1)