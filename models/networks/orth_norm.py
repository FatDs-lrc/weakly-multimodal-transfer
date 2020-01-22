import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OrthNorm(nn.Module):
    def __init__(self): # dim
        '''
        Builds torch layer that handles orthogonalization of x
        x:      an batch_size x dim input matrix
        dim:    the dim number of the input matrix
        returns:    a torch layer instance. during evaluation, the instance returns an n x d orthogonal matrix
                    if x is full rank and not singular
        '''
        super().__init__()
        # self.ortho_weight_store = nn.Parameter(torch.zeros([dim, dim]))
    
    def orthonorm_op(self, x, epsilon=1e-7):
        '''
        Computes a matrix that orthogonalizes the input matrix x
        x:      an n x d input matrix
        eps:    epsilon to prevent nonzero values in the diagonal entries of x
        returns:    a d x d matrix, ortho_weights, which orthogonalizes x by
                    right multiplication
        '''
        # x_2 = K.dot(K.transpose(x), x)
        # x_2 += K.eye(K.int_shape(x)[1])*epsilon
        # L = tf.cholesky(x_2)
        # ortho_weights = tf.transpose(tf.matrix_inverse(L)) * tf.sqrt(tf.cast(tf.shape(x)[0], dtype=K.floatx()))
        # return ortho_weights
        x_2 = torch.transpose(x, 0, 1) @ x
        x_2 += (torch.eye(x.size(1))* epsilon).cuda()
        L = torch.cholesky(x_2)
        ortho_weights = torch.transpose(torch.inverse(L), 0, 1) * math.sqrt(x.size(0))
        return ortho_weights

    
    def forward(self, x):
        # if self.training:
        #     ortho_weights = self.orthonorm_op(x)
        #     self.ortho_weights_store = ortho_weights
        #     return torch.dot(x, ortho_weights)
        # else:
        #     return torch.dot(x, ortho_weights_store)
        ortho_weights = self.orthonorm_op(x)
        self.ortho_weights_store = ortho_weights
        return x @ ortho_weights
            