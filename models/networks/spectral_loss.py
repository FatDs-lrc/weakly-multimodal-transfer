import torch
import torch.nn as nn

class SpectralLoss(nn.Module):
    ''' Calculate spectral loss 
        L_{spec} = mean(wij * ||Yi - Yj||_2) for each pair in a mini-batch.
    '''
    def __init__(self, adjacent):
        super().__init__()
        self.adjacent = torch.from_numpy(adjacent).cuda().float()
    
    def forward(self, batch_data, batch_indexs):
        ''' batch_data: [batch_size, feat_dim]
        '''
        # batch_size = batch_data.size(0)
        # feat_dim = batch_data.size(1)
        # ai = batch_data.expand(batch_size, batch_size, feat_dim)
        # aj = ai.transpose(0, 1)
        # local_adjacent = self.adjacent[batch_indexs][:, batch_indexs]
        # loss = torch.sum(torch.sqrt(torch.sum((ai-aj)**2, dim=2)) * local_adjacent)
        # return loss / (batch_size * batch_size)
        batch_size = batch_data.size(0)
        feat_dim = batch_data.size(1)
        local_adjacent = self.adjacent[batch_indexs][:, batch_indexs]
        total_loss = torch.as_tensor(0.0).cuda()
        for i in range(batch_size):
            for j in range(batch_size):
                weight = local_adjacent[i, j]
                total_loss += weight * torch.dist(batch_data[i], batch_data[j], p=2)
        return total_loss / (batch_size * batch_size)

class OrthPenalty(nn.Module):
    ''' Calculate orth penalty
        if input batch of feat is Y with size [batch_size, feat_dim]
        L_{orth} = sum(|Y@Y.T - I|) / batch_size**2 
                   where I is a diagonal matrix with size [batch_size, batch_size]
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, batch_data):
        ''' batch_data: [batch_size, feat_dim]
        '''
        # batch_size = batch_data.size(0)
        # I = torch.eye(batch_size).cuda()
        # loss = torch.sum(torch.sqrt(((batch_data @ batch_data.transpose(0, 1)) - I)**2))
        # return loss / (batch_size * batch_size) 
        batch_size = batch_data.size(0)
        orth_panalty_matrix = (batch_data @ batch_data.transpose(0, 1)) - torch.eye(batch_data.size(0)).cuda()
        orth_panalty_matrix = torch.sqrt(orth_panalty_matrix ** 2)
        orth_panalty = torch.sum(orth_panalty_matrix)
        return orth_panalty / (batch_size * batch_size) 