import torch

batch_data = torch.tensor([[0,0], [0,0], [0,0], [0,0]]).float()
batch_size = batch_data.size(0)
feat_dim = batch_data.size(1)
ai = batch_data.expand(batch_size, batch_size, feat_dim)
aj = ai.transpose(0, 1)
# local_adjacent = self.adjacent[batch_indexs][:, batch_indexs]
local_adjacent = torch.ones([4, 4])
loss = torch.sum(torch.sqrt(torch.sum((ai-aj)**2, dim=2)) * local_adjacent)
print(loss)

batch_size = batch_data.size(0)
feat_dim = batch_data.size(1)
# local_adjacent = self.adjacent[batch_indexs][:, batch_indexs]
total_loss = torch.as_tensor(0.0).cuda()
for i in range(batch_size):
    for j in range(batch_size):
        weight = local_adjacent[i, j]
        total_loss += weight * torch.dist(batch_data[i], batch_data[j], p=2)
print(total_loss)