import torch
import torch.nn as nn
import torch.nn.functional as F

class classifier(nn.Module):
    def __init__(self, target, input_dim):
        super(classifier, self).__init__()

        if target == 'discrete':
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(True),
                nn.Linear(64,16),
                nn.ReLU(True),
                nn.Linear(16,4),
                nn.ReLU(True),
                #nn.Softmax()
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(256,64),
                nn.ReLU(True),
                nn.Linear(64,16),
                nn.ReLU(True),
                nn.Linear(16,3),
                nn.ReLU(True),
                # nn.Softmax()
            )
 
    def forward(self, x):
        pred = self.net(x)
        return pred

if __name__ == '__main__':
    def hook(module, input, output):
        features.copy_(output.data.squeeze())
        get_feature = True

    c = classifier('discrete', 512)
    handel = c.net[3].register_forward_hook(hook)
    print(c)