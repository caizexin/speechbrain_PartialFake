import torch 
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(planes)
        self.batch_norm2 = nn.BatchNorm1d(planes)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual
    
    
class MelResNet(nn.Module):
    def __init__(self, in_planes, planes, embed_dim=128, res_blocks=8):
        super().__init__()
        self.conv_in = nn.Conv1d(in_planes, planes, kernel_size=5, bias=False, padding=2)
        self.batch_norm = nn.BatchNorm1d(planes)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(planes))
        self.conv_out = nn.Conv1d(planes, embed_dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers: x = f(x)
        x = self.conv_out(x)
        return x.transpose(1, 2)
    
    