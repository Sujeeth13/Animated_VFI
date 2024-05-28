import torch
import torch.nn as nn
from torch.nn import functional as F

class OutputLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutputLayer,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        return self.sigmoid(self.conv(x))

class OptOutputLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OptOutputLayer,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,1)
        self.conv2 = nn.Conv2d(in_channels,out_channels,1)
    
    def forward(self,x):
        return torch.cat((self.conv1(x), self.conv2(x)), dim=1)