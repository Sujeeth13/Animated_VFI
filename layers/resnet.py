import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    """ This is the basic resnet block used for feature extraction """
    def __init__(self,in_channels,out_channels,stride=1,padding=1,resnet_bias=False):
        super(BasicBlock,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,stride,padding,bias=resnet_bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,stride=1,padding=1,bias=resnet_bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels,out_channels,1,stride,bias=resnet_bias)
    
    def forward(self, x):
        residue = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + self.residual(residue)
        x = self.relu(x)
        return x

if __name__ == "__main__":
    # Unit tests
    print("Test 1: with stride 1")
    try:
        x = torch.rand(2,4,256,256)
        print(f"Shape before feature extraction block: {x.shape}")
        block = BasicBlock(4,64)
        x = block(x)
        print(f"Shape after feature extraction block: {x.shape}")
        print("Test 1 passed")
    except:
        print("Test 1 failed")
    
    print("Test 2: with stride 2")
    try:
        x = torch.rand(2,4,256,256)
        print(f"Shape before feature extraction block: {x.shape}")
        block = BasicBlock(4,64,2)
        x = block(x)
        print(f"Shape after feature extraction block: {x.shape}")
        print("Test 2 passed")
    except:
        print("Test 2 failed")
    
    print("Test 3: in_channels = out_channels")
    try:
        x = torch.rand(2,4,256,256)
        print(f"Shape before feature extraction block: {x.shape}")
        block = BasicBlock(4,4)
        x = block(x)
        print(f"Shape after feature extraction block: {x.shape}")
        print("Test 3 passed")
    except:
        print("Test 3 failed")