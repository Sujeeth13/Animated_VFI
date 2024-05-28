import torch
import torch.nn as nn
from torch.nn import functional as F

class Upsample(nn.Module):
    def __init__(self,channels):
        super(Upsample,self).__init__()
        self.conv = nn.Conv2d(channels,channels,3,padding=1)
    
    def forward(self,x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self,channels):
        super(Downsample,self).__init__()
        self.conv = nn.Conv2d(channels,channels,3,stride=2,padding=1)
    
    def forward(self,x):
        return self.conv(x)

class ConvBlock(nn.Module):
    '''This block is used for downsampling or feature extraction '''
    def __init__(self,in_channels,out_channels,kernel_size=3,
                    stride=1,padding=1,num_layers=1):
        super(ConvBlock,self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            if i == 0
            else
            nn.Sequential(
                nn.Conv2d(out_channels,out_channels,kernel_size,stride,padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for i in range(num_layers)
        ])
    
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == '__main__':
    # Unit tests
    print("Test 1: ConvBlock (size doesn't change)")
    try:
        x = torch.rand(2,4,256,256)
        print(f"Shape before conv block: {x.shape}")
        conv = ConvBlock(4,64)
        x = conv(x)
        print(f"Shape after conv block: {x.shape}")
        print("Test 1 passed\n")
    except:
        print("Test 1 failed\n")
    
    print("Test 2: ConvBlock (size is halved using stride 2)")
    try:
        x = torch.rand(2,4,256,256)
        print(f"Shape before conv block: {x.shape}")
        conv = ConvBlock(4,64,stride=2)
        x = conv(x)
        print(f"Shape after conv block: {x.shape}")
        print("Test 2 passed\n")
    except:
        print("Test 2 failed\n")

    print("Test 3: ConvBlock (size is made 1/4 using stride 2 and 2 layers)")
    try:
        x = torch.rand(2,4,256,256)
        print(f"Shape before conv block: {x.shape}")
        conv = ConvBlock(4,64,stride=2,num_layers=2)
        x = conv(x)
        print(f"Shape after conv block: {x.shape}")
        print("Test 3 passed\n")
    except:
        print("Test 3 failed\n")

    print("Test 4: Upsample")
    try:
        x = torch.rand(2,4,256,256)
        print(f"Shape before Upsample: {x.shape}")
        up = Upsample(4)
        x = up(x)
        print(f"Shape after Upsample: {x.shape}")
        print("Test 4 passed\n")
    except:
        print("Test 4 failed\n")

    print("Test 5: Downsample")
    try:
        x = torch.rand(2,4,256,256)
        print(f"Shape before Downsample: {x.shape}")
        down = Downsample(4)
        x = down(x)
        print(f"Shape after Downsample: {x.shape}")
        print("Test 5 passed\n")
    except:
        print("Test 5 failed\n")
    
