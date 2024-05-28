import torch
import torch.nn as nn
from torch.nn import functional as F
from .outputLayer import OutputLayer

class UNet(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim=16,n_heads=4,blocks=1,bn_blocks=1):
        super(UNet,self).__init__()
        assert len(in_channels) == blocks and len(out_channels) == blocks,\
        'Error: The len of in_channels and out_channels should be same as blocks'
        d_in_channels = out_channels[::-1]
        d_out_channels = in_channels[::-1]
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i],out_channels[i],3,1,1),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2)
            )
            for i in range(blocks)
        ])
        self.bottleNeck = nn.Sequential(
            nn.Conv2d(out_channels[-1],512,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,d_in_channels[0],1),
            nn.BatchNorm2d(d_in_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(2*d_in_channels[i],d_out_channels[i],3,1,1),
                nn.BatchNorm2d(d_out_channels[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(blocks)
        ])
        self.out = OutputLayer(in_channels[0],3)

    def forward(self,x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        skips = skips[::-1]
        x = self.bottleNeck(x)
        for i,layer in enumerate(self.decoder):
            x = layer(torch.cat((x,skips[i]),dim=1))
        x = self.out(x)
        return x