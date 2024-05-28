import torch
import torch.nn as nn
from torch.nn import functional as F
from .outputLayer import OptOutputLayer

class OptNet(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim=16,n_heads=4,blocks=1,bn_blocks=1):
        super(OptNet,self).__init__()
        assert len(in_channels) == blocks and len(out_channels) == blocks,\
        'Error: The len of in_channels and out_channels should be same as blocks'
        d_in_channels = out_channels[::-1]
        d_out_channels = in_channels[::-1]
        self.encoder1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i],out_channels[i],3,1,1),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2)
            )
            for i in range(blocks)
        ])
        self.encoder2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i],out_channels[i],3,1,1),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2)
            )
            for i in range(blocks)
        ])
        self.bottleNeck = nn.Sequential(
            nn.Conv2d(out_channels[-1]*2,512,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,d_in_channels[0],1),
            nn.BatchNorm2d(d_in_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(d_in_channels[i]*3,d_out_channels[i],3,1,1),
                nn.BatchNorm2d(d_out_channels[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(blocks)
        ])
        self.out = OptOutputLayer(in_channels[0],2)

    def forward(self, inp):
        x1 = inp[:,:3]
        x2 = inp[:,3:]
        
        skips1 = []
        for layer in self.encoder1:
            x1 = layer(x1)
            skips1.append(x1)
        skips1 = skips1[::-1]
        
        skips2 = []
        for layer in self.encoder2:
            x2 = layer(x2)
            skips2.append(x2)
        skips2 = skips2[::-1]
        
        x = torch.cat((x1,x2), dim=1)
        x = self.bottleNeck(x)
        for i,layer in enumerate(self.decoder):
            x = layer(torch.cat((x,skips1[i],skips2[i]),dim=1))
        x = self.out(x)
        
        return x