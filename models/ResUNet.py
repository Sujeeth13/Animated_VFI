import torch
import torch.nn as nn
from torch.nn import functional as F
from .encoder import ResEncoder
from .decoder import ResDecoder
from layers import TransformerEncoderBlock
from .outputLayer import OutputLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResUNet(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim=16,n_heads=4,blocks=1,bn_blocks=1):
        super(ResUNet,self).__init__()
        assert len(in_channels) == blocks and len(out_channels) == blocks,\
        'Error: The len of in_channels and out_channels should be same as blocks'
        d_in_channels = out_channels[::-1]
        d_out_channels = in_channels[::-1]
        self.encoder = ResEncoder(in_channels,out_channels,patch_dim,n_heads,num_layers=blocks)
        self.bottleNeck = nn.Sequential(
            nn.Conv2d(out_channels[-1],512,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,d_in_channels[0],1,1),
            nn.BatchNorm2d(d_in_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.decoder = ResDecoder(d_in_channels,d_out_channels,
                                patch_dim//2**(blocks-1),n_heads,num_layers=blocks)
        self.out = OutputLayer(in_channels[0],3)

    def forward(self,x,mask):
        x,skips = self.encoder(x,mask)
        x = self.bottleNeck(x)
        x = self.decoder(x,skips,mask)
        x = self.out(x)
        return x