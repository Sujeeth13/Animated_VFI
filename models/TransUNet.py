import torch
import torch.nn as nn
from torch.nn import functional as F
from .encoder import TransEncoder
from .decoder import TransDecoder
from layers import TransformerEncoderBlock
from .outputLayer import OutputLayer

class TransUNet(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim=16,n_heads=4,blocks=1,bn_blocks=1):
        super(TransUNet,self).__init__()
        assert len(in_channels) == blocks and len(out_channels) == blocks,\
        'Error: The len of in_channels and out_channels should be same as blocks'
        d_in_channels = out_channels[::-1]
        d_out_channels = in_channels[::-1]
        self.encoder = TransEncoder(in_channels,out_channels,patch_dim,n_heads,num_layers=blocks)
        self.bottleNeck = TransformerEncoderBlock(out_channels[-1],n_heads,bn_blocks)
        self.decoder = TransDecoder(d_in_channels,d_out_channels,
                                patch_dim//2**(blocks-1),n_heads,num_layers=blocks)
        self.out = OutputLayer(in_channels[0],3)

    def forward(self,x):
        x,skips = self.encoder(x)
        b,c,h,w = x.shape
        x = x.view(b,c,h*w).permute(0,2,1)
        x = self.bottleNeck(x)
        x = x.permute(0,2,1).reshape(b,c,h,w)
        x = self.decoder(x,skips)
        x = self.out(x)
        return x