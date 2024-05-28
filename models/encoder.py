import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import deque
from layers import BasicBlock, TransformerEncoderBlock, Downsample, AttentionModule

class ResEncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim,n_heads,stride,padding,resnet_bias
                    ,dim_ff,num_layers):
        super(ResEncoderBlock,self).__init__()
        self.p = patch_dim*patch_dim
        self.resnet = BasicBlock(in_channels,out_channels,stride,padding,resnet_bias)
        self.downsample = Downsample(out_channels)
        self.attention = AttentionModule(1)
        
    def forward(self, x, mask):
        x = self.resnet(x)
        x = self.attention(x,mask)
        resO = x
        tranO = x
        x = self.downsample(x)
        return x, resO, tranO

class ResEncoder(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim,n_heads,stride=1,padding=1,resnet_bias=False
                    ,dim_ff=2048,t_layers=1,num_layers=1):
        super(ResEncoder,self).__init__()
        assert len(in_channels) == num_layers and len(out_channels) == num_layers,\
        'Error: The len of in_channels and out_channels should be same as num_layers'
        self.skips = deque()
        self.layers = nn.ModuleList([
            ResEncoderBlock(in_channels[i],out_channels[i],patch_dim//2**i,n_heads,stride,padding,
                            resnet_bias,dim_ff,t_layers)
            if 2**i <= patch_dim
            else
            ResEncoderBlock(in_channels[i],out_channels[i],1,n_heads,stride,padding,
                resnet_bias,dim_ff,t_layers)
            for i in range(num_layers)
        ])
    def forward(self, x, mask):
        for layer in self.layers:
            x, resO, tranO = layer(x, mask)
            self.skips.append((resO, tranO))
        return x, self.skips
    
class TransEncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim,n_heads,stride,padding,resnet_bias
                    ,dim_ff,num_layers):
        super(TransEncoderBlock,self).__init__()
        self.p = patch_dim*patch_dim
        self.resnet = BasicBlock(in_channels,out_channels,stride,padding,resnet_bias)
        self.transformer = TransformerEncoderBlock(self.p*out_channels,
                                                    n_heads,dim_ff,num_layers)
        self.downsample = Downsample(out_channels)
        
    def forward(self, x):
        x = self.resnet(x)
        resO = x
        b,c,h,w = x.shape
        x = x.view(b,c*self.p,h*w//self.p).permute(0,2,1)
        x = self.transformer(x)
        tranO = x
        x = x.permute(0,2,1).reshape(b,c,h,w)
        x = self.downsample(x)
        return x, resO, tranO

class TransEncoder(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim,n_heads,stride=1,padding=1,resnet_bias=False
                    ,dim_ff=2048,t_layers=1,num_layers=1):
        super(TransEncoder,self).__init__()
        assert len(in_channels) == num_layers and len(out_channels) == num_layers,\
        'Error: The len of in_channels and out_channels should be same as num_layers'
        self.skips = deque()
        self.layers = nn.ModuleList([
            TransEncoderBlock(in_channels[i],out_channels[i],patch_dim//2**i,n_heads,stride,padding,
                            resnet_bias,dim_ff,t_layers)
            if 2**i <= patch_dim
            else
            TransEncoderBlock(in_channels[i],out_channels[i],1,n_heads,stride,padding,
                resnet_bias,dim_ff,t_layers)
            for i in range(num_layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x, resO, tranO = layer(x)
            self.skips.append((resO, tranO))
        return x, self.skips