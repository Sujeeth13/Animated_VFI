import torch
import torch.nn as nn
from torch.nn import functional as F
from layers import BasicBlock, TransformerDecoderBlock, Upsample, AttentionModule

class ResDecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim,n_heads,stride,padding,resnet_bias
                    ,dim_ff,num_layers):
        super(ResDecoderBlock,self).__init__()
        self.p = patch_dim*patch_dim
        self.resnet = BasicBlock(2*in_channels,out_channels,stride,padding,resnet_bias)
        self.upsample = Upsample(in_channels)
        self.attention = AttentionModule(1)
    
    def forward(self, x, context, mask):
        x = self.upsample(x)
        r_context,_ = context
        x = self.resnet(torch.cat((x,r_context),dim=1))
        x = self.attention(x,mask)
        return x

class ResDecoder(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim,n_heads,stride=1,padding=1,resnet_bias=False
                    ,dim_ff=2048,t_layers=1,num_layers=1):
        super(ResDecoder,self).__init__()
        assert len(in_channels) == num_layers and len(out_channels) == num_layers,\
        'Error: The len of in_channels and out_channels should be same as num_layers'
        self.layers = nn.ModuleList([
            ResDecoderBlock(in_channels[i],out_channels[i],patch_dim*2**i,n_heads,stride,padding,
                            resnet_bias,dim_ff,t_layers)
            for i in range(num_layers)
        ])
    def forward(self, x, skips, mask):
        for i,layer in enumerate(self.layers):
            x = layer(x,skips.pop(), mask)
        return x

class TransDecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim,n_heads,stride,padding,resnet_bias
                    ,dim_ff,num_layers):
        super(TransDecoderBlock,self).__init__()
        self.p = patch_dim*patch_dim
        self.resnet = BasicBlock(2*in_channels,out_channels,stride,padding,resnet_bias)
        self.transformer = TransformerDecoderBlock(self.p*in_channels,n_heads,dim_ff,num_layers)
        self.upsample = Upsample(in_channels)
    
    def forward(self, x, context):
        x = self.upsample(x)
        b,c,h,w = x.shape
        x = x.reshape(b,c*self.p,h*w//self.p).permute(0,2,1)
        r_context,t_context = context
        x = self.transformer(x,t_context)
        x = x.permute(0,2,1).reshape(b,c,h,w)
        x = self.resnet(torch.cat((x,r_context),dim=1))
        return x

class TransDecoder(nn.Module):
    def __init__(self,in_channels,out_channels,patch_dim,n_heads,stride=1,padding=1,resnet_bias=False
                    ,dim_ff=2048,t_layers=1,num_layers=1):
        super(TransDecoder,self).__init__()
        assert len(in_channels) == num_layers and len(out_channels) == num_layers,\
        'Error: The len of in_channels and out_channels should be same as num_layers'
        self.layers = nn.ModuleList([
            TransDecoderBlock(in_channels[i],out_channels[i],patch_dim*2**i,n_heads,stride,padding,
                            resnet_bias,dim_ff,t_layers)
            for i in range(num_layers)
        ])
    def forward(self, x, skips):
        for i,layer in enumerate(self.layers):
            x = layer(x,skips.pop())
        return x