import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionModule(nn.Module):
    def __init__(self, feature_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, mask):
        mask_resized = F.interpolate(mask, size=(features.size(2), features.size(3)), mode='bilinear', align_corners=True).float()
        attention = self.sigmoid(self.conv(mask_resized))
        return features * attention

class SelfAttentionBlock(nn.Module):
    ''' 
        This block is used from the encoder block 
        from the Attention is all you need paper
    '''
    def __init__(self,embed_dim,n_heads,dim_ff):
        super(SelfAttentionBlock,self).__init__()
        self.heads = n_heads
        self.embed_dim = embed_dim
        self.self_attention = nn.MultiheadAttention(embed_dim,n_heads,batch_first=True)
        self.linear1 = nn.Linear(embed_dim,dim_ff)
        self.linear2 = nn.Linear(dim_ff,embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self,x):
        short_residue = x
        x,_ = self.self_attention(x,x,x,need_weights=False)
        x += short_residue
        x = self.layernorm1(x)

        short_residue = x
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x += short_residue
        x = self.layernorm2(x)
        return x

class CrossAttentionBlock(nn.Module):
    ''' 
        This block is used from the decoder block 
        from the Attention is all you need paper
    '''
    def __init__(self,embed_dim,n_heads,dim_ff):
        super(CrossAttentionBlock,self).__init__()
        self.heads = n_heads
        self.embed_dim = embed_dim
        self.self_attention = nn.MultiheadAttention(embed_dim,n_heads,batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim,n_heads,batch_first=True)
        self.linear1 = nn.Linear(embed_dim,dim_ff)
        self.linear2 = nn.Linear(dim_ff,embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)

    def forward(self,x,context):
        short_residue = x
        x,_ = self.self_attention(x,x,x,need_weights=False)
        x += short_residue
        x = self.layernorm1(x)

        short_residue = x
        x,_ = self.cross_attention(x,context,context,need_weights=False)
        x += short_residue
        x = self.layernorm2(x)

        short_residue = x
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x += short_residue
        x = self.layernorm3(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self,embed_dim,n_heads,dim_ff=2048,num_layers=1):
        super(TransformerEncoderBlock,self).__init__()
        self.layers = nn.ModuleList([SelfAttentionBlock(embed_dim,n_heads,dim_ff) 
                                        for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self,embed_dim,n_heads,dim_ff=2048,num_layers=1):
        super(TransformerDecoderBlock,self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([CrossAttentionBlock(embed_dim,n_heads,dim_ff)
                                        for _ in range(num_layers)])

    def forward(self, x, context):
        for layer in self.layers:
            x = layer(x,context)
        return x

if __name__ == '__main__':
    # Unit tests
    print("Test 1: SelfAttentionBlock")
    try:
        x = torch.rand(2,20,768)
        print(f"Shape before self attention block: {x.shape}")
        sa_attn = SelfAttentionBlock(768,4)
        x = sa_attn(x)
        print(f"Shape after self attention block: {x.shape}")
        print("Test 1 passed\n")
    except:
        print("Test 1 failed\n")
    
    print("Test 2: CrossAttentionBlock")
    try:
        x = torch.rand(2,20,768)
        y = torch.rand(2,20,768)
        print(f"Shape before cross attention block: {x.shape}")
        cross_attn = CrossAttentionBlock(768,4)
        x = cross_attn(x,y)
        print(f"Shape after cross attention block: {x.shape}")
        print("Test 2 passed\n")
    except:
        print("Test 2 failed\n")

    print("Test 3: TransformerEncoderBlock")
    try:
        x = torch.rand(2,20,768)
        print(f"Shape before transformer encoder block: {x.shape}")
        encoder = TransformerEncoderBlock(768,4,num_layers=5)
        x = encoder(x)
        print(f"Shape after transformer encoder block: {x.shape}")
        print("Test 3 passed\n")
    except:
        print("Test 3 failed\n")
    
    print("Test 4: TransformerEncoderBlock")
    try:
        x = torch.rand(2,20,768)
        y = torch.rand(2,20,768)
        print(f"Shape before transformer decoder block: {x.shape}")
        decoder = TransformerDecoderBlock(768,4,num_layers=5)
        x = decoder(x,y)
        print(f"Shape after transformer decoder block: {x.shape}")
        print("Test 4 passed\n")
    except:
        print("Test 4 failed\n")