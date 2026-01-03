import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention,CrossAttention

class SwitchSequential(nn.Sequential):
    def forward(self,x:torch.Tensor,context:torch.Tensor,time:torch.Tensor)->torch.Tensor:
        for layer in self:
            if isinstance(layer,UNET_ResidualBlock):
                x = layer(x,time)
            elif isinstance(layer,UNET_AttentionBlock):
                x = layer(x,context) # we will compute cross attention between context and x
            else:
                x = layer(x)
        return x

class UNET_ResidualBlock(nn.Module):
    def ___init__(self,in_channels:int,out_channels:int,n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32,in_channels)
        self.conv_feature = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.linear_time = nn.Linear(n_time,out_channels)
        self.groupnorm_merged = nn.GroupNorm(32,out_channels)
        self.conv_merged = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if in_channels==out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
    
    def forward(self,feature,time):
        #feature the latewnt : (Batch,in channels,H,W)
        #time : (1,1280)
        residual = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        residual = self.residual_layer(residual)
        return merged + residual

class UNET_AttentionBlock(nn.Module):
    def ___init__(self,n_heads:int,n_embed:int,d_context:int):
        super(UNET_AttentionBlock,self).__init__()
        channels = n_heads * n_embed
        self.groupnorm = nn.GroupNorm(32,channels,eps=1e-6)
        self.conv_input = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads,channels,in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads,channels,d_context,in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels,4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels,channels)
        self.conv_output = nn.Conv2d(channels,channels,kernel_size=1,padding=0)

    def forward(self,x,context):
        residual_long= x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n,c,h,w = x.shape
        x = x.view((n,c,h*w))
        x = x.transpose(-1,-2)
        residual_short = x
        # Norm + SelfAttention
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residual_short
        residual_short = x
        # Norm + Cross Attention
        x = self.layernorm_2(x)
        x = self.attention_2(x,context)
        x += residual_short
        residual_short = x
        # Norm + Feed Forward
        x = self.layernorm_3(x)
        x,gate = self.linear_geglu_1(x).chunk(2,dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residual_short
        x = x.transpose(-1,-2)
        x = x.view((n,c,h,w))
        return self.conv_output(x) + residual_long




class UpSample(nn.Module):
    def __init__(self,channels:int):
        super(UpSample,self).__init__()
        self.conv = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = F.interpolate(x,scale_factor=2,mode='nearest')
        x = self.conv(x)
        return x

class UNET_OutputLayer(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super(UNET_OutputLayer,self).__init__()
        self.groupnorm = nn.GroupNorm(32,in_channels)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x  # channels : 320->4
 
class UNET(nn.Module):
    def __init__(self):
        super(UNET,self).__init__()
        self.encoders = nn.Module([
            SwitchSequential(nn.Conv2d(4,320,kernel_size=3,padding=1)),
            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttentionBlock(8,40)),
            SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(UNET_ResidualBlock(320,640),UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(640,640),UNET_AttentionBlock(8,80)),
            SwitchSequential(nn.Conv2d(640,640,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(UNET_ResidualBlock(640,1280),UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(1280,1280),UNET_AttentionBlock(8,160)),
            SwitchSequential(nn.Conv2d(1280,1280,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280,1280)),
            SwitchSequential(UNET_ResidualBlock(1280,1280)),

        ])
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280,1280),
            UNET_AttentionBlock(8,160),
            UNET_ResidualBlock(1280,1280), 
        )
        self.decoder = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280),UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280),UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(2560,1280),UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(1920,1280),UNET_AttentionBlock(8,160),UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920,640),UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(1280,640),UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(960,640),UNET_AttentionBlock(8,80),UpSample(640)),
            SwitchSequential(UNET_ResidualBlock(960,320),UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(640,320),UNET_AttentionBlock(8,40)),#check this
            SwitchSequential(UNET_ResidualBlock(640,320),UNET_AttentionBlock(8,40)),
        ])