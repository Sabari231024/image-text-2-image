import torch 
from torch import nn
from torch import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super(VAE_Encoder, self).__init__(
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            VAE_ResidualBlock(128,256),
            VAE_ResidualBlock(256,256),
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),
            VAE_ResidualBlock(256,512),
            VAE_ResidualBlock(512,512),
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            nn.GroupNorm(32,512),
            nn.SiLU(),
            nn.Conv2d(512,8,kernel_size=3,padding=1),
            nn.Conv2d(8,8,kernel_size=1,padding=0),

        )


    def forward(self,x:torch.Tensor, noise:torch.Tensor)->torch.Tensor:
        # x: (Batch_size,Channel,Height,Width)
        # noise: (Batch_size,Channel,Height/8,Width/8)
        
        for module in self:
            if getattr(module,'stride',None) == (2,2): # i am manually doing the padding 
                # (Padding_Left,Padding_Right,Padding_Top,Padding_Bottom)
                x = F.pad(x,(0,1,0,1))
            x = module(x)
        mean , logvar = torch.chunk(x,2,dim=1)#dividing the output into mean and logvar over the channel dimension(two parts)
        logvar = torch.clamp(logvar,-30,20)#clamping the logvar to a range to prevent very small values
        var = logvar.exp()
        std = var.sqrt()
        z = mean + std * noise # N(0,1)->N(mean,var)
        z *= 0.18215 #scale by constant
        return z
