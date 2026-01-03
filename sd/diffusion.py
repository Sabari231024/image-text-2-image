import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention,CrossAttention
from unet import UNET,UNET_OutputLayer
class TimeEmbedding(nn.Module):
    def __init__(self,n_embed:int):
        super(TimeEmbedding,self).__init__()
        self.linear_1 = nn.Linear(n_embed,n_embed*4)
        self.linear_2 = nn.Linear(n_embed*4,4*n_embed)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = F.silu(self.linear_1(x))
        x = self.linear_2(x)
        return x    


class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)
    
    def forward(self,latent:torch.Tensor,context:torch.Tensor,time:torch.Tensor)->torch.Tensor:
        time = self.time_embedding(time) #it is like number with sin and cosine for time information like positional information
        #time : (1,1280)
        output = self.unet(latent,context,time) #(batch,4,h/8,w/8)->(batch,320,h/8,w/8)
        output = self.final(output) #(batch,320,h/8,w/8)->(batch,4,h/8,w/8)
        return output




