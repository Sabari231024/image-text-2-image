import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self,n_heads:int,d_embed:int,in_proj_bias:bool=True,out_proj_bias:bool=True):
        super(SelfAttention,self).__init__()
        self.in_proj = nn.Linear(d_embed,d_embed*3,bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed,d_embed,bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self,x:torch.Tensor,casual_mask:bool=False)->torch.Tensor:
        input_shape = x.shape
        batch_size , seq_len , d_embed = input_shape 
        interm_shape = (batch_size,seq_len,self.n_heads,self.d_head)
        q,k,v = self.in_proj(x).chunk(3,dim=-1) # (batch_size,seq_len,d_embed)->(batch_size,seq_len,d_embed*3)-> three differnt tensors of shape (batch_size,seq_len,d_head)
        q = q.view(interm_shape).transpose(1,2)
        k = k.view(interm_shape).transpose(1,2)
        v = v.view(interm_shape).transpose(1,2)
        weight = q @ k.transpose(-1,-2)
        if casual_mask:
            mask = torch.ones_like(weight,dtype=torch.bool).triu(1) #mask upper triangle made of 1s
            weight.masked_fill_(mask,-float('inf'))
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight,dim=-1)
        x = weight @ v
        x = x.transpose(1,2)
        x = x.reshape(input_shape)
        x = self.out_proj(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self,n_heads:int,d_embed:int,d_cross:int,in_proj_bias:bool=True,out_proj_bias:bool=True):
        super(CrossAttention,self).__init__()
        self.q_proj = nn.Linear(d_embed,d_embed,bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross,d_embed,bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross,d_embed,bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed,d_embed,bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self,x,y): 
        # x: (latent:query),:: (batch_size,seq_len_q,dim_q)
        # y: (context,key and value)::(batch_size,seq_len_kv,dim_kv) = (Batch_size,77,768)
        input_shape = x.shape
        batch_size, sequence_length,d_embed = input_shape
        interim_shape = (batch_size,-1,self.n_heads,self.d_head)
        q = self.q_proj(x).view(interim_shape).transpose(1,2)
        k = self.k_proj(y).view(interim_shape).transpose(1,2)
        v = self.v_proj(y).view(interim_shape).transpose(1,2) 
        weight = q @ k.transpose(-1,-2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight,dim=-1)
        output = weight @ v
        output = output.transpose(1,2).contiguous().view(input_shape)
        output = self.out_proj(output)
        return output




