import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(494408,768,77)
        self.layers = nn.Module([
            CLIPLayer(12,768) for i in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self,tokens:torch.LongTensor)->torch.Tensor:
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens) # (Batch_size,Seq_len,Embedding_dim)
        for layer in self.layers:
            state = layer(state)
        state = self.layernorm(state)
        return state

class CLIPEmbedding(nn.Module):
    def __init__(self,n_vocab:int,n_embed:int,n_tokens:int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab,n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens,n_embed))
    def forward(self,tokens):
        x = self.token_embedding(tokens) + self.position_embedding
        return x

class CLIPLayer(nn.Module):
    def __init__(self,n_heads:int,n_embed:int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads,n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed,n_embed*4)
        self.linear_2 = nn.Linear(n_embed*4,n_embed)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        residual = x
        x = self.layernorm_1(x)
        x = self.attention(x,casual_mask=True)
        x += residual
        residual = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702*x)  #QuickGelu
        x = self.linear_2(x)
        x += residual
        return x
        