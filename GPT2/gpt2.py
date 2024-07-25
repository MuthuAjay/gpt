import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP


# hyperparameters
batch_size = 64
block_size = 256
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        
    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    
    def __ini__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        
        
class CasualSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # not really  a bias, more of a mask, but following the OpenAI/Hugging Face naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1,config.block_size, config.block_size))
        
    def forward(self, x: torch.Tensor):
        B,T,C = x.size() # batch, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dimension
        # nh is 'number of heads' , hs is 'head size' and C is 'embedding dimension'
        # e.g in GPT-2 (124M), n_head = 12, hs=64, so nh*hs =C=768 channels in the transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        q= q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        v= v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        
        # attention (materialize the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.c_proj(y)
        return y
        
    

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size : int = 65
    n_layer : int = 6
    n_head : int = 6
    n_embd : int = 384
    
class GPT(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_layer % config.n_head == 0
        assert config.vocab_size is not None, "Please specify vocab size"
        assert config.block_size is not None, "Please specify block size"
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        
        
        
    
    
    
        
