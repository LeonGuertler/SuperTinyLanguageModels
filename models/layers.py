import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config['arch']["hidden_dim"] % config['arch']["num_heads"] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config['arch']["hidden_dim"], 3 * config['arch']["hidden_dim"], bias=config['arch']["bias"])
        # output projection
        self.c_proj = nn.Linear(config['arch']["hidden_dim"], config['arch']["hidden_dim"], bias=config['arch']["bias"])
        # regularization
        self.attn_dropout = nn.Dropout(config['arch']["dropout"])
        self.resid_dropout = nn.Dropout(config['arch']["dropout"])
        self.n_head = config['arch']["num_heads"]
        self.n_embd = config['arch']["hidden_dim"]
        self.dropout = config['arch']["dropout"]
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class FFN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config['arch']["hidden_dim"], config['arch']["mlp_dim"], bias=config['arch']["bias"])
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config['arch']["mlp_dim"], config['arch']["hidden_dim"], bias=config['arch']["bias"])
        self.dropout = nn.Dropout(config['arch']["dropout"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class FFN_with_LoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config['arch']["hidden_dim"], config['arch']["mlp_dim"], bias=config['arch']["bias"])
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config['arch']["mlp_dim"], config['arch']["hidden_dim"], bias=config['arch']["bias"])
        self.dropout = nn.Dropout(config["dropout"])

        self.lora_down_proj = nn.Linear(config['arch']["hidden_dim"], config['arch']["eval_iters"], bias=config['arch']["bias"])
        self.lora_up_proj = nn.Linear(config['arch']["hidden_dim"], config['arch']["eval_iters"], bias=config['arch']["bias"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        # LoRA
        down = self.lora_down_proj(x)
        down = self.gelu(down)
        up = self.lora_up_proj(down)
        
        return x+up
