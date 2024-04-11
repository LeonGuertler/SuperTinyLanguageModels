from models.architectures import baseline
from models import layers

import torch
import torch.nn as nn

class LoraFNN(nn.Module):
    """
    A simple Feed Forward Network block.
    """
    def __init__(self, hidden_dim, ffn_dim, bias=False, dropout=0.0,
        rank=32, lora_weighting=1
        ):
        super().__init__()
        self.c_fc = layers.LoraLinear(
            hidden_dim,
            ffn_dim,
            bias=bias,
            rank=rank,
            alpha=lora_weighting
        )

        self.gelu = nn.GELU()
        self.c_proj = layers.LoraLinear(
            ffn_dim,
            hidden_dim,
            bias=bias,
            rank=rank,
            alpha=lora_weighting

        )
        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):
        """
        Forward pass
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class LoraBlock(nn.Module):
    """
    A simple abstraction to combine the 
    LayerNorms, SelfAttention and FeedForward layers
    """
    def __init__(self, hidden_dim, ffn_dim, bias, num_heads, dropout, rank=32, lora_weighting=1):
        super().__init__()
        self.ln_1 = layers.LayerNorm(hidden_dim, bias=bias)
        self.attn = layers.CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
        )
        self.ln_2 = layers.LayerNorm(hidden_dim, bias=bias)
        self.mlp = LoraFNN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            bias=bias,
            dropout=dropout,
            rank=rank,
            lora_weighting=lora_weighting
        )

    def forward(self, x, attention_mask=None):
        """
        A simple, residual forward 
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
        """
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class SharedFNNLora(baseline.BaseGPT):
    def build_transformer(self):
        
        transformers = nn.ModuleDict(dict(
            drop=nn.Dropout(self.cfg["dropout"]),
            h=nn.ModuleList(
                [LoraBlock(
                    hidden_dim=self.cfg["hidden_dim"], 
                    ffn_dim=self.cfg["ffn_dim"], 
                    bias=self.cfg["bias"], 
                    num_heads=self.cfg["num_heads"], 
                    dropout=self.cfg["dropout"],
                    rank=self.cfg["rank"],
                    lora_weighting=self.cfg["lora_weighting"]
                ) for _ in range(self.cfg["depth"])]
            )
        ))
        for block in transformers.h:
            block.mlp.c_fc.weight = transformers.h[0].mlp.c_fc.weight
            block.mlp.c_fc.bias = transformers.h[0].mlp.c_fc.bias
            block.mlp.c_proj.weight = transformers.h[0].mlp.c_proj.weight
            block.mlp.c_proj.bias = transformers.h[0].mlp.c_proj.bias
        return transformers