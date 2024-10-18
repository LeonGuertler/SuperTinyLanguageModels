"""
A collection of different model heads. 
"""

import torch
import torch.nn as nn 
import math 
from models.components.normalization import build_normalization


class AutoregressiveLMHead(torch.nn.Module):
    """
    Generic autoregressive language model head.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.layer_norm = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )
        self.linear = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["vocab_size"],
            bias=model_cfg["lm_head_bias"],
        )
        self.dropout = torch.nn.Dropout(
            p=model_cfg.get("lm_head_dropout", 0.0) # Default is no Dropout
        )

    def forward(self, x):
        """
        Pass the input through the model.
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, V)
        """

        # apply layer norm
        x = self.layer_norm(x)

        # apply dropout if necessary
        x = self.dropout(x)

        # pass through the linear layer
        x = self.linear(x)

        return x, None

    def inference(self, x):
        """
        Pass the input through the model, then
        Return the final token logits
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, V)
        """
        return self.forward(x[:, -1, :])[0]



class ClassificationLMHead(torch.nn.Module):
    """ TODO """

    def __init__(self, model_cfg):
        super().__init__()
        self.layer_norm = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )
        self.linear = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["lm_head_num_classes"],
            bias=model_cfg["lm_head_bias"],
        )
        self.dropout = torch.nn.Dropout(
            p=model_cfg.get("lm_head_dropout", 0.0) # Default is no Dropout
        )

    def forward(self, x):
        """
        Pass the input through the model.
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, V)
        """

        # only use the final token
        x = x[:, -1, :]

        # apply layer norm
        x = self.layer_norm(x)

        # apply dropout if necessary
        x = self.dropout(x)

        # pass through the linear layer
        x = self.linear(x)

        return x, None


class AttentionLMHead(torch.nn.Module):
    """ 
    TODO  
        - shouldn't re-calculate embeddings for accumulation_steps
    """
    def __init__(self, model_cfg):
        super().__init__()
        vocab_size = model_cfg["vocab_size"]

        self.q_proj = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["hidden_dim"],
            bias=False
        )

        self.k_proj = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["hidden_dim"],
            bias=False
        )

        #self.v = torch.nn.Identity()

        self.layer_norm = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )

        self.layer_norm2 = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )


    def forward(self, x, embedding_model):
        """ TODO """
        q = self.q_proj(self.layer_norm(x))
        k = self.k_proj(self.layer_norm2(embedding_model.token_embedder.weight)).unsqueeze(0)
        y = torch.matmul(q, k.transpose(1,2)) / math.sqrt(q.size(-1))
        return y, None


# class AttentionLMHead(torch.nn.Module):
#     """ 
#     TODO  
#         - shouldn't re-calculate embeddings for accumulation_steps
#     """
#     def __init__(self, model_cfg):
#         super().__init__()
#         vocab_size = model_cfg["vocab_size"]
#         depth = 2

#         self.q_proj = torch.nn.Linear(
#             in_features=model_cfg["hidden_dim"],
#             out_features=256,
#             bias=False
#         )

#         self.neg_q_proj = torch.nn.Linear(
#             in_features=model_cfg["hidden_dim"],
#             out_features=256,
#             bias=False
#         )

#         self.k_proj = torch.nn.Linear(
#             in_features=model_cfg["hidden_dim"],
#             out_features=256,
#             bias=False
#         )

#         #self.v = torch.nn.Identity()

#         self.layer_norm = build_normalization(
#             normalization_name=model_cfg["lm_head_normalization"],
#             dim=model_cfg["hidden_dim"],
#             bias=model_cfg["lm_head_bias"],
#         )

#         # lambda
#         self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth) # from the paper
#         self.lambda_q1 = torch.nn.Parameter(torch.zeros(256, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_k1 = torch.nn.Parameter(torch.zeros(256, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_q2 = torch.nn.Parameter(torch.zeros(256, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_k2 = torch.nn.Parameter(torch.zeros(256, dtype=torch.float32).normal_(mean=0,std=0.1))



#     def forward(self, x, embedding_model):
#         """ TODO """
#         x = self.layer_norm(x)

#         q = self.q_proj(x)
#         q_neg = self.neg_q_proj(x)

#         k = self.k_proj(embedding_model.token_embedder.weight).unsqueeze(0)
#         y_pos = torch.matmul(q, k.transpose(1,2)) / math.sqrt(q.size(-1))
#         y_neg = torch.matmul(q_neg, k.transpose(1,2)) / math.sqrt(q.size(-1))

#         # extract the negative heads, and combine them to create final attention weights
#         lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
#         lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
#         lambda_full = lambda_1 - lambda_2 + self.lambda_init


#         y = y_pos - lambda_full * y_neg
#         return y, None





class AttentionLMHead(nn.Module):
    """
    Multi-Headed Grouped Attention-based Language Modeling Head.

    This LM head uses multi-headed attention over normalized token embeddings
    instead of a traditional linear layer. The token embeddings are normalized
    before attention, and the output logits are raw scores without softmax.

    Args:
        model_cfg (dict): Configuration dictionary with the following keys:
            - vocab_size (int): Size of the vocabulary.
            - hidden_dim (int): Dimensionality of the hidden states.
            - lm_head_normalization (str): Type of normalization ('layernorm', 'batchnorm', etc.).
            - lm_head_bias (bool): Whether to include bias in normalization layers.
            - num_attention_heads (int): Number of attention heads.
            - attention_groups (int): Number of groups to divide the attention heads into.
    """
    def __init__(self, model_cfg):
        super().__init__()
        vocab_size = model_cfg["vocab_size"]
        hidden_dim = model_cfg["hidden_dim"]
        num_heads = 4 #model_cfg["num_attention_heads"]
        num_groups = 2 #model_cfg.get("attention_groups", 1)

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = hidden_dim // num_heads
        self.group_size = num_heads // num_groups

        # MultiheadAttention expects (seq_len, batch, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            bias=False,
            batch_first=True  # Use batch_first for (batch, seq, embed)
        )

        # Layer normalization for queries
        self.q_norm = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=hidden_dim,
            bias=model_cfg["lm_head_bias"],
        )

        # Optional: Grouping mechanism (if specific grouping is needed)
        # Here, we assume standard multihead attention with grouped heads handled internally

    def forward(self, x, embedding_model):
        """
        Forward pass for the LM head.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).
            embedding_model (nn.Module): The embedding model containing token embeddings.

        Returns:
            logits (Tensor): Output logits of shape (batch_size, seq_length, vocab_size).
            None: Placeholder for compatibility (e.g., hidden states).
        """
        # Normalize input queries
        q = self.q_norm(x)  # Shape: (batch_size, seq_length, hidden_dim)

        # Project queries using the multihead attention
        # Key and Value are the normalized token embeddings
        # Normalize token embeddings before attention
        token_embeddings = embedding_model.token_embedder.weight  # Shape: (vocab_size, hidden_dim)
        token_embeddings_normalized = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)  # Shape: (vocab_size, hidden_dim)

        # Expand token embeddings to include batch dimension
        # For MultiheadAttention, key and value should be (batch_size, vocab_size, hidden_dim)
        # We'll repeat the normalized token embeddings across the batch dimension
        batch_size, seq_length, _ = x.size()
        key = token_embeddings_normalized.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, vocab_size, hidden_dim)
        value = key.clone()  # Typically, value is the same as key in language modeling heads

        # Perform multihead attention
        # Query: (batch_size, seq_length, hidden_dim)
        # Key/Value: (batch_size, vocab_size, hidden_dim)
        # Output: (batch_size, seq_length, hidden_dim)
        attn_output, _ = self.multihead_attn(q, key, value)  # attn_output shape: (batch_size, seq_length, hidden_dim)

        # Project the attention output to logits over the vocabulary
        # Typically, this can be a linear layer, but since we're using attention over token embeddings,
        # we can compute the similarity between attn_output and token_embeddings_normalized
        # However, to keep it efficient, we can use a linear projection without bias
        logits = torch.matmul(attn_output, token_embeddings_normalized.transpose(0, 1))  # Shape: (batch_size, seq_length, vocab_size)

        return logits, None




class AttentionLMHead(torch.nn.Module):
    """ 
    TODO  
        - shouldn't re-calculate embeddings for accumulation_steps
    """
    def __init__(self, model_cfg):
        super().__init__()
        vocab_size = model_cfg["vocab_size"]
        self.vocab_size = vocab_size
        self.q_proj = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["hidden_dim"]*4,
            bias=False
        )

        self.k_proj = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["hidden_dim"]*4,
            bias=False
        )

        #self.v = torch.nn.Identity()

        self.layer_norm = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )

        self.layer_norm2 = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )


    def forward(self, x, embedding_model):
        """ TODO """
        B, S, H = x.size()
        x = self.layer_norm(x)
        embed = self.layer_norm2(embedding_model.token_embedder.weight)

        # embed and reshape 
        # input(f"{self.q_proj(x).size()=}")
        # input(f"{self.k_proj(embed).size()=}")
        q = self.q_proj(x).reshape(B, 4, S, H)
        k = self.k_proj(embed).reshape(1, 4, self.vocab_size, H)

        y = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.size(-1))

        # average across second dim
        y = torch.mean(y, dim=1)





        # q = self.q_proj(self.layer_norm(x))
        # k = self.k_proj(self.layer_norm2()).unsqueeze(0)
        # y = torch.matmul(q, k.transpose(1,2)) / math.sqrt(q.size(-1))
        return y, None




class AttentionLMHead(torch.nn.Module):
    """ 
    TODO  
    """
    def __init__(self, model_cfg):
        super().__init__()
        vocab_size = model_cfg["vocab_size"]
        depth = 2

        self.q_proj = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["hidden_dim"],
            bias=False
        )

        self.neg_q_proj = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["hidden_dim"],
            bias=False
        )

        # self.k_proj = torch.nn.Linear(
        #     in_features=model_cfg["hidden_dim"],
        #     out_features=256,
        #     bias=False
        # )

        #self.v = torch.nn.Identity()

        self.layer_norm = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )
        self.layer_norm2 = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )

        # lambda
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth) # from the paper
        self.lambda_q1 = torch.nn.Parameter(torch.zeros(256, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = torch.nn.Parameter(torch.zeros(256, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = torch.nn.Parameter(torch.zeros(256, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = torch.nn.Parameter(torch.zeros(256, dtype=torch.float32).normal_(mean=0,std=0.1))



    def forward(self, x, embedding_model):
        """ TODO """
        x = self.layer_norm(x)

        q = self.q_proj(x)
        q_neg = self.neg_q_proj(x)

        # k = self.k_proj(self.layer_norm2(embedding_model.token_embedder.weight)).unsqueeze(0)
        k = self.layer_norm2(embedding_model.token_embedder.weight).unsqueeze(0)
        y_pos = torch.matmul(q, k.transpose(1,2)) / math.sqrt(q.size(-1))
        y_neg = torch.matmul(q_neg, k.transpose(1,2)) / math.sqrt(q.size(-1))

        # extract the negative heads, and combine them to create final attention weights
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init


        y = y_pos - lambda_full * y_neg
        return y, None


class AttentionLMHead(torch.nn.Module):
    """ 
    TODO  
        - shouldn't re-calculate embeddings for accumulation_steps
    """
    def __init__(self, model_cfg):
        super().__init__()
        vocab_size = model_cfg["vocab_size"]

        # self.q_proj = torch.nn.Linear(
        #     in_features=model_cfg["hidden_dim"],
        #     out_features=model_cfg["hidden_dim"],
        #     bias=False
        # )

        # self.k_proj = torch.nn.Linear(
        #     in_features=model_cfg["hidden_dim"],
        #     out_features=model_cfg["hidden_dim"],
        #     bias=False
        # )

        #self.v = torch.nn.Identity()

        self.layer_norm = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )

        self.layer_norm2 = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )


    def forward(self, x, embedding_model):
        """ TODO """
        q = self.layer_norm(x)
        k = self.layer_norm2(embedding_model.token_embedder.weight).unsqueeze(0)
        y = torch.matmul(q, k.transpose(1,2)) / math.sqrt(q.size(-1))
        return y, None


class AttentionLMHead(torch.nn.Module):
    """ 
    TODO  
        - shouldn't re-calculate embeddings for accumulation_steps
    """
    def __init__(self, model_cfg):
        super().__init__()
        vocab_size = model_cfg["vocab_size"]

        self.q_proj = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["hidden_dim"],
            bias=False
        )

        self.k_proj = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["hidden_dim"],
            bias=False
        )

        #self.v = torch.nn.Identity()

        self.layer_norm = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )

        self.layer_norm2 = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )


    def forward(self, x, embedding_model):
        """ TODO """
        q = self.q_proj(self.layer_norm(x))
        k = self.k_proj(self.layer_norm2(embedding_model.token_embedder.weight)).unsqueeze(0)
        y = torch.matmul(q, k.transpose(1,2)) / math.sqrt(q.size(-1))


        y = y - torch.mean(y, dim=-1, keepdim=True)
        return y, None
"""

Maybe split up for pos/neg at the second last transformer block
"""