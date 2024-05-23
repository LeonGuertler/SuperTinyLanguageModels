"""
The Embedding model for a VAE style sequence to sequence model.
"""
import torch 

from models.embedding_models import GenericEmbedder
from models.components.layers.transformer_blocks import GenericTransformerBlock

from models.components.positional_encoding import build_positional_encodings
from models.components.tokenizers import build_tokenizer


# import local components
from models.experimental.next_thought.layers import AttentionPoolingRemoval



class HierarchicalEncoder(GenericEmbedder):
    """
    Accepts an arbitrary length sequence as input,
    uses the QK^T matrix to, at every layer,
    pick the top n-percent of nodes to pool into 
    a single token (the one paying most attention 
    to the other should be pooled into the other token).
    """
    def __init__(self, model_cfg):
        super().__init__()
        # build the tokenizer
        self.tokenizer = build_tokenizer(
            tokenizer_type=model_cfg["embedder"]["tokenizer_type"],
            vocab_size=model_cfg["vocab_size"],
            dataset_name=model_cfg["embedder"]["dataset_name"],
        )

        # build the token embeddings
        self.token_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["vocab_size"],
            embedding_dim=model_cfg["embedder"]["pooling_dims"][0],
        )

        # build the positional encodings
        self.positional_encodings = build_positional_encodings(model_cfg=model_cfg)


        self.standard_transformer = torch.nn.ModuleList(
            [
                GenericTransformerBlock(
                    hidden_dim=model_cfg["embedder"]["pooling_dims"][0],
                    context_window=model_cfg["embedder"]["context_window"],
                    use_rope=model_cfg["embedder"]["positional_encoding_type"] == "rope",
                    ffn_cfg=model_cfg["embedder"]["standard_ffn_block"],
                    attn_cfg=model_cfg["embedder"]["standard_attn_block"],
                )
            ]
        )

        self.pooling_transformer = torch.nn.ModuleList(
            [
                AttentionPoolingRemoval(
            
                ))
            ]






        hidden_size = hidden_size_in
        hidden_size_ff = 3072
        num_heads = 12

        self.embedding = torch.nn.Embedding(50304, hidden_size)
        self.positional_encoding = LearnedPosEncoding(768, 512)

        self.standard = torch.nn.ModuleList([
            NormalAttentionBlock(hidden_size, hidden_size_ff, num_heads),
            NormalAttentionBlock(hidden_size, hidden_size_ff, num_heads),
        ])
        h1 = 768
        h2 = 1920
        h3 = 1920
        h4 = 1920
        h5 = 4800
        # convert all to int
        #768 1920 4800 4800 30000
        h1, h2, h3, h4, h5 = map(int, [h1, h2, h3, h4, h5])
        print(h1, h2, h3, h4, h5)

        #768 1919.0 4797.0 11992.0 29979.0


        self.pooling_attention = torch.nn.ModuleList([
            AttentionPoolingRemoval(
                hidden_size_in=h1,
                hidden_size_out=h2,
                num_topk_heads=12, 
                num_attention_heads=12,
                pct_pool_per_layer=0.3,
            ),
            AttentionPoolingRemoval(
                hidden_size_in=h2,
                hidden_size_out=h3,
                num_topk_heads=12, 
                num_attention_heads=12,
                pct_pool_per_layer=0.5
            ),
            AttentionPoolingRemoval(
                hidden_size_in=h3,
                hidden_size_out=h4,
                num_topk_heads=12, 
                num_attention_heads=12,
                pct_pool_per_layer=0.6
            ),
            AttentionPoolingRemoval(
                hidden_size_in=h4,
                hidden_size_out=h5,
                num_topk_heads=12, 
                num_attention_heads=12,
                pct_pool_per_layer=0.6
            ),
        ])




    def forward(self, x):
        # embed the input 
        x = self.embedding(x)

        # apply positional encoding 
        x = x + self.positional_encoding(x)


        # first pass through normal attention blocks
        for layer in self.standard:
            x = layer(x)

        # then pass through pooling attention blocks
        for layer in self.pooling_attention:
            x = layer(x)
        # mean pool final representation
        x = x.mean(dim=-2)
        return x