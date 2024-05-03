"""
The Model Shell holds the tokenizer, core-model and model head.
"""

import torch 
import torch.nn as nn

from models.components.tokenizers import build_tokenizer
from models.components.LMHeads import (
    NextTokenHead
)

from models.utils import (
    print_model_stats
)

from models.components.layers import BidirectionalTransformerBlock
from models.components.layers import BaseTransformerBlock
from models.components.positional_encoding import LearnedPosEncoding

class AutoregressiveByteModelShell(nn.Module):
    """
    A model shell for byte-level learning.
    """
    def __init__(
            self,
            cfg,
            core_model,
        ):
        super().__init__()

        # move to class
        self.cfg = cfg
        self.core_model = core_model

        # build the tokenizer 
        self.byte_tokenizer = build_tokenizer(
            tokenizer_type=self.cfg["model_shell"]["pooling_tokenizer"],
            vocab_size=self.cfg["model_shell"]["pooling_vocab_size"],
            dataset_name=self.cfg["model_shell"]["tokenizer_dataset_name"],
        )

        self.pooling_tokenizer = build_tokenizer(
            tokenizer_type=self.cfg["model_shell"]["tokenizer"],
            vocab_size=self.cfg["model_shell"]["vocab_size"],
            dataset_name=self.cfg["model_shell"]["tokenizer_dataset_name"],
        )

        # build the embedder 
        self.token_embedder = nn.Embedding(
            num_embeddings=self.cfg["model_shell"]["vocab_size"],
            embedding_dim=self.cfg["model_shell"]["embedding_dim"],
        )

        self.byte_token_processor = ByteLevelProcessor(
            embedding_dim=self.cfg["model_shell"]["embedding_dim"],
            hidden_dim=self.cfg["core_model"]["hidden_dim"],
            pooling_tokenizer=self.pooling_tokenizer,
            byte_tokenizer=self.byte_tokenizer,
            token_embedder=self.token_embedder
        )

         # build the language model head
        self.lm_head = NextTokenHead(
            hidden_dim=self.cfg["core_model"]["hidden_dim"],
            vocab_size=self.cfg["model_shell"]["pooling_vocab_size"],
        )

        self.byte_decoder = ByteLevelDecoder(
            core_hidden_dim=self.cfg["core_model"]["hidden_dim"],
            byte_hidden_dim=self.cfg["model_shell"]["embedding_dim"],
            byte_vocab_size=self.cfg["model_shell"]["pooling_vocab_size"],
            lm_head = self.lm_head
        )

       

        # share the weights between the token embeddings and the final logit layer
        #self.token_embedder.weight = (
        #    self.lm_head.linear.weight
        #) # https://paperswithcode.com/method/weight-tying


        # report number of parameters
        print_model_stats(self)


    def forward(self, token_ids):
        """
        The default forward pass is used for training and accepts the 
        token_ids as input. When the model is in eval mode, only the 
        last token is passed into the NextTokenHead.
        """

        #b, s = token_ids.size()

        # check that the sequence length is not longer than the context window
        #assert (
        #    s <= self.cfg["model_shell"]["context_window"]
        #), f"Cannot forward sequence of length {s}, block size is only {self.cfg['model_shell']['context_window']}"


        # embed token_ids
        #x = self.token_embedder(token_ids)

        # process to sub-word tokens
        x, x_emb = self.byte_token_processor(token_ids)
        print("second", x.size())

        # forward through the core model
        x_return = self.core_model(x)
        print("second", x_return.size())

        # pass into the byte level decoder 
        x_return = self.byte_decoder(x_emb, x_return)

        input(x_return.size())

        return x_return, None


        if isinstance(x, tuple):
            x, loss = x_return
        else:
            x, loss = x_return, None

        # get logits
        logits = self.lm_head(x)
        print("second", logits.size())

        return logits, loss
        



class ByteLevelDecoder(nn.Module):
    """
    Use multiple learned heads to decode into by hidden size,
    pre-append to the byte embeddings of the answers and 
    autoregressively decode the next token, applying the 
    LM (byte level) head only to the actual tokens, not 
    the latent ecoded ones.
    """
    def __init__(self, core_hidden_dim, byte_hidden_dim, byte_vocab_size, lm_head):
        super().__init__()
        self.core_hidden_dim = core_hidden_dim
        self.byte_hidden_dim = byte_hidden_dim
        self.byte_vocab_size = byte_vocab_size
        self.num_projection_heads = 12

        # project via linear and then split
        self.projection = nn.Linear(
            in_features=core_hidden_dim,
            out_features=self.num_projection_heads * byte_hidden_dim,
            bias=False
        )

        # small autoregressive transformer 
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(),
                h=nn.ModuleList(
                    [
                        BaseTransformerBlock(
                            hidden_dim=byte_hidden_dim,
                            ffn_dim=byte_hidden_dim*4,
                            ffn_activation="gelu",
                            bias=False,
                            num_heads=8,
                        ),
                        BaseTransformerBlock(
                            hidden_dim=byte_hidden_dim,
                            ffn_dim=byte_hidden_dim*4,
                            ffn_activation="gelu",
                            bias=False,
                            num_heads=8,
                        ),
                    ]
                ),
            )
        )

        self.pos_encoder = LearnedPosEncoding(
            hidden_dim=byte_hidden_dim,
            context_window=12+self.num_projection_heads
        )
        self.lm_head = lm_head

        """self.lm_head = nn.Linear(
            in_features=byte_hidden_dim,
            out_features=byte_vocab_size,
            bias=False
        )"""

    def forward(self, x_raw_emb, x):
        """
        x_raw_emb (the original byte embeddings): (B, S, 12, H_b)
        x (the latent embeddings): (B, S, H)
        """
        # project the latent embeddings
        x = self.projection(x)
        x = x.view(x.size(0), x.size(1), self.num_projection_heads, self.byte_hidden_dim)
        print('important shapes', x.size())
        print('important shapes', x_raw_emb.size())

        # view x_raw_emb
        x_raw_emb = x_raw_emb.view(x.size(0), x.size(1), 12, self.byte_hidden_dim)
        # concat x with x_byte_emb
        print('important shapes', x.size())
        print('important shapes', x_raw_emb.size())
        
        x = torch.cat([x, x_raw_emb], dim=-2)

        # flatten across B and S and pass through transformer
        B, S, _, _ = x.size()
        x = x.view(B*S, self.num_projection_heads+12, self.byte_hidden_dim)

        # positional encoding
        x = x + self.pos_encoder(x)

        # pass through transformer
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)

        # pass final 12 byte tokens through lm head
        x = x[:, -12:, :]
        x = self.lm_head(x)

        # reshape and return
        x = x.view(B, S, 12, self.byte_vocab_size)

        return x







class ByteLevelProcessor(nn.Module):
    """
    Takes byte level encodings, processes them via 
    two local-attention transformer blocks and pools 
    the resultant tokens based on gpt-2 tokenizer 
    boundaries.
    Inputs are batches of lists of token blocks 
    in the gpt2 tokenizer boundaries.
    """
    def __init__(self, embedding_dim, hidden_dim, pooling_tokenizer, byte_tokenizer, token_embedder):
        super().__init__() 
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pooling_tokenizer = pooling_tokenizer
        self.byte_tokenizer = byte_tokenizer
        self.token_embedder = token_embedder
        self.transformer = nn.ModuleList(
            [
                BidirectionalTransformerBlock(
                    hidden_dim=embedding_dim,
                    ffn_dim=embedding_dim*4,
                    ffn_activation="gelu",
                    bias=False,
                    num_heads=8,
                    dropout=0.0
                ),
                BidirectionalTransformerBlock(
                    hidden_dim=hidden_dim,
                    ffn_dim=hidden_dim*4,
                    ffn_activation="gelu",
                    bias=False,
                    num_heads=8,
                    dropout=0.0
                ),
            ]
        )
        self.up_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)


    def forward(self, x):
        """
        A batch of sequences of byte tokens.
        First flatten across first dim, pass through transformer, pool and reshape
        """
        input(x.size())

        B, S, S_char = x.size()

        x = x.view(B*S, S_char)

        # embed 
        x_emb = self.token_embedder(x)
        print(x_emb.size())
        x = self.transformer[0](x_emb)
        x = self.up_proj(x)
        x = self.transformer[1](x)
        print(x.size())
        x = x.mean(dim=-2)
        print(x.size())
        print(B, S)
        x = x.view(B, S, -1)
        input(x.size())
        return x, x_emb
