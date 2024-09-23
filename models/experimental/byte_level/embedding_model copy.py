"""
A collection of embedding models. A collection model includes
the tokenizer(s), token embeddings and positional encodings
(if necessary).
"""

import torch

from models.components.positional_encoding import LearnedPosEncoding
from models.components.layers.tokenizers import build_tokenizer
from models.embedding_models import EmbedderInterface
from models.experimental.byte_level.layers import ByteLevelTransformerBlock

from models.components.layers.transformer_blocks import GenericTransformerBlock
from copy import deepcopy

class TokenizerEncoder(torch.nn.Module):
    """
    Take seq of byte embeddings, return sequence of delimiters. (binary)
    """

    def __init__(self, max_chunck_length):
        super().__init__()

        layers = 2
        self.max_chunck_length = max_chunck_length

        self.transformer = torch.nn.ModuleList(
            [
                GenericTransformerBlock(
                    hidden_dim=128,
                    context_window=5*2048,
                    use_rope=True,
                    ffn_cfg={
                        "ffn_type": "generic", #128*128 + 128*4*128*2
                        "ffn_dim": 4*128,
                        "activation": "gelu",
                        "normalization": "rms_norm",
                        "bias": False,
                        "dropout": 0.0
                    },
                    attn_cfg={
                        "attn_type": "causal",
                        "num_kv_heads": 8,
                        "num_q_heads": 8,
                        "normalization": "rms_norm",
                        "bias": False,
                        "dropout": False,
                    }
                ) for _ in range(layers)
            ]
        )


        self.end_of_seq_head = torch.nn.Linear(
            128,
            2, # 2 because we just need to predict whether it's a <sep> or <end>
            bias=True
        )


    def forward(self, x, pad_token_vector, x_ids, pad_token_id):
        # input is seq of bytes, output is seq of eoc/neoc
        # x_chunking = deepcopy(x)
        x_chunking = x.clone()
        #x_chunking = x
        for block in self.transformer:
            x_chunking = block(x_chunking)

        # pass through head 
        #assert checkpoint == x, "deepcopy is required"

        output = self.end_of_seq_head(x) # should be shape (seq-len, 2) ; x = (seq-len, 128) -> N_c * S_c * 128
        #print("Output shape:", output.shape)

        
        # optimally, outputformat is num_chunks x variable chunk length (padded) x original byte embed
        self.max_chunk_length = 16 # statisically 
        max_num_chuncks = 1_024

        # initalize the output tensor 
        output_tensor = torch.ones((x.size(0), max_num_chuncks, self.max_chunk_length, 128), device="cuda") * pad_token_vector # B, N_c, S_c, 128
        output_token_ids = torch.ones((x.size(0), max_num_chuncks, self.max_chunk_length), device="cuda") * pad_token_id

        # loop over the 
        for i in range(output.size(0)): # iterate over batch
            
            prev_idx = 0
            chunk_counter = 0
            for ii in range(output.size(1)-1): # iterate over the byte sequence
                # check if end of chunk
                if output[i, ii, 0] > output[i, ii, 1]: # end of chunk
                    # set output tensor to relevant values
                    next_idx = min(prev_idx+self.max_chunk_length, ii+1)
                    # print(output_tensor.size(), output_tensor[i, chunk_counter, :, :].size())
                    # print(x.size(), x[i, prev_idx:next_idx, :].size())
                    # print(prev_idx, next_idx, chunk_counter)
                    # input()
                    chunck_len = next_idx - prev_idx
                    output_tensor[i, chunk_counter, :chunck_len, :] += x[i, prev_idx:next_idx, :]
                    
                    output_token_ids[i, chunk_counter, :chunck_len] = x_ids[i, prev_idx:next_idx]
                    prev_idx = next_idx 
                    chunk_counter += 1
                    if chunk_counter >= max_num_chuncks:
                        break



        # print(f"Ouput_token_ids: {output_token_ids.size()}")
        # input(output_tensor)

        return output_tensor, output_token_ids



class ByteBidirectionEncoding(torch.nn.Module):
    """
    Input shape: batch x max_num_chuncks x max_chunck_length x 128 (byte_embed_dim)
    return batch x max_num_chuncks x hidden_dim (512)
    """
    def __init__(self, max_chunck_length):
        super().__init__()
        # build the transformer blocks
        self.transformer = torch.nn.ModuleList(
            [
                ByteLevelTransformerBlock(
                    input_dim=128,
                    output_dim=128 * 2,
                    ffn_dim=128 * 4,
                    context_window=max_chunck_length,
                    use_rope=True,
                ),
                ByteLevelTransformerBlock(
                    input_dim=128*2,
                    output_dim=128 * 2,
                    ffn_dim=128 * 8,
                    context_window=max_chunck_length,
                    use_rope=True,
                ),
                # ByteLevelTransformerBlock(
                #     input_dim=128*2,
                #     output_dim=128 * 4,
                #     ffn_dim=128 * 8,
                #     context_window=max_chunck_length,
                #     use_rope=True,
                # ),
                # ByteLevelTransformerBlock(
                #     input_dim=128 * 4,
                #     output_dim=128*4,
                #     ffn_dim=128 * 16,
                #     context_window=max_chunck_length,
                #     use_rope=True,
                # ),
            ]
        )



    def forward(self, x):
        # B, Num_chunck, Chunck_len, 128 
        B, C_num, C_len, h_b = x.size()
        # flatten first two dims 
        x = x.view(B*C_num, C_len, h_b)

        # pass through blocks
        for block in self.transformer:
            x = block(x)

        # use last # TODO try mean pooling
        x = x[:, -1, :] # (batch*C_num), 1, 512 


        # reshape it back to 3
        x = x.view(B, C_num, -1)  # batch, chunk_num, 512

        return x 


class ByteLevelEmbedder(EmbedderInterface):
    """
    Input is a sequence of byte-level token ids
    """

    # pylint: disable=super-init-not-called
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        max_chunck_length = 16

        self.byte_tokenizer = build_tokenizer(
            tokenizer_type="bpe",
            vocab_size=model_cfg["vocab_size"],
            simplify=False,
            dataset_name="simple_en_wiki"
        )

        self.byte_emebdder = torch.nn.Embedding(
            num_embeddings=model_cfg["vocab_size"],
            embedding_dim=128
        )

        self.delimiter_model = TokenizerEncoder(
            max_chunck_length=max_chunck_length
        )

        self.word_encoding_model = ByteBidirectionEncoding(
            max_chunck_length=max_chunck_length
        )

        # get pad token id and pad token vector
        self.pad_token_id = self.byte_tokenizer.pad_token
        self.eot_token = self.byte_tokenizer.eot_token

    def forward(self, x):
        # pass through byte_tokenizer
        #x = self.byte_tokenizer.encode(token_ids)

        # pass through delimiter 
        x, output_token_ids = self.delimiter_model(
            x=self.byte_emebdder(x), 
            pad_token_vector=self.byte_emebdder(torch.tensor([self.pad_token_id]).to("cuda"))[0], 
            x_ids=x, 
            pad_token_id=self.pad_token_id
        )

        # pass through word embedding model
        x = self.word_encoding_model(x)

        # double-check the size
        #input(x.size())
        
        return x, output_token_ids


    def tokenize_input(self, input_string, truncate=False, add_eot=True):
        token_ids = self.byte_tokenizer.encode(input_string)
        if add_eot:
            token_ids.append(self.eot_token)
        return token_ids 