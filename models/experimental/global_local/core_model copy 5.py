import torch
import torch.nn as nn
from models.components.attention import build_attention
from models.components.feedforward import build_ffn
from typing import Optional

from models.components.transformer_blocks import GenericTransformerBlock


import torch
import torch.nn as nn
from models.components.attention import build_attention
from models.components.feedforward import build_ffn
from typing import Optional, Tuple

from models.components.transformer_blocks import GenericTransformerBlock




class DualCoreModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()


        # Initialize global_token with correct dimensions
        self.global_token = nn.Parameter(torch.randn(1, model_cfg["global_hidden_dim"]))
        self.global_token.requires_grad = True

        # global projections
        self.proj_a = torch.nn.Linear(
            model_cfg["global_hidden_dim"], 
            model_cfg["global_down_proj_mult"]*model_cfg["hidden_dim"],
            bias=False
        )
        self.proj_b = torch.nn.Linear(
            model_cfg["global_down_proj_mult"]*model_cfg["hidden_dim"],
            model_cfg["global_hidden_dim"], 
            bias=False
        )

        # share weights
        self.proj_a.weight = torch.nn.Parameter(self.proj_b.weight.T)


        self.hidden_dim = model_cfg["hidden_dim"]
        self.down_proj_mult = model_cfg["global_down_proj_mult"]


        # initialize both transformer models (global and byte)

        self.global_transformer = nn.ModuleDict(
            {
                "drop": nn.Dropout(model_cfg.get("dropout_rate", 0.1)),
                "h": nn.ModuleList(
                    [
                        GenericTransformerBlock(
                            # model_cfg=model_cfg,
                            hidden_dim=model_cfg["global_hidden_dim"],
                            context_window=model_cfg["context_window"]//4,
                            ffn_cfg=model_cfg["global_ffn"],
                            attn_cfg=model_cfg["global_attn"],
                        )
                        for _ in range(model_cfg["num_layers"])
                    ]
                ),
            }
        )
        self.byte_transformer = nn.ModuleDict(
            {
                "drop": nn.Dropout(model_cfg.get("dropout_rate", 0.1)),
                "h": nn.ModuleList(
                    [
                        GenericTransformerBlock(
                            # model_cfg=model_cfg,
                            hidden_dim=model_cfg["hidden_dim"],
                            context_window=model_cfg["context_window"]*4,
                            ffn_cfg=model_cfg["ffn"],
                            attn_cfg=model_cfg["attn"],
                        )
                        for _ in range(model_cfg["num_layers"])
                    ]
                ),
            }
        )

    def set_embedder(self, embedder):
        """
        Set the embedder for the model.

        Args:
            embedder: The embedder object containing token embeddings.
        """
        self.embedder = embedder


    def _insert_global_tokens(
        self, 
        byte_sequence_emb, 
        delimiter_sequence,
        projected_global,
        start_global_emb,
        end_global_emb
    ):
        """ TODO """
        assert byte_sequence_emb.size(0) == 1, "Not implemented for batch size > 1"
        byte_emb = byte_sequence_emb[0]         # Shape: (BYTE_SEQ, BYTE_HIDDEN)
        delimiter = delimiter_sequence[0]       # Shape: (BYTE_SEQ,)
        projected_global = projected_global[0]       


        # Identify delimiter positions where insertion should occur
        insert_positions = (delimiter == 1).nonzero(as_tuple=False).squeeze()
        if insert_positions.dim() == 0:
            insert_positions = insert_positions.unsqueeze(0)
        insert_positions = insert_positions.tolist()  # List of indices

        num_delimiters = len(insert_positions)
        total_length = byte_emb.size(0)


        split_sizes = [insert_positions[0]] if insert_positions[0] != 0 else []
        for i in range(1, len(insert_positions)):
            split_sizes.append(insert_positions[i] - insert_positions[i-1])
        # split_sizes.append(128 - insert_positions[-1])
        split_sizes.append(byte_sequence_emb.size(1) - insert_positions[-1])

        # split
        splits = torch.split(byte_emb, split_sizes, dim=0)


        # Prepare start and end embeddings
        start_emb = start_global_emb.expand(1, -1)    # Shape: (1, BYTE_HIDDEN)
        end_emb = end_global_emb.expand(1, -1)        # Shape: (1, BYTE_HIDDEN)

        x_full = []
        global_idx = []
        dim_counter = 0

        for split, p_global in zip(splits, projected_global):

            x_full += [
                split,
            ]
            dim_counter += split.size(0)
            global_idx.append(dim_counter) #len(x_full))
            # print(split.size(), global_idx[-1])
            # input()
            x_full += [
                start_emb,
                p_global.view(self.down_proj_mult, self.hidden_dim),
                end_emb
            ]
            dim_counter += 2+self.down_proj_mult
            # print("\n\n")
            # print(f"{split.size()=}")
            # print(f"{start_emb.size()=}")
            # print(f"{p_global.view(self.down_proj_mult, self.hidden_dim).size()=}")
            # print(f"{end_emb.size()=}")
            # print("\n\n")
            # input()
            # split.size()=torch.Size([3, 64])
            # start_emb.size()=torch.Size([1, 64])
            # p_global.size()=torch.Size([256])
            # end_emb.size()=torch.Size([1, 64])

        # concat all
        # print(x_full)
        x_full = torch.cat(x_full, dim=0).unsqueeze(0)
        global_idx = torch.tensor(global_idx, dtype=torch.long).unsqueeze(0)

        return x_full, global_idx


    def _update_global_values(self, x_full, global_idx, global_transformer_block):
        """ TODO """
        assert x_full.size(0) == 1, "Only supports Batch=1"
        
        # extract global tokens
        x_full = x_full[0]
        global_idx = global_idx[0]

        global_token_list = []

        for idx in global_idx:
            global_token_list.append(
                self.proj_b(x_full[idx:idx+self.down_proj_mult].view(-1).unsqueeze(0))
            )
        global_tokens = torch.cat(global_token_list, dim=0).unsqueeze(0)

        # pass through transformer
        global_tokens = global_transformer_block(global_tokens)

        global_tokens = global_tokens[0]

        for i, idx in enumerate(global_idx):
            x_full[idx+1:idx+self.down_proj_mult+1] += self.proj_a(global_tokens[i]).view(
                self.down_proj_mult, self.hidden_dim
            )

        return x_full.unsqueeze(0)

    def _extract_updated_byte_tokens(self, x_full, global_idx):
        """ TODO """
        # mask = torch.ones_like(x_full, dtype=torch.long)
        # for idx in global_idx[0]:
        #     mask[0, idx:idx+self.down_proj_mult+2] = 0
        # input(mask)
        # print(mask.size())
        # print(x_full.size())
        # print(x_full[mask].size())
        # return x_full[mask]
        global_idx = global_idx[0]
        byte_out_list = [x_full[0, :global_idx[0]]]
        print(global_idx)

        for idx in range(1, len(global_idx)):
            byte_out_list.append(
                x_full[0, global_idx[idx-1]+2+self.down_proj_mult:global_idx[idx]]
            )
            # input(x_full[0, global_idx[idx-1]+2+self.down_proj_mult:global_idx[idx]].size())

        return torch.cat(byte_out_list, dim=0).unsqueeze(0)

    def forward(self, x, delimitations):
        print(x.size())
        # down_project the global tokens and insert them
        num_global_tokens = torch.sum(delimitations)
        global_tokens = self.global_token.expand(1, num_global_tokens, -1).clone()

        projected_global = self.proj_a(global_tokens)



        x_full, global_idx = self._insert_global_tokens(
            byte_sequence_emb=x,
            delimiter_sequence=delimitations,
            projected_global=projected_global,
            start_global_emb=self.embedder(torch.tensor([self.embedder.tokenizer.start_global])),
            end_global_emb=self.embedder(torch.tensor([self.embedder.tokenizer.end_global])),
        )

        # for multiple iterations, bass this through the local and global transformer blocks
        for (local_transformer_block, global_transformer_block) in zip(
            self.byte_transformer.h, 
            self.global_transformer.h
        ):
            # input(x_full.size()) # [1, 4663, 64]
            # pass through local
            x_full = local_transformer_block(x_full)

            # input(x_full.size())
            # pass through global
            x_full = self._update_global_values(
                x_full=x_full,
                global_idx=global_idx,
                global_transformer_block=global_transformer_block
            )

            break

        # remove the global tokens and return it 
        x = self._extract_updated_byte_tokens(x_full=x_full, global_idx=global_idx)
        input(x.size())
        return x