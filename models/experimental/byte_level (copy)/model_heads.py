"""
A collection of different model heads.
"""

import torch

from models.components.positional_encoding import LearnedPosEncoding
from models.experimental.byte_level.layers import (
    ByteLevelTransformerBlock,
    GenericTransformerBlockByte
)
from models.components.transformer_blocks import GenericTransformerBlock

import time


class ByteLevelDecoder(torch.nn.Module):
    """
    Use multiple learned heads to decode into by hidden size,
    pre-append to the byte embeddings of the answers and
    autoregressively decode the next token, applying the
    LM (byte level) head only to the actual tokens, not
    the latent ecoded ones.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.hidden_dim = model_cfg["hidden_dim"]
        self.byte_hidden = model_cfg["byte_hidden"]
        self.byte_vocab_size = model_cfg["vocab_size"]
        self.max_chunk_length = model_cfg["max_chunk_length"]
        self.num_byte_decoder_layers = model_cfg["num_byte_decoder_layers"]

        self.projection = torch.nn.Linear(
            in_features=self.hidden_dim, # 512
            out_features=self.max_chunk_length * self.byte_hidden,
            bias=False,
        )

        # build transformer block
        self.transformer = torch.nn.ModuleList(
            [
                ByteLevelTransformerBlock(
                    input_dim=self.byte_hidden,
                    output_dim=self.byte_hidden,
                    ffn_dim=self.byte_hidden * 4,
                    context_window=self.max_chunk_length,
                    use_rope=True,
                ) for _ in range(self.num_byte_decoder_layers)
            ]
        )

        self.lm_head = torch.nn.Linear(
            in_features=self.byte_hidden, # 128
            out_features=self.byte_vocab_size, # 259 (256 bytes + 3 special)
            bias=False,
        )

    def forward(self, x):
        """
        Bidirectionally decode all tokens at once
        """
        # project the latent embeddings
        x = self.projection(x)
        x = x.view(x.size(0), x.size(1), self.max_chunk_length, self.byte_hidden)


        # Reshape for transformer
        B, S, _, _ = x.size()
        x = x.view(B * S, self.max_chunk_length, self.byte_hidden).contiguous()

        # pass through transformer
        for block in self.transformer:
            x = block(x)
        # pass final self.max_chunk_length byte tokens through lm head
        x = self.lm_head(x)

        # reshape and return
        x = x.view(B, S, self.max_chunk_length, self.byte_vocab_size)
        return x

    def inference(self, x):
        """
        inference
        """
        return self.forward(x)[0][:, -1, :, :]


class ByteLevelDecoder(torch.nn.Module):
    """
    Selective autoregressive decoder.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.hidden_dim = model_cfg["hidden_dim"]
        self.byte_hidden = model_cfg["byte_hidden"]
        self.byte_vocab_size = model_cfg["vocab_size"]

        self.num_global_proj = model_cfg["global_byte_projection"]
        self.num_byte_decoder_layers = model_cfg["num_byte_decoder_layers"]

        self.projection = torch.nn.Linear(
            in_features=self.hidden_dim, # 512
            out_features=self.num_global_proj * self.byte_hidden,
            bias=False,
        )

        self.eos_token_id = 257
        self.pad_token_id = 256

        # build transformer block
        self.transformer = torch.nn.ModuleList(
            [
                GenericTransformerBlockByte(
                    hidden_dim=self.byte_hidden,
                    context_window=model_cfg["context_window"],
                    ffn_cfg={
                        "ffn_type": "generic",
                        "ffn_dim": self.byte_hidden*4,
                        "activation": "gelu",
                        "normalization": "rms_norm",
                        "bias": False,
                        "dropout": 0.0
                    },
                    attn_cfg={
                        "attn_type": "standard",
                        "num_kv_heads": 8,
                        "num_q_heads": 8,
                        "normalization": "rms_norm",
                        "is_causal": True,
                        "bias": False,
                        "dropout": 0.0
                    }
                ) for _ in range(self.num_byte_decoder_layers)
            ]
        )

        self.lm_head = torch.nn.Linear(
            in_features=self.byte_hidden, # 128
            out_features=self.byte_vocab_size, # 259 (256 bytes + 3 special)
            bias=False,
        )

    # def _update_tokens(self, x):
    #     """ Update the tokens + 1 """
    #     # input(x.size())
    #     for block in self.transformer:
    #         x = block(
    #             x=x,
    #         )

    #     return x[:, -1, :]

    def _update_tokens(self, x_input, past_key_values):
        """
        Update the tokens with KV caching.

        Args:
            x_input: Input tensor of shape [num_unfinished, current_length, hidden_dim]
            past_key_values: List of past_key_values for each layer

        Returns:
            Tuple[Tensor, List]: Output tensor and updated past_key_values
        """
        new_past_key_values = []

        for layer_idx, block in enumerate(self.transformer):
            past = past_key_values[layer_idx]
            x_input, new_past = block(
                x_input,
                attention_mask=None,  # Update as needed
                past_key_value=past
            )
            new_past_key_values.append(new_past)

        # Return the logits for the last token
        return x_input[:, -1, :], new_past_key_values


    # def forward(self, x, target):
    #     # TODO: do kv caching for the autoregressive generation
    #     """
    #     Forward pass with iterative decoding.

    #     Basically, create an empty (masked) target sequence
    #     where the first four elements are the projected tokens,
    #     then iteratively update this, one token at a time, until 
    #     all tokens are updated.
    #     """
    #     B, S, H = x.size()
    #     B, S, S_c = target.size()
    #     # After projecting x
    #     x = self.projection(x)  # x: [B, S, H]

    #     # Reshape x to have shape [B*S, self.num_global_proj, self.byte_hidden]
    #     x = x.view(B * S, self.num_global_proj, self.byte_hidden)

    #     # Initialize x_output to store the outputs; it has extra space for the generated tokens
    #     x_output = torch.zeros(
    #         (B * S, self.num_global_proj + S_c, self.byte_hidden),
    #         dtype=x.dtype, device=x.device
    #     )

    #     # Set the initial part of x_output with the projected x
    #     x_output[:, :self.num_global_proj, :] = x

    #     # Initialize a mask to track finished sequences
    #     finished = torch.zeros(B * S, dtype=torch.bool, device=x.device)

    #     # Initialize a tensor to track the current length of each sequence
    #     current_lengths = torch.full(
    #         (B * S,), self.num_global_proj, dtype=torch.long, device=x.device
    #     )

    #     for i in range(S_c):
    #         # Identify unfinished sequences
    #         unfinished = ~finished
    #         if not unfinished.any():
    #             # All sequences have finished decoding
    #             break

    #         # Get indices of unfinished sequences
    #         unfinished_idx = unfinished.nonzero(as_tuple=False).squeeze(-1)
    #         num_unfinished = unfinished_idx.size(0)

    #         # Prepare inputs for unfinished sequences
    #         # Determine the maximum current length among unfinished sequences
    #         max_current_length = current_lengths[unfinished_idx].max().item()
    #         # print(max_current_length)
    #         # # Create x_input for unfinished sequences up to the current length
    #         # print(x_output.size())
    #         # print(x_output[unfinished_idx].size())
    #         x_input = x_output[unfinished_idx, :max_current_length, :]

    #         # Call _update_tokens with x_input
    #         generated_step = self._update_tokens(x_input)  # Shape: [num_unfinished, hidden_size]

    #         # Append generated_step to x
    #         x_output[unfinished_idx, max_current_length, :] = generated_step

    #         # Update current lengths
    #         current_lengths[unfinished_idx] += 1

    #         # Store the output tokens
    #         # x_output[unfinished_idx, max_current_length, :] = generated_step

    #         # Get predicted token IDs (assuming generated_step gives logits over vocab)
    #         next_token_logits = generated_step  # Shape: [num_unfinished, vocab_size]
    #         next_token_ids = next_token_logits.argmax(dim=-1)  # Shape: [num_unfinished]

    #         # Check for EOT tokens
    #         is_eot = next_token_ids == self.eos_token_id
    #         finished_idx = unfinished_idx[is_eot]
    #         finished[finished_idx] = True

    #     # reshape output
    #     x_output = x_output.view(B, S, self.num_global_proj + S_c, self.byte_hidden)

    #     return self.lm_head(
    #         x_output[:, :, self.num_global_proj:, :]
    #     )
    def forward(self, x, target):
        B, S, H = x.size()
        _, _, S_c = target.size()

        x = self.projection(x)  # [B, S, H]
        x = x.view(B * S, self.num_global_proj, self.byte_hidden)  # [B*S, num_global_proj, hidden_dim]

        x_output = torch.zeros(
            (B * S, self.num_global_proj + S_c, self.byte_hidden),
            dtype=x.dtype, device=x.device
        )

        # Assuming x after projection is logits over the vocab
        x_output[:, :self.num_global_proj, :] = x

        finished = torch.zeros(B * S, dtype=torch.bool, device=x.device)
        current_lengths = torch.full(
            (B * S,), self.num_global_proj, dtype=torch.long, device=x.device
        )

        # Initialize past_key_values
        num_layers = len(self.transformer)
        past_key_values = [None] * num_layers

        for i in range(S_c):
            unfinished = ~finished
            if not unfinished.any():
                break

            unfinished_idx = unfinished.nonzero(as_tuple=False).squeeze(-1)
            num_unfinished = unfinished_idx.size(0)

            max_current_length = current_lengths[unfinished_idx].max().item()
            x_input = x_output[unfinished_idx, :max_current_length, :]

            # Initialize new_past_key_values for this step
            new_past_key_values = []

            # Pass through transformer blocks with KV caching
            for layer_idx, block in enumerate(self.transformer):
                past = past_key_values[layer_idx]
                x_input_layer, new_past = block(
                    x_input,
                    attention_mask=None,  # Update as needed
                    past_key_value=past
                )
                new_past_key_values.append(new_past)
                x_input = x_input_layer  # Update input for next layer

            # Update past_key_values
            past_key_values = new_past_key_values

            # Get the generated step (last token's output)
            generated_step = x_input[:, -1, :]  # Shape: [num_unfinished, hidden_dim]

            # Append generated_step to x
            x_output[unfinished_idx, max_current_length, :] = generated_step

            # Update current lengths
            current_lengths[unfinished_idx] += 1

            # Store the output tokens
            # x_output[unfinished_idx, max_current_length, :] = generated_step

            # Get predicted token IDs
            next_token_logits = generated_step
            next_token_ids = next_token_logits.argmax(dim=-1)

            # Check for EOT tokens
            is_eot = next_token_ids == self.eos_token_id
            finished_idx = unfinished_idx[is_eot]
            finished[finished_idx] = True

        # Reshape x_output if necessary
        x_output = x_output.view(B, S, self.num_global_proj + S_c, self.byte_hidden)

        return self.lm_head(
            x_output[:, :, self.num_global_proj:, :]
        )

        # print(x_output)
        # input(x_output.size())
        # exit()





        # input(x.size()) # 2, 2445, 384  batch, max num chunks, hidden
        # input(target.size()) # 2, 2017, 19  batch, max num chunks, byte_seq_len, idx
        # B, S, H = x.size()
        # B, S, S_c = target.size()
        # # generated_tokens = torch.zeros(
        # #     (B, S, S_c, self.byte_hidden),
        # #     dtype=x.dtype, device=x.device
        # # )
        # # generated_tokens = torch.full(
        # #     (B, S, S_c, self.byte_hidden),
        # #     self.pad_token_id, dtype=x.dtype, device=x.device            
        # # )

        # # requires_decoding = torch.full(
        # #     (B, S),
        # #     True, dtype=bool, device=x.device
        # # )

        # # generated_tokens = torch.full(
        # #     (B, S, S_
        # # )

        # # output_tokens = torch.full(
        # #     (B, S, S_c, self.byte_vocab_size),
        # #     0, dtype=x.dtype, device=x.device
        # # )
        

        # x = self.projection(x)
        # # reshape
        # x = x.view(B * S, self.num_global_proj, self.byte_hidden) # 2, 1018, 4, 96

        # x_output = torch.zeros(
        #     (B*S, S_c+self.num_global_proj, self.byte_vocab_size)
        # )
        # x_output[:, :self.num_global_proj] = x

        # for i in range(S_c):
        #     # requires_decoding: [2, 561]
        #     # x: [2, 4, 384]
        #     input(x_output.size()) # 2998, 4, 96]
        #     generated_step = self._update_tokens(
        #         x=x
        #     )
        #     input(generated_step.size()) # [2998, 96]

        #     # expand generated steps and concatenate
        #     x = torch.concatenate(
        #         (
        #             x,
        #             generated_step.unsqueeze(1)
        #         ),
        #         dim=1
        #     )
        #     input(x.size())
        #     exit()
            




        # # generated_tokens = torch.full(
        # #     (B, S, 4+)
        # # )




        # B, S, H = x.size()
        # generated_tokens = torch.full(
        #     (batch_size, self.num_global_proj)
        # )




        # batch_size = x.size(0)
        # seq_len = x.size(1)
        # device = x.device
        # max_target_length = target.size(2)

        # # Project global sequence into local sequences
        # # Shape after projection: [batch_size, seq_len, num_global_proj * byte_hidden]
        # x = self.projection(x)

        # # Reshape to [batch_size, seq_len, num_global_proj, byte_hidden]
        # x = x.view(batch_size, seq_len, self.num_global_proj, self.byte_hidden)

        # # Reshape to [batch_size * num_global_proj, seq_len, byte_hidden]
        # x = x.view(batch_size * self.num_global_proj, seq_len, self.byte_hidden)

        # # Initialize the output tensor to collect generated tokens
        # generated_tokens = torch.full(
        #     (batch_size * self.num_global_proj, max_target_length),
        #     self.pad_token_id, dtype=torch.long, device=device
        # )

        # # Start decoding with the first four projected tokens (starting context)
        # decoded_tokens = x[:, :4, :]  # Initial four local tokens
        # sequence_lengths = torch.full((batch_size * self.num_global_proj,), 4, dtype=torch.long, device=device)
        # eos_mask = torch.zeros(batch_size * self.num_global_proj, dtype=torch.bool, device=device)

        # # Run iterative decoding until all sequences have ended or max target length is reached
        # for step in range(4, max_target_length):
        #     if eos_mask.all():
        #         break

        #     # Take last token of each sequence as input for this step
        #     current_input = decoded_tokens[:, -1, :]  # Shape: [batch_size * num_global_proj, byte_hidden]

        #     # Transform current input through the transformer blocks
        #     for layer in self.transformer:
        #         current_input = layer(current_input.unsqueeze(0)).squeeze(0)  # Shape: [batch_size * num_global_proj, byte_hidden]

        #     # Compute logits for the current step
        #     logits = self.lm_head(current_input)  # Shape: [batch_size * num_global_proj, vocab_size]

        #     # Get next token prediction
        #     next_token = torch.argmax(logits, dim=-1)  # Shape: [batch_size * num_global_proj]

        #     # Update generated tokens for active sequences
        #     active_sequences = ~eos_mask  # Mask for active sequences
        #     generated_tokens[active_sequences, step] = next_token[active_sequences]

        #     # Concatenate to decoded tokens
        #     current_input = current_input.unsqueeze(1)  # [batch_size * num_global_proj, 1, byte_hidden]
        #     decoded_tokens = torch.cat((decoded_tokens, current_input), dim=1)  # [batch_size * num_global_proj, step+1, byte_hidden]

        #     # Update sequence lengths for active sequences
        #     sequence_lengths[active_sequences] += 1

        #     # Update eos_mask for sequences that have generated eos_token_id
        #     eos_mask = eos_mask | (next_token == self.eos_token_id)

        # # Reshape generated tokens back to [batch_size, num_global_proj, max_target_length]
        # generated_tokens = generated_tokens.view(batch_size, self.num_global_proj, max_target_length)

        # return generated_tokens

    def inference(self, x):
        """
        inference
        """
        return self.forward(x)[0][:, -1, :, :]