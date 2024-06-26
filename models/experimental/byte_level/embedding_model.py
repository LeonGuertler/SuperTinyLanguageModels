"""
A collection of embedding models. A collection model includes
the tokenizer(s), token embeddings and positional encodings
(if necessary).
"""

import torch

from models.components.positional_encoding import LearnedPosEncoding
from models.components.tokenizers import build_tokenizer
from models.embedding_models import EmbedderInterface
from models.experimental.byte_level.layers import ByteLevelTransformerBlock


class ByteLevelEmbedder(EmbedderInterface):
    """
    Takes byte level encodings, processes them via
    two local-attention transformer blocks and pools
    the resultant tokens based on gpt-2 tokenizer
    boundaries.
    Inputs are batches of lists of token blocks
    in the gpt2 tokenizer boundaries.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

        # build the tokenizers
        self.byte_tokenizer = build_tokenizer(
            tokenizer_type=model_cfg["embedder"]["byte_tokenizer_type"],
            vocab_size=model_cfg["byte_vocab_size"],
            dataset_name=model_cfg["embedder"]["dataset_name"],
        )
        self.pooling_tokenizer = build_tokenizer(
            tokenizer_type=model_cfg["embedder"]["tokenizer_type"],
            vocab_size=model_cfg["vocab_size"],
            dataset_name=model_cfg["embedder"]["dataset_name"],
        )

        # positional encodings
        self.pos_encoder = LearnedPosEncoding(
            hidden_dim=model_cfg["byte_embedding_dim"],
            context_window=model_cfg["byte_context_window"],
        )

        # build the token embeddings
        self.byte_token_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["byte_vocab_size"],
            embedding_dim=model_cfg["byte_embedding_dim"],
        )

        # build the transformer blocks
        self.transformer = torch.nn.ModuleList(
            [
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_embedding_dim"],
                    output_dim=model_cfg["byte_embedding_dim"] * 2,
                    ffn_dim=model_cfg["byte_embedding_dim"] * 4,
                    context_window=model_cfg["byte_context_window"],
                    use_rope=False,
                ),
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_embedding_dim"]*2,
                    output_dim=model_cfg["byte_embedding_dim"] * 2,
                    ffn_dim=model_cfg["byte_embedding_dim"] * 8,
                    context_window=model_cfg["byte_context_window"],
                    use_rope=False,
                ),
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_embedding_dim"]*2,
                    output_dim=model_cfg["byte_embedding_dim"] * 2,
                    ffn_dim=model_cfg["byte_embedding_dim"] * 8,
                    context_window=model_cfg["byte_context_window"],
                    use_rope=False,
                ),
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_embedding_dim"] * 2,
                    output_dim=model_cfg["hidden_dim"],
                    ffn_dim=model_cfg["byte_embedding_dim"] * 8,
                    context_window=model_cfg["byte_context_window"],
                    use_rope=False,
                ),
            ]
        )

    def tokenize_input(self, input_string: str, truncate=False, add_eot=True):
        """Tokenize an input string.

        In this case we actually want to pre-tokenize using the pooling tokenizer,
        the byte tokenizer is then used in the forward pass. Its a bit complicated...
        """
        pooling_ids = self.pooling_tokenizer.encode(input_string)
        if add_eot:
            pooling_ids += [self.pooling_tokenizer.eot_token]
        if truncate:
            pooling_ids = self.truncate([pooling_ids])[0]
        tokens = [
            self.byte_tokenizer.encode(self.pooling_tokenizer.decode([pool_id]))
            for pool_id in pooling_ids
        ]
        # truncate bytes
        tokens = [
            token_seq[: self.model_cfg["byte_context_window"]] for token_seq in tokens
        ]
        # pad bytes
        tokens = [
            token_seq
            + [self.byte_tokenizer.pad_token]
            * (self.model_cfg["byte_context_window"] - len(token_seq))
            for token_seq in tokens
        ]
        return tokens

    def pad_batch(self, token_lists, direction="right"):
        """
        Pad the batch of token lists into a tensor
        Args:
            token_lists: list of lists of tokens
            direction: "right" or "left" - whether to add
                padding to the right or left of the tokens
        Returns:
            padded_token_lists: torch.tensor(B, S, S_c)
            mask: torch.tensor(B, S, S_c)
        """
        max_len = max([len(token_list) for token_list in token_lists])
        padded_token_lists = []
        mask = []
        byte_context_window = self.model_cfg["byte_context_window"]
        for token_list in token_lists:
            if direction == "right":
                padded_token_list = token_list + [
                    [self.byte_tokenizer.pad_token]
                    * byte_context_window
                ] * (max_len - len(token_list))
                padded_token_lists.append(padded_token_list)
                mask.append(
                    [1] * len(token_list)
                    + [0] * (max_len - len(token_list))
                )
            else:
                padded_token_list = token_list + [
                    [self.byte_tokenizer.pad_token]
                    * byte_context_window
                ] * (max_len - len(token_list))
                padded_token_lists.append(padded_token_list)
                mask.append(
                    [0] * (max_len - len(token_list))
                    + [1] * len(token_list)
                )
            # expand the mask to include the byte context window
            mask[-1] = [[it] * byte_context_window for it in mask[-1]]
        return torch.tensor(padded_token_lists), torch.tensor(mask)

    def truncate(self, token_lists):
        # get model max length
        max_length = self.model_cfg["context_window"]
        return [token_seq[-max_length:] for token_seq in token_lists]

    def decode(self, list_of_token_idss):
        """
        Decode the token ids.
        """
        return_strings = []
        for list_of_token_ids in list_of_token_idss:
            return_string = ""
            for token_ids in list_of_token_ids:
                token_ids = [
                    token_id
                    for token_id in token_ids
                    if token_id != self.byte_tokenizer.pad_token
                ]
                return_string += self.byte_tokenizer.decode(token_ids)
            return_strings.append(return_string)
        return return_strings

    def forward(self, token_ids):
        """
        Forward pass.
        """
        # get the byte embeddings
        x = self.byte_token_embedder(token_ids)

        # collapse the text sequence and batch dim
        B, S, S_c, H_c = x.size()
        x = x.view(B * S, S_c, H_c)

        # positional encoding
        x = self.pos_encoder(x)

        # pass through transformer
        for block in self.transformer:
            x = block(x)

        # un-collapse the text sequence and batch dim
        # mean pool the tokens
        x = x.mean(dim=-2)
        x = x.view(B, S, -1)

        return x

    def get_sequence_info(self, x):
        """
        Given a batch of sequences of tokens, return
        the token lengths and total number of bytes per
        sequence.
        Args:
            x: torch.tensor(B, S, S_c)
        """
        # flatten along S, S_c
        B, S, S_c = x.size()
        x = x.view(B, S * S_c)

        sequence_char_lengths = []
        # then we decode everything
        # batch decode
        sequences = self.byte_tokenizer.decode_batch(x)
        for seq in sequences:
            sequence_char_lengths.append(len(seq))

        # obtain the mask for end-of-word and pad tokens
        mask = x != self.byte_tokenizer.pad_token
        mask = mask & (x != self.byte_tokenizer.eot_token)

        return sequence_char_lengths, mask
