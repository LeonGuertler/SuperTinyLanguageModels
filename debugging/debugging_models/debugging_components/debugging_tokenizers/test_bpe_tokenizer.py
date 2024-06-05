"""
For the BPE tokenizer, test training, and inference.
"""

# import os

# import pytest
# import torch

# from models.components.tokenizers import build_tokenizer
# from models.components.tokenizers.utils import get_tokenizer_path

# def test_train_bpe_tokenizer():
#     """
#     Train the BPE tokenizer.
#     """
#     tokenizer = build_tokenizer(
#         tokenizer_type="bpe", vocab_size=259, dataset_name="simple_en_wiki"
#     )

#     text = "Hello, world!"

#     tokens = tokenizer.encode(text)
#     assert tokenizer.decode(tokens) == text

#     # delete the trained tokenizer file

#     os.remove(
#         get_tokenizer_path(
#             tokenizer_type="bpe", vocab_size=259, dataset_name="simple_en_wiki"
#         )[1]
#     )
