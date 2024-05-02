"""
A collection of utils for the tokenizers.
"""

import os
import unicodedata
from collections import Counter

import hydra  # to get the absolute path to the tokenizer


def get_tokenizer_path(tokenizer_type, vocab_size, dataset_name):
    """
    Get the path to the tokenizer.
    """
    tokenizer_folder = hydra.utils.to_absolute_path("tokenizers")
    tokenizer_full_path = os.path.join(
        tokenizer_folder, f"{tokenizer_type}_{dataset_name}_{vocab_size}.model"
    )
    return tokenizer_folder, tokenizer_full_path


def check_if_tokenizer_exists(tokenizer_type, vocab_size, dataset_name):
    """
    Check if the tokenizer already exists.
    """
    _, tokenizer_path = get_tokenizer_path(tokenizer_type, vocab_size, dataset_name)
    return os.path.exists(tokenizer_path)


def get_stats(ids):
    return Counter(zip(ids, ids[1:]))


def multi_merge(ids, pairs):
    skip = False
    newids = [
        (
            pairs[(ids[i], ids[i + 1])]
            if (ids[i], ids[i + 1]) in pairs and (skip := True)
            else ids[i]
        )
        for i in range(len(ids) - 1)
        if not skip or (skip := False)
    ]
    if not skip:  # if the last pair was not replaced, append the last token
        newids.append(ids[-1])
    return newids


def merge(ids, pair, idx):
    skip = False
    newids = [
        (
            idx
            if (ids[i] == pair[0] and ids[i + 1] == pair[1] and (skip := True))
            else ids[i]
        )
        for i in range(len(ids) - 1)
        if not skip or (skip := False)
    ]
    if not skip:  # if the last pair was not replaced, append the last token
        newids.append(ids[-1])
    return newids


def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode("utf-8", errors="replace")
    s = replace_control_characters(s)
    return s
