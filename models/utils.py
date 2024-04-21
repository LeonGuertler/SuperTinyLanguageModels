"""
Some general util functions
"""

import torch
import pandas as pd
from collections import Counter, defaultdict


def print_model_stats(model):
    """
    Print relevant model statistics, including the number of parameters
    with and without embeddings for a given PyTorch model, formatted for better readability.
    """
    total_params = sum(p.numel() for p in model.parameters())

    embeddings_params = sum(p.numel() for p in model.embedder.parameters())
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())

    # Check if the parameters are shared
    shared_embedding = model.embedder.embedding.weight is model.lm_head.linear.weight
    if shared_embedding:
        core_model_params = total_params - embeddings_params
        lm_head_and_embeddings_params = lm_head_params
    else:
        core_model_params = total_params - embeddings_params - lm_head_params
        lm_head_and_embeddings_params = lm_head_params + embeddings_params

    # Format the numbers for better readability
    def format_number(n):
        if n >= 1e6:
            return f"{n / 1e6:.2f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.2f}K"
        return str(n)

    # Prepare the data
    data = {
        "Component": ["Total", "LM Head + Embeddings", "Core Model"],
        "Parameters": [format_number(total_params), format_number(lm_head_and_embeddings_params), format_number(core_model_params)]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Print the table
    print(df.to_string(index=False))


# utils for the tokenizer 

def get_stats(ids):   # using collections.Counter
    return Counter(zip(ids, ids[1:]))


def multi_merge(ids, pairs):
    skip = False
    newids = [(pairs[(ids[i], ids[i+1])] if (ids[i], ids[i+1]) in pairs and (skip := True) else ids[i])
              for i in range(len(ids) - 1)
              if not skip or (skip := False)]
    if not skip:   # if the last pair was not replaced, append the last token
        newids.append(ids[-1])
    return newids

# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s