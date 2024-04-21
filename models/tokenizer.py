"""
Collection of tokenizers.
"""



import tiktoken  
import os 

import unicodedata

from models.utils import (
    replace_control_characters,
    get_stats,
    multi_merge,
    render_token
)

from trainers.utils import load_data
from heapq import nlargest

# based on https://colab.research.google.com/drive/1S4TbDqHWdLH_uwQfU9EpNWA_n81aCa2D?usp=sharing#scrollTo=KkFXOwHccfu7
# and the karpathy minbpe
class CustomBPE:
    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=True):

        # input text preprocessing 
        text_bytes = text.encode("utf-8")
        text_bytes = [*map(int, text_bytes)]
        ids = list(text_bytes)
        current_vocab_size = 256
        num_merges = vocab_size - current_vocab_size
        max_clutch_size = 64

        # iteratively merge the most frequent pair
        merges = {}  # (int, int) -> int 

        while num_merges > 0:
            stats = get_stats(ids)
            top_pairs = nlargest(
                min(max_clutch_size, num_merges),
                stats,
                key=stats.get
            )
            pairs_to_merge = {}
            first_seen = set()
            second_seen = set()
            for pair in top_pairs:
                if pair[0] in second_seen or pair[1] in first_seen:
                    first_seen.add(pair[0])
                    second_seen.add(pair[1])
                    continue # skip this pair but keep looking for mergeable top_pairs
                first_seen.add(pair[0])
                second_seen.add(pair[1])
                pairs_to_merge[pair] = current_vocab_size
                current_vocab_size += 1
                num_merges -= 1
            ids = multi_merge(ids, pairs_to_merge)
            merges.update(pairs_to_merge)
            input(merges)



        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()









def load_custom_bpe(bpe_name):
    """
    Load a custom BPE tokenizer
    """
    input('loading bpe')
    # check if exists
    if os.path.exists(bpe_name):
        bpe = CustomBPE()
        bpe.load(bpe_name)
        return bpe
    else:
        # create and train with necessary prints
        print(f"Training BPE model from scratch")

        # load the text string to train it on
        text = "".join(load_data("simple_en_wiki")["train"]["text"])


        bpe = CustomBPE()
        bpe.train(
            text,
            vocab_size=int(bpe_name.split("-")[-1].split(".")[0]),
            verbose=True
        )
        bpe.save(bpe_name)


TOKENIZER_DICT = {
    "gpt2": lambda: tiktoken.get_encoding("gpt2"),
    "bpe-4096": lambda: load_custom_bpe("tokenizers/bpe-4096.model")
}

def build_tokenizer(tokenizer_name):
    """
    Get the tokenizer from the dictionary
    """
    return TOKENIZER_DICT[tokenizer_name]()