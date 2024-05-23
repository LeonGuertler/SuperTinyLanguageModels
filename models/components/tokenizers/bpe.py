"""
A simple implementation of the Byte Pair Encoding tokenizer, based on
https://github.com/karpathy/minbpe and sped up using https://t.co/MkTecNoWNP
by https://twitter.com/lexandermorgan/status/1778793836929495098.

Original Paper: https://arxiv.org/abs/1508.07909v5
"""
import torch 
import os
from heapq import nlargest

from tqdm import tqdm

from models.components.tokenizers import utils
from models.components.tokenizers.base_class import Tokenizer
from trainers.utils import load_data


class BPETokenizer(Tokenizer):
    """Tokenizer for Byte Pair Encoding."""

    def __init__(self, vocab_size, dataset_name):
        """
        Check if the specific tokenizer already exists, if not, create it.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dataset_name = dataset_name
        self.special_tokens = {
            "<|pad|>": vocab_size - 2,
            "<|endoftext|>": vocab_size - 1,
        }
        self.pad_token = self.special_tokens["<|pad|>"]
        self.eot_token = self.special_tokens["<|endoftext|>"]

        assert self.vocab_size >= 256 + len(
            self.special_tokens
        ), f"Vocab size too small! Must be > {256+len(self.special_tokens)})"

        if not utils.check_if_tokenizer_exists(
            tokenizer_type="bpe", vocab_size=vocab_size, dataset_name=dataset_name
        ):
            # train the tokenizer and save it
            self._train_tokenizer()
            self._save()

        else:
            # load the stored tokenizer
            self._load()

        self.eot_token = self.special_tokens["<|endoftext|>"]

    def encode(self, text):
        """
        Encode the text into Byte Pair Encoding tokens.
        """
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = utils.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = utils.merge(ids, pair, idx)
        return ids

    def encode_batch(self, texts):
        """
        Encode a batch of texts into Byte Pair Encoding tokens.
        """
        return [self.encode(text) for text in texts]

    def decode(self, tokens):
        """
        Decode the Byte Pair Encoding tokens back into text.
        """
        # if tensor, convert to list
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        text_bytes = b"".join(self.vocab[idx] for idx in tokens)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def decode_batch(self, token_lists):
        """
        Decode a batch of Byte Pair Encoding token lists back into text.
        """
        # if tensor, convert to list
        if torch.is_tensor(token_lists):
            token_lists = token_lists.tolist()
        return [self.decode(tokens) for tokens in token_lists]

    def _train_tokenizer(self, verbose=True):
        """
        Train the Byte Pair Encoding tokenizer
        on the given dataset.
        """
        # load the dataset
        dataset = load_data(dataset_name=self.dataset_name)

        # convert it into a large string
        dataset_text = "".join(dataset["train"]["text"])

        # preprocess the input text
        text_bytes = dataset_text.encode("utf-8")
        text_bytes = [*map(int, text_bytes)]
        ids = list(text_bytes)
        current_vocab_size = 256
        num_merges = self.vocab_size - current_vocab_size - len(self.special_tokens)
        max_clutch_size = 64

        # iteratively merge the most frequent pair
        merges = {}  # (int, int) -> int

        with tqdm(total=num_merges, desc="Training BPE", disable=not verbose) as pbar:
            while num_merges > 0:
                stats = utils.get_stats(ids)
                top_pairs = nlargest(
                    min(max_clutch_size, num_merges), stats, key=stats.get
                )
                pairs_to_merge = {}
                first_seen = set()
                second_seen = set()
                for pair in top_pairs:
                    if pair[0] in second_seen or pair[1] in first_seen:
                        first_seen.add(pair[0])
                        second_seen.add(pair[1])
                        continue  # skip this pair but keep looking for mergeable top_pairs
                    first_seen.add(pair[0])
                    second_seen.add(pair[1])
                    pairs_to_merge[pair] = current_vocab_size
                    current_vocab_size += 1
                    num_merges -= 1
                    pbar.update(1)

                ids = utils.multi_merge(ids, pairs_to_merge)
                merges.update(pairs_to_merge)

        # save as class variable
        self.merges = merges
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        """
        Build the vocabulary from the merges.
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def _save(self):
        """
        Save the tokenizer as a .model file, and save the vocabulary
        for easy debugging as a .vocab file.
        """
        tokenizer_folder, tokenizer_path = utils.get_tokenizer_path(
            tokenizer_type="bpe",
            vocab_size=self.vocab_size,
            dataset_name=self.dataset_name,
        )
        # create folder if necessary
        if not os.path.exists(tokenizer_folder):
            os.makedirs(tokenizer_folder)

        # store the .model file
        # pylint: disable=unspecified-encoding
        with open(tokenizer_path, "w") as f:
            # write the merges
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # pylint: enable=unspecified-encoding

        # store the .vocab file
        vocab_path = tokenizer_path.replace(".model", ".vocab")
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_path, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # try rendering the tokens
                s = utils.render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = utils.render_token(self.vocab[idx0])
                    s1 = utils.render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def _load(self):
        """
        Load the .model file of merges and build
        the vocabulary.
        """
        _, tokenizer_path = utils.get_tokenizer_path(
            tokenizer_type="bpe",
            vocab_size=self.vocab_size,
            dataset_name=self.dataset_name,
        )

        merges = {}
        idx = 256
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.vocab = self._build_vocab()
