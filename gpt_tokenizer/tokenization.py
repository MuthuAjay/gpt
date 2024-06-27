"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from base import Tokenizer, merge, get_stats
from typing import Optional, List


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self,
              text: str,
              vocab_size: int,
              verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        # iteratively merge the most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes(idx) for idx in range(256)}
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]} had {stats[pair]} occurrences")

        # save the class variables
        self.merges = merges
        self.vocab = vocab

    def decode(self,
               ids: Optional[List[int]]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode('utf-8', errors='replace')
        return text

    def encode(self,
               text: str) -> List[int]:
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) > 1:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
