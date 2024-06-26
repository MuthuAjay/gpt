import unicodedata
from typing import Optional, List, Dict, Tuple


def get_stats(ids: Optional[List[int]],
              counts=None) -> Dict:
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: Optional[List[int]],
          pair: Tuple[int],
          idx: int
          ) -> List[int]:
    new_ids = []
    i = 0

    while i < len(ids):

        if ids[i] == pair[0] and i < len(ids) and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids


# helper function
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table

    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape

    return "".join(chars)


def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)

    return s


# ------------------------------------------------------------

# the base tokenizers class

class Tokenizer:
    """Base Class for Tokenizer"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {}
        self.pattern = {}
        self.special_tokens = {}
        self.vocab = self.build_vocab()

    def train(self):
        raise NotImplementedError

    def encode(self):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    def build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes(idx) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
