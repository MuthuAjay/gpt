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

    def train(self,
              text: str,
              vocab_size: int,
              verbose=False):
        raise NotImplementedError

    def encode(self,
               text: str):
        raise NotImplementedError

    def decode(self,
               ids: Optional[List[int]]) -> str:
        raise NotImplementedError

    def build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes(idx) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        :param file_prefix:
        saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """

        # write the model: to be used in load() later

        model_file = file_prefix + '.model'
        with open(model_file, 'w') as f:
            # write the version, pattern and merges
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")

            # write the special tokens, first the total number of them and later each
            f.write(f"{len(self.special_tokens)}")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            # the merge dict

            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        # write the vocab: for the human to look
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}

        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, i f any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """
        Inverse of save() but only for the model file
        :param model_file:
        :return:
        """
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256

        with open(model_file, 'r', encoding='utf-8') as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the special tokens
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
        self.vocab = self.build_vocab()
