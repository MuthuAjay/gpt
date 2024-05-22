from pathlib import Path
from typing import List, Dict, Tuple
import torch
import random

class Dataset:

    def __init__(self,
                 path: str | Path,
                 block_size: int = 3):
        """Initialize the Dataset object.

        Args:
            path (str or Path): Path to the dataset file.
            block_size (int, optional): Size of the context window. Defaults to 3.
        """
        self.vocab_size = None
        self.block_size = block_size
        self.file_path = Path(path) if isinstance(path, str) else path
        self.words = self.get_data()
        self.stoi = self.string_to_index(self.words)
        self.itos = self.index_to_string(self.stoi)

    def get_data(self) -> List:
        """Read the dataset file and return a list of words.

        Returns:
            List: List of words read from the dataset file.
        """
        return open(self.file_path, 'r').read().splitlines()

    def string_to_index(self,
                        words: List[str]):
        """Map characters to indices.

        Args:
            words (List[str]): List of words.

        Returns:
            Dict: Mapping of characters to indices.
        """
        chars = sorted(list(set(''.join(words))))
        stoi = {s: i + 1 for i, s in enumerate(chars)}
        stoi['.'] = 0
        self.vocab_size = len(stoi)
        return stoi

    def index_to_string(self,
                        stoi: Dict) -> Dict:
        """Map indices to characters.

        Args:
            stoi (Dict): Mapping of characters to indices.

        Returns:
            Dict: Mapping of indices to characters.
        """
        return {i: s for s, i in stoi.items()}

    def build_dataset(self, words):
        """Build the dataset for training.

        Args:
            words (List[str]): List of words.

        Returns:
            Tuple: Tuple containing X (input) and Y (target) tensors.
        """
        assert isinstance(words, list), 'words must be a list'

        X, Y = [], []

        for w in words:
            context = [0] * self.block_size
            for ch in w + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        print(X.shape, Y.shape)
        return X, Y

    def get_dataset(self) -> Tuple:
        """Get the training, validation, and test datasets.

        Returns:
            Tuple: Tuple containing training, validation, and test datasets.
        """
        random.seed(42)
        random.shuffle(self.words)
        n1 = int(0.8 * len(self.words))
        n2 = int(0.9 * len(self.words))
        Xtr, Ytr = self.build_dataset(words=self.words[:n1])
        Xdev, Ydev = self.build_dataset(words=self.words[n1:n2])
        Xte, Yte = self.build_dataset(words=self.words[n2:])
        return Xtr, Ytr, Xdev, Ydev, Xte, Yte
