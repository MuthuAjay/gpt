from typing import List

import torch

from dataset import Dataset
from layers import Embedding, Sequential, Linear, FlattenConsecutive, Tanh, BatchNorm1d
from train import Train

g = torch.Generator().manual_seed(42)


class Main:

    def __init__(self,
                 n_embd: int = 10,
                 n_hidden: int = 100,
                 ):
        self.n_embd = n_embd
        self.n_hidden = n_hidden
        self.g = torch.Generator().manual_seed(42)

    def instantiate_model(self,
                          block_size: int,
                          vocab_size: int,
                          ):

        # model = Sequential([
        #     Embedding(vocab_size, self.n_embd),
        #     FlattenConsecutive(2), Linear(self.n_embd * 2, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden),
        #     Tanh(),
        #     FlattenConsecutive(2), Linear(self.n_hidden * 2, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden),
        #     Tanh(),
        #     FlattenConsecutive(2), Linear(self.n_hidden * 2, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden),
        #     Tanh(),
        #     FlattenConsecutive(2), Linear(self.n_hidden * 2, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden),
        #     Tanh(),
        #     Linear(self.n_hidden, vocab_size),
        # ])

        model1 = Sequential(
            [Embedding(vocab_size, self.n_embd)] + self.initianlze_layers(block_size=block_size,
                                                                          wavnet_dim=2) + [
                Linear(self.n_hidden, vocab_size), ]
        )
        return model1

    def _layers_count(self,
                      block_size: int,
                      wavenet_dim: int):
        layer_count = 0
        if wavenet_dim == 2:
            while block_size > 0:
                if block_size % 2 == 0:
                    layer_count += 1
                    block_size /= 2
                else:
                    block_size = 0

        print(layer_count)
        return layer_count

    def initianlze_layers(self,
                          block_size: int,
                          wavnet_dim: int = 2):
        layers = []
        num_layers = self._layers_count(block_size, wavnet_dim)
        for i in range(num_layers):
            if i == 0:
                layers = layers + [FlattenConsecutive(wavnet_dim), Linear(self.n_embd * 2, self.n_hidden, bias=False),
                                   BatchNorm1d(self.n_hidden), Tanh()]
            else:
                layers += [FlattenConsecutive(wavnet_dim), Linear(self.n_hidden * 2, self.n_hidden, bias=False),
                           BatchNorm1d(self.n_hidden), Tanh()]
        print(len(layers))
        return layers

    def initialize_weights(self,
                           model: Sequential):
        # parameter init
        with torch.no_grad():
            model.layers[-1].weight *= 0.1  # last layer make less confident
        return model

    def parameters(self,
                   model: Sequential) -> List:
        # parameters for back propagation

        parameters = model.parameters()
        print(sum(p.nelement() for p in parameters))  # number of parameters in total
        for p in parameters:
            p.requires_grad = True

        return parameters

    def __call__(self,
                 block_size: int = 3,
                 max_steps: int = 100,
                 lr: float = 0.1,
                 ):
        dataset = Dataset(path=r'C:\Users\CD138JR\Documents\GitHub\gpt\names.txt',
                          block_size=block_size)
        Xtr, Ytr, Xdev, Ydev, Xte, Yte = dataset.get_dataset()
        vocab_size = dataset.vocab_size

        model = self.instantiate_model(block_size=block_size,
                                       vocab_size=vocab_size)
        model = self.initialize_weights(model)
        parameters = self.parameters(model=model)
        train = Train(batch_size=32,
                      lr=0.1,
                      max_steps=max_steps)
        layers, parameters, lossi, ud = train(Xtr=Xtr,
                                              Ytr=Ytr,
                                              parameters=parameters,
                                              model=model)


if __name__ == "__main__":
    add_norm_values = [False, True]
    learning_rates = [0.1, 0.01]

    main = Main(n_embd=10,
                n_hidden=100)
    # train_settings = {
    #     'lr': lr,
    #     'weight_init': weight_init
    # }
    main(
        block_size=8,
        max_steps=1000
    )
