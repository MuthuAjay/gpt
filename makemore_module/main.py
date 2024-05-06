import torch
from activations import Tanh, BatchNorm1d
from plot import Plot
from typing import List
from model import Linear
from dataset import Dataset
from train import Train

g = torch.Generator().manual_seed(42)


class Main:

    def __init__(self,
                 n_embd: int = 10,
                 n_hidden: int = 100,
                 add_norm: bool = False,
                 ):
        self.n_embd = n_embd
        self.n_hidden = n_hidden
        self.g = torch.Generator().manual_seed(42)
        self.add_norm = add_norm

    def instantiate_embeddings(self, vocab_size, n_embd):
        return torch.randn((vocab_size, n_embd), generator=g)

    def instantiate_layers(self,
                           block_size: int,
                           vocab_size: int,
                           ):
        if not self.add_norm:
            layers = [
                Linear(self.n_embd * block_size, self.n_hidden), Tanh(),
                Linear(self.n_hidden, self.n_hidden), Tanh(),
                Linear(self.n_hidden, self.n_hidden), Tanh(),
                Linear(self.n_hidden, self.n_hidden), Tanh(),
                Linear(self.n_hidden, self.n_hidden), Tanh(),
                Linear(self.n_hidden, vocab_size)
            ]
        else:
            layers = [
                Linear(self.n_embd * block_size, self.n_hidden),BatchNorm1d(self.n_hidden), Tanh(),
                Linear(self.n_hidden, self.n_hidden), BatchNorm1d(self.n_hidden),Tanh(),
                Linear(self.n_hidden, self.n_hidden),BatchNorm1d(self.n_hidden), Tanh(),
                Linear(self.n_hidden, self.n_hidden), BatchNorm1d(self.n_hidden),Tanh(),
                Linear(self.n_hidden, self.n_hidden), BatchNorm1d(self.n_hidden),Tanh(),
                Linear(self.n_hidden, vocab_size), BatchNorm1d(vocab_size)
            ]

        return layers

    def initialize_weights(self,
                           layers: List[object],
                           final_layer_weights: float = 0.1,
                           other_weights: float = 5 / 3):
        with torch.no_grad():
            # last layer : make less confident
            if not self.add_norm:
                layers[-1].weight *= final_layer_weights
            else:
                layers[-1].gamma *= final_layer_weights
            # all other layers: apply again
            for layer in layers[:-1]:
                if isinstance(layer, Linear):
                    layer.weight *= other_weights
        return layers

    def parameters(self,
                   layers: List[object]) -> List:
        # parameters for back propagation

        parameters = [self.C] + [p for layer in layers for p in layer.parameters()]
        print(sum(p.nelement() for p in parameters))  # number of parameters in total
        for p in parameters:
            p.requires_grad = True

        return parameters

    def __call__(self,
                 block_size: int = 3,
                 max_steps : int = 100):
        dataset = Dataset(path=r'C:\Users\CD138JR\Documents\GitHub\gpt\names.txt',
                          block_size=3)
        Xtr, Ytr, Xdev, Ydev, Xte, Yte = dataset.get_dataset()
        vocab_size = dataset.vocab_size
        self.C = self.instantiate_embeddings(vocab_size=vocab_size,
                                             n_embd=self.n_embd)
        layers = self.instantiate_layers(block_size=3,
                                         vocab_size=vocab_size)
        layers = self.initialize_weights(layers=layers)
        parameters = self.parameters(layers=layers)
        train = Train(batch_size=32,
                      lr=0.1,
                      max_steps=max_steps)
        layers, parameters, lossi, ud = train.train(Xtr=Xtr,
                                                    Ytr=Ytr,
                                                    parameters=parameters,
                                                    C=self.C,
                                                    layers=layers)
        Plot.plot_activation_distribution(layers=layers,
                                          act_fn=Tanh)
        Plot.plot_gradient_distribution(layers=layers,
                                        act_fn=Tanh)
        Plot.weights_gradient_distribution(parameters=parameters)
        Plot.plot_updates_to_gradients_ratio(parameters=parameters,
                                             ud=ud)


if __name__ == "__main__":
    add_norm = [False, True]

    for val in add_norm:
        main = Main(n_embd=10,
                    n_hidden=100,
                    add_norm=val)
        main(max_steps= 1000)
