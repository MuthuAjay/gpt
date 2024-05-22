import torch
from plot import Plot
from typing import List
from dataset import Dataset
from train import Train
from layers import Embedding, Sequential, Linear, FlattenConsecutive, Tanh, BatchNorm1d

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

    def instantiate_model(self,
                          block_size: int,
                          vocab_size: int,
                          ):
        model = Sequential([
            Embedding(vocab_size, self.n_embd),
            FlattenConsecutive(2), Linear(self.n_embd * 2, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden),
            Tanh(),
            FlattenConsecutive(2), Linear(self.n_hidden * 2, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden),
            Tanh(),
            FlattenConsecutive(2), Linear(self.n_hidden * 2, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden),
            Tanh(),
            FlattenConsecutive(2), Linear(self.n_hidden * 2, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden),
            Tanh(),
            Linear(self.n_hidden, vocab_size),
        ])
        return model

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
                          block_size=3)
        Xtr, Ytr, Xdev, Ydev, Xte, Yte = dataset.get_dataset()
        vocab_size = dataset.vocab_size

        model = self.instantiate_model(block_size=3,
                                       vocab_size=vocab_size)
        model = self.initialize_weights(model)
        parameters = self.parameters(model=model)
        train = Train(batch_size=32,
                      lr=0.1,
                      max_steps=max_steps)
        layers, parameters, lossi, ud = train(Xtr=Xtr,
                                              Ytr=Ytr,
                                              parameters=parameters,
                                              C=self.C,
                                              model=model)
        Plot.plot_activation_distribution(layers=layers,
                                          act_fn=Tanh)
        Plot.plot_gradient_distribution(layers=layers,
                                        act_fn=Tanh)
        Plot.weights_gradient_distribution(parameters=parameters)
        Plot.plot_updates_to_gradients_ratio(parameters=parameters,
                                             ud=ud)


if __name__ == "__main__":
    add_norm_values = [False, True]
    learning_rates = [0.1, 0.01]
    weight_initializations = [[0.1, 1.1], [0.2, 1.2], [0.5, 2]]

    for add_norm_val in add_norm_values:
        for lr in learning_rates:
            for weight_init in weight_initializations:
                main = Main(n_embd=10,
                            n_hidden=100,
                            add_norm=add_norm_val)
                train_settings = {
                    'lr': lr,
                    'weight_init': weight_init
                }
                print(f"Training with add_norm={add_norm_val}, lr={lr}, weight_init={weight_init}")
                main(max_steps=1000, **train_settings)
