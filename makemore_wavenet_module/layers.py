import torch


# Linear Layer
class Linear:
    def __init__(self,
                 fan_in: int,
                 fan_out: int,
                 bias=True):
        self.weight = torch.randn((fan_in, fan_out)) // fan_in ** 0.5  # note kaiming init
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self,
                 x: torch.Tensor):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


# Batch Normalisation Layer

class BatchNorm1d:

    def __init__(self,
                 dim: int,
                 eps: int | float = 1e-5,
                 momentum: int | float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self,
                 x: torch.Tensor):

        # calculate the forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalise to unit variance
        self.out = self.gamma * xhat + self.beta

        # Update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# Activation Layer
class Tanh:
    def __call__(self,
                 x: torch.Tensor):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


# Module for Embedding

class Embedding:

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self,
                 IX: torch.Tensor):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


# Module for Flatten
class FlattenConsecutive:

    def __init__(self,
                 n: int):
        self.n = n

    def __call__(self,
                 x: torch.Tensor):
        B, T, C = x.size()
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []


# Module for Sequential

class Sequential:

    def __init__(self,
                 layers: list):
        self.layers = layers

    def __call__(self,
                 x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
