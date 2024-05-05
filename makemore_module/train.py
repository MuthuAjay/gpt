import torch
from torch.nn import functional as F
from typing import List

g = torch.Generator().manual_seed(42)


class Train:

    def __init__(self,
                 batch_size: int = 32,
                 lr: float = 0.1,
                 max_steps: int = 20000):
        self.batch_size = batch_size
        self.lr = lr
        self.max_steps = max_steps

    def train(self,
              layers: List[object],
              parameters: List,
              Xtr,
              Ytr,
              C):
        # same optimization as the last time
        lossi = []
        ud = []  # update to data ratio that is udpate to the gradient to the data ratio

        for i in range(self.max_steps):

            # minibatch construct 
            ix = torch.randint(0, Xtr.shape[0], (self.batch_size,), generator=g)
            Xb, Yb = Xtr[ix], Ytr[ix]  # batch X,Y

            # forward pass
            emb = C[Xb]  # embed the characters into vectors
            x = emb.view(emb.shape[0], -1)  # concatenate the vectors
            for layer in layers:
                x = layer(x)
            loss = F.cross_entropy(x, Yb)  # loss function

            # backward pass
            for layer in layers:
                layer.out.retain_grad()  # after Debug: would take out retian_graph
            for p in parameters:
                p.grad = None
            loss.backward()

            # update
            lr = 0.1 if i < 100000 else 0.01  # step learning rate decay
            for p in parameters:
                p.data += -lr * p.grad

            # tarck stats
            if i % 10000 == 0:
                print(f'{i:7d}/{self.max_steps:7d}: {loss.item():4f}')
            lossi.append(loss.log10().item())
            with torch.no_grad():
                ud.append([((lr * p.grad).std() / p.data.std()).log10().item() for p in parameters])

            if i >= 1000:
                break

        return layers, parameters, lossi, ud


