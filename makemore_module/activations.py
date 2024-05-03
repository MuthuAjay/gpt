import torch


class BatchNorm1d:
    def __init__(self,
                 dims: int,
                 eps: float = 1e-5,
                 momentum=0.5) -> None:
        """Initialize the Batch Normalization layer.

        Args:
            dims (int): Number of dimensions in the input tensor.
            eps (float, optional): Small value to prevent division by zero in normalization. Defaults to 1e-5.
            momentum (float, optional): Momentum factor for updating the running mean and variance. Defaults to 0.5.
        """
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # parameters (trained with back propagation)
        self.gamma = torch.ones(dims)
        self.beta = torch.zeros(dims)

        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dims)
        self.running_var = torch.ones(dims)

    def __call__(self,
                 x: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as x.
        """
        if self.training:
            xmean = x.mean(0, keepdim=True)  # batch mean
            xvar = x.var(0, keepdim=True, unbiased=True)  # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(x.var + self.eps)  # Normalize to unit variance
        self.out = self.gamma * xhat + self.beta

        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        """Return the parameters of the BatchNorm1d layer.

        Returns:
            List[torch.Tensor]: List containing gamma and beta tensors.
        """
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self,
                 x: torch.Tensor) -> torch.Tensor:
        """Compute the hyperbolic tangent element-wise.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with hyperbolic tangent applied element-wise.
        """
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        """Return an empty list as Tanh has no trainable parameters.

        Returns:
            List: Empty list.
        """
        return []
