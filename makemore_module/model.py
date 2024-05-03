import torch

g = torch.Generator().manual_seed(42)


class Linear:
    def __init__(self, fan_in: int, fan_out: int, bias: bool = True) -> None:
        """Initialize the linear layer.

        Args:
            fan_in (int): Number of input features.
            fan_out (int): Number of output features.
            bias (bool, optional): Whether to include bias. Defaults to True.
        """
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in ** 0.5
        self.bias = torch.randn(fan_out) if bias else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the output of the linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_features).
        """
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias

        return self.out

    def parameters(self):
        """Return the parameters of the linear layer.

        Returns:
            List[torch.Tensor]: List containing weight and bias tensors (if bias is not None).
        """
        return [self.weight] + ([self.bias] if self.bias is not None else [])
