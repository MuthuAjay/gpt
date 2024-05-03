import torch 

g = torch.Generator().manual_seed(42)

class Linear:
    
    def __init__(self, 
                 fan_in : int,
                 fan_out: int ,
                 bias : bool = True) -> None:
        
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in ** 0.5
        self.bais = torch.randn(fan_out) if bias else None
        
    def __call__(self, 
                 x : torch.Tensor) -> torch.Any:
        self.out = x @ self.weight
        if self.bais is not None:
            self.out += self.bias
        
        return self.out
    
    def parameters(self):
        
        return [self.weight] + ([self.bias] if self.bias is not None else [])
    

    
