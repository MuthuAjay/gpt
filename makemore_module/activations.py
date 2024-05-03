import torch


class BatchNorm1d:
    
    def __init__(self,
                 dims : int,
                 eps: float = 1e-5,
                 momentum = 0.5) -> None:
        
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        #parameters (trained with back propagation)
        self.gamma = torch.ones(dims)
        self.beta = torch.zeros(dims)
        
        #buffers (trained with a running 'momwntum update')
        
        self.running_mean = torch.zeros(dims)
        self.running_var = torch.ones(dims)
        
    def __call__(self,
                 x: torch.Tensor) -> torch.Any:
        
        #calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True) #batch mean
            xvar = x.var(0, keepdim= True, unbiased=True) #batch variance
        
        else:
            xmean = self.running_mean
            xvar = self.running_var
            
        xhat = (x - xmean) / torch.sqrt(x.var + self.eps)  #Noemalise to unit variance , sqrt of variance is the standard deviation
        self.out = self.gamma * xhat + self.bias
        
        #update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1-self.momentum) * self.running_var + self.momentum * xvar 
                
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
    

class Tanh:
    
    def __call__(self,
                 x: torch.Tensor) -> torch.Any:
        
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
    

        
            