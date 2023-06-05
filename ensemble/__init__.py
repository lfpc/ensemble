import torch

class Ensemble(torch.nn.Module):
    def __init__(self,model,
                apply_softmax:bool = True,
                reduction:str = 'mean'):
        super().__init__()

        self.reduction = reduction
        self.model = model
        self.apply_softmax = apply_softmax
    
    def get_samples(self,x:torch.tensor):
        '''Default ensemble model is to assume that self.model returns samples'''
        ensemble = self.model(x)
        return ensemble

    def ensemble_forward(self,x:torch.tensor):
        ensemble = self.get_samples(x)
        if self.apply_softmax:
            ensemble = ensemble.softmax(-1)
        return ensemble.mean(0)
        
    def forward(self,x):
        if self.reduction == 'mean':
            return self.ensemble_forward(x)
        elif self.reduction == 'none':
            return self.get_samples(x)