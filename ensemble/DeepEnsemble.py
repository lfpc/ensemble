import torch
from . import Ensemble
class DeepEnsemble(Ensemble):
    def __init__(self,models,
                apply_softmax:bool = True,
                reduction:str = 'mean'):
        super().__init__(torch.nn.ParameterList(models), apply_softmax, reduction)
    
    def get_samples(self,x):
        ensemble = []
        for model in self.model:
            if self.apply_softmax:
                ensemble.append(model(x).softmax(-1))
            else:
                ensemble.append(model(x))
        return torch.stack(ensemble)
