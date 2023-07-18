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
            ensemble.append(model(x))
        return torch.stack(ensemble)
