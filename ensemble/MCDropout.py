import torch
from . import Ensemble


class MonteCarloDropout(Ensemble):
    def __init__(self,model, n_samples:int, reduction:str = 'mean', apply_softmax:bool = True):
        super().__init__(model, reduction=reduction, apply_softmax=apply_softmax)
        self.n_samples = n_samples
        self.__get_Dropout_modules()
        self.set_dropout()

    def __get_Dropout_modules(self):
        self.__modules = []
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                self.__modules.append(m)
        assert len(self.__modules)>0, "No Dropout modules found in model"

    def set_dropout(self):
        for m in self.__modules:
            m.train()

    def get_samples(self,x):
        '''Returns an array with n evaluations of the model with dropout enabled.'''
        ensemble = []
        for i in range(self.n_samples):
            ensemble.append(self.model(x))
        return torch.stack(ensemble)

    

    
    
