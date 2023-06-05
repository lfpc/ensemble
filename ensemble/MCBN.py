import torch
from . import Ensemble
from copy import copy,deepcopy
from collections import defaultdict

class MonteCarloBatchNormalization(Ensemble):
    def __init__(self,model, n_samples:int, batch_loader,
                      reduction:str = 'mean', apply_softmax:bool = True):
        super().__init__(model, apply_softmax=apply_softmax, reduction= reduction)
        self.batch_loader = batch_loader
        self.n_samples = n_samples

        self.__get_BN_modules()
        self.__save_main_attributes()

    def __get_BN_modules(self):
        self._BN_modules = []
        for m in self.model.modules():
            if m.__class__.__name__.startswith('BatchNorm'):
                self._BN_modules.append(m)
        assert len(self._BN_modules)>0, "No BatchNormalization modules found in model"

    def __save_main_attributes(self):
        self.__momentum = {}
        self.__running_mean= {}
        self.__running_var= {}

        for m in self._BN_modules:
            self.__momentum[m] = deepcopy(m.momentum)
            self.__running_mean[m] = deepcopy(m.running_mean)
            self.__running_var[m] = deepcopy(m.running_var)
    def __set_main_attributes(self):
        for m in self._BN_modules:
            m.momentum = deepcopy(self.__momentum[m])
            m.running_mean = deepcopy(self.__running_mean[m])
            m.running_var = deepcopy(self.__running_var[m])

    def set_BN_mode(self):
        for m in self._BN_modules:
            m.train()
            m.track_running_stats = True
            m.momentum = 1.0
            
    def reset_normal_mode(self):
        self.eval()
        self.__set_main_attributes()

    def get_samples(self,x):
        ensemble = []
        batch_loader = iter(self.batch_loader)
        for _ in range(self.n_samples):
            im_train,_ = next(batch_loader)
            im_train = im_train.to(x.device)
            self.set_BN_mode()
            with torch.no_grad():
                self.model(im_train)
                self.model.eval()
                y = self.model(x)
                ensemble.append(y)
        ensemble = torch.stack(ensemble)
        return ensemble

    def deterministic(self,x):
        self.reset_normal_mode()
        return super().deterministic(x)

class Fast_MCBN(MonteCarloBatchNormalization):
    def __init__(self, model, n_samples, batch_loader,reduction:str = 'mean', apply_softmax:bool = True):
        super().__init__(model, n_samples, None, reduction=reduction, apply_softmax=apply_softmax)
        self.__get_BN_parameters(batch_loader)
        self.reset_normal_mode()
        self.eval()

    def __get_BN_parameters(self,batch_loader):
        self.params = defaultdict(list)
        batch_loader = iter(batch_loader)
        self.set_BN_mode()
        device = next(self.model.parameters()).device
        for _ in range(self.n_samples):
            im,_ = next(batch_loader)
            im = im.to(device)
            with torch.no_grad():
                self.model(im)
                for m in self._BN_modules:
                    self.params[m].append((deepcopy(m.running_mean),deepcopy(m.running_var)))

    def __set_BN_parameters(self,i):
        for m in self._BN_modules:
            mean,var = self.params[m][i]
            m.running_mean = mean
            m.running_var = var

    def get_samples(self,x):
        ensemble = []
        for i in range(self.n_samples):
            self.__set_BN_parameters(i)
            with torch.no_grad():
                ensemble.append(self.model(x))
        return torch.stack(ensemble)