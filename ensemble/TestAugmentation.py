import torch
from . import Ensemble
from torchvision import transforms as torch_transforms




class Rotate:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        rot = torch.nn.functional.rotate(x, self.angle)
        return rot
class Affine:
    def __init__(self, angle,translate,scale,shear):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, x):
        new = torch.nn.functional.affine(x,self.angle,self.translate,self.scale,self.shear)
        return new

class Scale(Affine):
    """Rotate by one of the given angles."""

    def __init__(self, scale):
        super().__init__(0,[0,0],scale,0)
class Multiply:       
    def __init__(self, a:float):
        self.a = a

    def __call__(self, x):
        #torch.clamp(x*self.a,min = 0.0,max = 1.0)
        return x*self.a
class Add:       
    def __init__(self, a:float):
        self.a = a

    def __call__(self, x):
        #torch.clamp(x+self.a,min = 0.0,max = 1.0)
        return x+self.a
class FiveCrop():
    def __init__(self,size,pad = 0) -> None:
        self.pad = torch_transforms.Pad(pad)
        self.fivecrop = torch_transforms.FiveCrop(size)
    def __call__(self, x):
        x = self.pad(x)
        return self.fivecrop(x)


class TTA(Ensemble):
    # ver se essas coisas funcionam com batches
    transforms = [torch.nn.functional.hflip,
                  Scale(1.04),
                  Scale(1.1),
                  Rotate(15),
                  Rotate(-15),
                  Multiply(0.8),
                  Multiply(1.2),
                  Add(0.1),
                  Add(-0.1),
                  FiveCrop(32,4)]

    def __init__(self, model, transforms = transforms,
                 reduction:str = 'mean', apply_softmax:bool = True):
        super().__init__(model, reduction=reduction, apply_softmax= apply_softmax)

        self.transforms = transforms

    def get_samples(self,x):
        samples = []
        pred = self.model(x)
        samples.append(pred)
        for t in self.transforms:
            x = t(x)
            if isinstance(x,tuple):
                for x_ in x:
                    samples.append(self.model(x_))
            else:
                samples.append(self.model(x))
        return torch.stack(samples)