import torch 


class PassThroughCore(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        pass 

    def forward(self, x):
        return x