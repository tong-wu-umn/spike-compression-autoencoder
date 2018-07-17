import torch

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
__all__ = gpu