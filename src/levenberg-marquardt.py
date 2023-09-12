import torch
import numpy as np
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from NARNN import NARNN

class LM(self):
    def __init__(red, lr=0.1, λ = 0.1):
        self.red = red
        self.lr = lr
        self.λ = λ

red = NARNN(input_dim=8, hidden_dim=0, output_dim=1, num_layers=0)
input = torch.Tensor([1,2,3,4,5,6,7,8])

print(input)
entrada = input #la entrada se da como un parametro global
salida_esperada = torch.tensor([-0.0834]) #lo mismo para la salida esperada
λ = 0.1