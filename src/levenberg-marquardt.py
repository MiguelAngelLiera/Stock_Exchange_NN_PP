import torch
import numpy as np
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from NARNN import NARNN

class LM:
    def __init__(self, red, lr=0.1, λ = 0.1):
        self.red = red
        self.lr = lr
        self.λ = λ

    def calcula_perdida(*parametros):
        params = []
        n_params = []
        i=0
        n_dim = 0
        for param in parametros:
            params.append(param)#recibiria un solo tensor con todos los pesos
        for r_param in red.parameters():
            p_partida = n_dim
            n_dim = r_param.size(0)*(r_param.size(1) if r_param.dim() == 2 else 1) + p_partida#se obtiene la dimension del primer conjunto de pparametros de la red
            n_params.append(params[0][p_partida:n_dim])
            n_params[i] = n_params[i].view(r_param.shape)#se le da la forma del parametro correspondiente a los parametros que llegan como entrada
            i = i+1

        #Recrea el funcionamiento de la red
        l1 = F.linear(entrada,n_params[0],n_params[1])
        l2 = F.linear(l1,n_params[2],n_params[3])
        salida = F.linear(l2,n_params[4],n_params[5])

        criterion = nn.MSELoss()
        return criterion(salida,salida_esperada)#devuelve la perdida
    
    def step():
        x_1 = torch.cat([_.view(-1) for _ in red.parameters()], dim = 0)#concatena los paremetros de la red en un solo vector unidimensional
        h = torch.autograd.functional.hessian(lm.calcula_perdida, x_1) #calculamos la matriz hessiana
        grad_f = torch.autograd.grad(lm.calcula_perdida(x_1), x_1)[0] #calculamos el gradiente de la funcion
        r = -torch.inverse(h+λ*torch.eye(211))*grad_f

red = NARNN(input_dim=8, hidden_dim=0, output_dim=1, num_layers=0)
input = torch.Tensor([1,2,3,4,5,6,7,8])

print(input)
entrada = input #la entrada se da como un parametro global
salida_esperada = torch.tensor([-0.0834]) #lo mismo para la salida esperada
λ = 0.1

lm = LM

v = torch.cat([_.view(-1) for _ in red.parameters()], dim = 0)#concatena los paremetros de la red en un solo vector unidimensional
h = torch.autograd.functional.hessian(lm.calcula_perdida, v) #calculamos la matriz hessiana
grad_f = torch.autograd.grad(lm.calcula_perdida(v), v)[0] #calculamos el gradiente de la funcion
r = -torch.inverse(h+λ*torch.eye(211))*grad_f