import torch
import numpy as np
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from NARNN import NARNN

#entrada = torch.Tensor([1,2,3,4,5,6,7,8])   
#salida_esperada = torch.tensor([-0.0834])

class LM:
    def __init__(self, red, entrada, salida_esperada, lr=0.1, λ = 0.1, c1 = 0.0001, c2 = 0.1):
        self.red = red
        self.salida_esperada = salida_esperada
        self.entrada = entrada
        self.lr = lr
        self.λ = λ
        self.c1 = c1
        self.c2 = c2

    def exec(self,epocas = 100):
        f_i = self.calcula_perdida(self.aux_convierte_parametros())
        for i in range(epocas):
            #print("epoca: " + str(i))
            self.step()
            f_i1 = self.calcula_perdida(self.aux_convierte_parametros())
            self.λ = 0.5*self.λ if f_i1 < f_i else 2*self.λ #se actualiza la variable lamba segun el rendimiento de la actualizacion
            #print("nuevo: " + str(f_i1))
            #print("ant: " + str(f_i))
            #print("abs: " + str(abs(f_i1 - f_i)))
            #if abs(f_i1 - f_i) < self.c1:
            if abs(f_i1) < self.c1:
                break
            f_i = f_i1

    def calcula_perdida(self,*parametros):
        # params = []
        # n_params = []
        # i=0
        # n_dim = 0
        # for param in parametros:
        #     params.append(param)#recibiria un solo tensor con todos los pesos
        # for r_param in red.parameters():
        #     p_partida = n_dim
        #     n_dim = r_param.size(0)*(r_param.size(1) if r_param.dim() == 2 else 1) + p_partida #se obtiene la dimension del primer conjunto de parametros de la red
        #     n_params.append(params[0][p_partida:n_dim])
        #     n_params[i] = n_params[i].view(r_param.shape)#se le da la forma del parametro correspondiente a los parametros que llegan como entrada
        #     i = i+1
        
        parametros = list(parametros)[0]
        #print("parametros1: " + str(parametros))
        n_params = self.asigna_parametros(parametros)

        #Recrea el funcionamiento de la red

        #print("parametrosi: " + str(entrada))
        l1 = F.linear(self.entrada,n_params[0],n_params[1])
        l2 = F.linear(l1,n_params[2],n_params[3])
        salida = F.linear(l2,n_params[4],n_params[5])
        #print("----->SALIDA OBTENIDA: " + str(salida))

        criterion = nn.MSELoss()
        return criterion(salida,self.salida_esperada)#devuelve la perdida
    
    def step(self):
        
        x_n = self.aux_convierte_parametros() #concatena los paremetros de la red en un solo vector unidimensional
        h = torch.autograd.functional.hessian(self.calcula_perdida, x_n) #calculamos la matriz hessiana
        #print("tamaño de h: " + str(h.shape))
        grad_f = torch.autograd.grad(self.calcula_perdida(x_n), x_n)[0] #calculamos el gradiente de la funcion
        grad_f = torch.transpose(torch.unsqueeze(grad_f, 0),0, 1) # calculamos la transpuesta del gradiente
        #print("tamaño de grad " + str(grad_f.shape))
        #λ*torch.eye(211): multiplica un escalar por la matriz identidad
        #print("h size(): " +  str(h.size(1)))
        x_n1 = torch.matmul(-torch.inverse(h+self.λ*torch.eye(h.size(1))),grad_f) 
        #print("x_n antes: " + str(x_n))
        x_n = x_n.reshape(211, 1)#se le da la forma adecuada para que se pueda sumar con el vector de nuevos pesos

        #print("x_n antes1: " + str(x_n))
        #print("x_n1 antes: " + str(x_n1.shape))
        #print("parametros de la red: " + str([i for i in self.red.parameters()]))
        x_n = x_n + x_n1
        #print("transpuesta: " + str(torch.transpose(x_n,0,1)[0]))
        self.asigna_parametros(torch.transpose(x_n,0,1)[0],reasignar=True)
        #print("Da un paso")
        

    def asigna_parametros(self,*parametros,reasignar=False):
        print("asigna parametros")
        params = []
        n_params = []
        i=0
        n_dim = 0
        
        for param in list(parametros):
            
            params.append(param)#recibiria un solo tensor con todos los pesos
        for r_param in self.red.parameters():
            p_partida = n_dim
            n_dim = r_param.size(0)*(r_param.size(1) if r_param.dim() == 2 else 1) + p_partida #se obtiene la dimension del primer conjunto de parametros de la red
            #print("p_partida: " + str(p_partida) + "n_dim: "+ str(n_dim))
            n_params.append(params[0][p_partida:n_dim])
            #print("j: " + str(params[0][p_partida:n_dim]))
            n_params[i] = n_params[i].view(r_param.shape)#se le da la forma del parametro correspondiente a los parametros que llegan como entrada
            
            i = i+1
        if (reasignar):
            i = 0
            for r_param in self.red.parameters():
                n_params[i].view(r_param.shape)#se le da la forma del parametro correspondiente a los parametros que llegan como entrada
                r_param.data = n_params[i]
                #r_param.data = torch.ones_like(n_params[i])
                #print("parametro: " + str(r_param))
                i = i+1
        return n_params
    
    def aux_convierte_parametros(self):
        """
        Funcion auxiliar que convierte la entrada de los parametros de una red neuronal a un solo vector
        """
        print("aux_convierte_parametros")
        #print(torch.cat([_.view(-1) for _ in self.red.parameters()], dim = 0))
        return torch.cat([_.view(-1) for _ in self.red.parameters()], dim = 0)

#red = NARNN(input_dim=8, hidden_dim=0, output_dim=1, num_layers=0)
#input = torch.Tensor([1,2,3,4,5,6,7,8])

# print(input)
#entrada = input #la entrada se da como un parametro global
#salida_esperada = torch.tensor([-0.0834]) #lo mismo para la salida esperada
# λ = 0.1

#lm = LM(red)

# v = torch.cat([_.view(-1) for _ in red.parameters()], dim = 0)#concatena los paremetros de la red en un solo vector unidimensional
# h = torch.autograd.functional.hessian(lm.calcula_perdida, v) #calculamos la matriz hessiana
# grad_f = torch.autograd.grad(lm.calcula_perdida(v), v)[0] #calculamos el gradiente de la funcion
# r = -torch.inverse(h+λ*torch.eye(211))*torch.transpose(grad_f)