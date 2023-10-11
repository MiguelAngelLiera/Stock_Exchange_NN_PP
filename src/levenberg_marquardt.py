import torch
import numpy as np
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from NARNN import NARNN
import copy

tan_sigmoid = lambda a : F.tanh(F.sigmoid(a))
criterion = nn.MSELoss()
#entrada = torch.Tensor([1,2,3,4,5,6,7,8])   
#salida_esperada = torch.tensor([-0.0834])

class LM:
    def __init__(self, red, entrada, salida_esperada, lr=0.05, λ = 0.1, c1 = 2, c2 = 0.1):
        print(" >> Entrada: " + str(entrada))
        self.red = red
        self.salida_esperada = salida_esperada
        self.entrada = entrada
        self.lr = lr
        self.λ = λ
        self.c1 = c1
        self.c2 = c2
        self.epoch = 0

    def exec(self,epocas = 40):
        self.epoch = 0
        perdidas = {}
        
        #print("paramtros de LM al iniciar1: " + str([i for i in self.red.parameters()][0]))
        print(">>Se calcula perdida inicial...")
        f_i = self.calcula_perdida(self.aux_convierte_parametros())
        #print("paramtros de LM al iniciar2: " + str([i for i in self.red.parameters()][0]))
        for i in range(epocas):
            self.epoch = self.epoch+1
            print("epoca: " + str(self.epoch))
            self.red_ant = copy.deepcopy(self.red)
            self.step()
            print(">>Se calcula perdida despues del paso...")
            f_i1 = self.calcula_perdida(self.aux_convierte_parametros())
            self.λ = 0.5*self.λ if f_i1 < f_i else 2*self.λ #se actualiza la variable lamba segun el rendimiento de la actualizacion
            print("Error Anterior: " + str(f_i.item()))
            print("Error nuevo: " + str(f_i1.item()))
            
            #print("abs: " + str(abs(f_i1 - f_i)))
            #if abs(f_i1 - f_i) < self.c1 :
            if(f_i1.item() > f_i.item()):
                print("ERROR: la modificacion de los pesos dió un error mayor: " + str(f_i1.item()) + ", se regresa al estado anterior de la red")
                print("paramtros RED_ANT: " + str([i for i in self.red_ant.parameters()][0]))
                print("paramtros self.red antes: " + str([i for i in self.red.parameters()][0]))
                #red_error = copy.deepcopy(self.red)
                #self.red = red_ant
                self.rollback()
                print("paramtros self.red despues: " + str([i for i in self.red.parameters()][0]))
                
                return perdidas
            if abs(f_i1.item()) < self.c1:
                #print("paramtros de LM al finalizar: " + str([i for i in self.red.parameters()][0]))
                print("Se registra la perdida: " + str(self.epoch) + " " + str(f_i1))
                perdidas[self.epoch] = f_i1
                #writer.flush()
                print("Finaliza exec...")
                print("paramtros red antes de salir del ejec " + str([i for i in self.red.parameters()][0]))
                return perdidas
                
            perdidas[self.epoch] = f_i
            f_i = f_i1
        return perdidas
            
    def rollback(self):
        """
        Reestablece los parametros originales de la red
        """
        # Restaurar los pesos originales
        for param, original_param in zip(self.red.parameters(), self.red_ant.parameters()):
            param.data.copy_(original_param.data)
            

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
        
        l1 = tan_sigmoid(F.linear(self.entrada,n_params[0],n_params[1]))
        l2 = F.logsigmoid(F.linear(l1,n_params[2],n_params[3]))
        salida = F.linear(l2,n_params[4],n_params[5])
        #print("Pesos funcion: "+ str(n_params))
        #print("Pesos red: " + str([i for i in self.red.parameters()]))
        #print("----->SALIDA OBTENIDA: " + str(salida))
        #print("----->SALIDA DE LA RED OBTENIDA: " + str(self.red(self.entrada)))
        #print("----->SALIDA ESPERADA: " + str(self.salida_esperada))
        #print("Salidas: " + str(salida[0]) + ", " + str(self.salida_esperada))
        loss = criterion(salida[0],self.salida_esperada)#devuelve la perdida
        
        return loss
    
    def step(self):
        print(">>Inicio de paso (Los valores de la perdida aqui contenidos solo son usados para calculos)")
        x_n = self.aux_convierte_parametros() #concatena los paremetros de la red en un solo vector unidimensional
        h = torch.autograd.functional.hessian(self.calcula_perdida, x_n) #calculamos la matriz hessiana
        #print("tamaño de h: " + str(h.shape))
        grad_f = torch.autograd.grad(self.calcula_perdida(x_n), x_n)[0] #calculamos el gradiente de la funcion
        grad_f = torch.transpose(torch.unsqueeze(grad_f, 0),0, 1) # calculamos la transpuesta del gradiente
        #print("tamaño de grad " + str(grad_f.shape))
        #λ*torch.eye(211): multiplica un escalar por la matriz identidad
        #print("h size(): " +  str(h.size(1)))
        h_p = h+self.λ*torch.eye(h.size(1))
        #print("Determinante: " + str(int(torch.linalg.det(h_p).item()) != 0))
        #if(int(torch.linalg.det(h_p)) != 0.0):
         #   print("Determinante: " + str(torch.linalg.det(h_p).item()))
        x_n1 = torch.matmul(-torch.inverse(h_p),grad_f) 
        #print("x_n antes: " + str(x_n))
        x_n = x_n.reshape(h.size(1), 1)#se le da la forma adecuada para que se pueda sumar con el vector de nuevos pesos

        #print("x_n antes1: " + str(x_n))
        #print("x_n1 antes: " + str(x_n1))
        #print("lr*x_n1 antes: " + str(self.lr*x_n1))
        #print("parametros de la red: " + str([i for i in self.red.parameters()]))
        print("--Pre-Actualización:-- " + str(x_n))
        x_n = x_n + self.lr*x_n1
        print("--Post-Actualización:-- " + str(x_n))
        #print("transpuesta: " + str(torch.transpose(x_n,0,1)[0]))
        self.asigna_parametros(torch.transpose(x_n,0,1)[0],reasignar=True)
        print(">>Fin de paso")
        

    def asigna_parametros(self,*parametros,reasignar=False):
        #print("asigna parametros")
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
        #print("aux_convierte_parametros")
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