import torch
import numpy as np
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import copy

tan_sigmoid = lambda a : F.tanh(F.sigmoid(a))
criterion = nn.MSELoss()


class LM:
    def __init__(self, red, entrada, salida_esperada, lr, λ, c1 = 2, c2 = 0.1, epoch = 0):
        # print(" >> Entrada: " + str(entrada))
        self.red = red
        self.salida_esperada = salida_esperada
        self.entrada = entrada
        self.lr = lr
        self.λ = λ
        self.c1 = c1
        self.c2 = c2
        self.epoch = epoch
        self.imprimir = False
        self.result = {'grad':0,'hessian':0}

    def exec(self):
        perdidas = {}
        
        #print("paramtros de LM al iniciar1: " + str([i for i in self.red.parameters()][0]))
        # print(">>Se calcula perdida inicial...")
        f_i = self.calcula_perdida(self.aux_convierte_parametros())
        #print("paramtros de LM al iniciar2: " + str([i for i in self.red.parameters()][0]))
        #self.epoch = self.epoch+1

        # se hace un respaldo de la red en caso de que el paso del entrenamiento falle
        self.red_ant = copy.deepcopy(self.red)

        self.step()
        self.imprimir = True
        print(">>Se calcula perdida despues del paso...")
        f_i1 = self.calcula_perdida(self.aux_convierte_parametros())
        self.imprimir = False
        self.λ = 0.5*self.λ if f_i1 < f_i else 2*self.λ #se actualiza la variable lamdba segun el rendimiento de la actualizacion
        print(f"Lambda: {self.λ}")
        # print("Error Anterior: " + str(f_i.item()))
        # print("Error nuevo: " + str(f_i1.item()))
        
        #print("abs: " + str(abs(f_i1 - f_i)))
        #if abs(f_i1 - f_i) < self.c1 :
        if(f_i1.item() > f_i.item()):
            print("ERROR: la modificacion de los pesos dió un error mayor: " + str(f_i1.item()) + ", se regresa al estado anterior de la red")
            # print("paramtros RED_ANT: " + str([i for i in self.red_ant.parameters()][0]))
            # print("paramtros self.red antes: " + str([i for i in self.red.parameters()][0]))
            #red_error = copy.deepcopy(self.red)
            #self.red = red_ant
            self.rollback()
            # print("paramtros self.red despues: " + str([i for i in self.red.parameters()][0]))
            
            return self.result
        # if abs(f_i1.item()) < self.c1:
        #     #print("paramtros de LM al finalizar: " + str([i for i in self.red.parameters()][0]))
        #     # print("Se registra la perdida: " + str(self.epoch) + " " + str(f_i1))
        #     perdidas[1] = f_i1#self.epoch
        #     #writer.flush()
        #     # print("Finaliza exec...")
        #     # print("paramtros red antes de salir del ejec " + str([i for i in self.red.parameters()][0]))
        #     return self.result
            
        perdidas[1] = f_i#self.epoch
        f_i = f_i1
        return self.result
            
    def rollback(self):
        """
        Reestablece los parametros originales de la red
        """
        # Restaurar los pesos originales
        for param, original_param in zip(self.red.parameters(), self.red_ant.parameters()):
            param.data.copy_(original_param.data)
            

    def calcula_perdida(self,*parametros):
        salidas_obtenidas = torch.tensor([])
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
        for entrada,salida_esperada in zip(self.entrada,self.salida_esperada):
            l1 = tan_sigmoid(F.linear(entrada,n_params[0],n_params[1]))
            l2 = F.logsigmoid(F.linear(l1,n_params[2],n_params[3]))
            salida = F.linear(l2,n_params[4],n_params[5])
            #print("Pesos funcion: "+ str(n_params))
            #print("Pesos red: " + str([i for i in self.red.parameters()]))
            #print("----->SALIDA OBTENIDA: " + str(salida))
            if(self.imprimir):
                print(f"entrada: {entrada}")
                print("----->Salida de la red: " + str(self.red(entrada)))
                print("----->Salida esperada: " + str(salida_esperada))
            #print("Salidas: " + str(salida) + ", " + str(self.salida_esperada))
            salidas_obtenidas = torch.cat([salidas_obtenidas,salida])
        # Convertir la lista de tensores a un solo tensor
        loss = criterion(salidas_obtenidas,torch.cat(self.salida_esperada))#devuelve la perdida
        
        return loss
    
    def step(self):
        print(">>Paso...<<")

        x_n = self.aux_convierte_parametros() #concatena los paremetros de la red en un solo vector unidimensional
        h = torch.autograd.functional.hessian(self.calcula_perdida, x_n) #calculamos la matriz hessiana
        #print("tamaño de h: " + str(h.shape))
        grad_f = torch.autograd.grad(self.calcula_perdida(x_n), x_n)[0] #calculamos el gradiente de la funcion
        self.result['grad'] = grad_f
        self.result['hessian'] = h
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
        #print("--Pre-Actualización:-- " + str(x_n))
        x_n = x_n + self.lr*x_n1 #se realiza la actualización a los parametros
        #print("--Post-Actualización:-- " + str(x_n))
        #print("transpuesta: " + str(torch.transpose(x_n,0,1)[0]))
        self.asigna_parametros(torch.transpose(x_n,0,1)[0],reasignar=True)
        print(">>Fin de paso<<")
        

    def asigna_parametros(self,*parametros,reasignar=False):
        """
        Convierte los parametros dados 

        Args:
            *parametros: arreglo de parametros a asignar
            reasignar: si los parametros dados se reasignan a los pesos originales de la red
        """
        params = []
        n_params = []
        i=0
        n_dim = 0
        
        for param in list(parametros):
            params.append(param)#recibiria un solo tensor con todos los pesos
        for r_param in self.red.parameters():
            p_partida = n_dim
            n_dim = r_param.size(0)*(r_param.size(1) if r_param.dim() == 2 else 1) + p_partida # se obtiene la dimension del primer conjunto de parametros de la red
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
        Función auxiliar que convierte la entrada de los parametros de una red neuronal a un solo vector
        """
        return torch.cat([_.view(-1) for _ in self.red.parameters()], dim = 0)