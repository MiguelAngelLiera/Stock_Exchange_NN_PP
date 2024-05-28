import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

tan_sigmoid = lambda a : F.tanh(F.sigmoid(a))
criterion = nn.MSELoss()

class LM:
    def __init__(self, red, entrada, salida_esperada, lr, λ, c1 = 2, c2 = 0.1, epoch = 0):
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
        """
        Execución de un solo paso del algoritmo Levenberg-Marquardt
        """
        # Se calcula la perdida de la predicción contra la salida esperada
        f_i = self.calcula_perdida(self.aux_convierte_parametros())

        # se hace un respaldo de la red en caso de que el paso del entrenamiento falle
        self.red_ant = copy.deepcopy(self.red)

        # Se actualizan los pesos de la red a partir de una sola iteración del algoritmo LM
        x_n = self.step()
        
        self.imprimir = True
        #print(">>Se calcula perdida despues del paso...")
        f_i1 = self.calcula_perdida(self.aux_convierte_parametros(n_params=x_n))
        self.imprimir = False
        #se actualiza la variable lamdba segun el rendimiento de la actualizacion
        if f_i1 < f_i:
            self.λ = 0.5*self.λ
            self.aux_reasigna_parametros(x_n)
        else:
            self.λ = 2*self.λ
        print(f"Lambda: {self.λ}")
        # print("Función de Error Anterior: " + str(f_i.item()))
        # print("Función de Error nuevo: " + str(f_i1.item()))
        
        
        if(f_i1.item() > f_i.item()):
            print("ERROR: la modificacion de los pesos dió un error mayor: " + str(f_i1.item()) + ", se regresa al estado anterior de la red")
            # print("paramtros RED_ANT: " + str([i for i in self.red_ant.parameters()][0]))
            # print("paramtros self.red antes: " + str([i for i in self.red.parameters()][0]))
            self.rollback()
            # print("paramtros self.red despues: " + str([i for i in self.red.parameters()][0]))
            
            return self.result
        
        # print("abs: " + str(abs(f_i1 - f_i)))
        # en el caso de que supere la tolerancia para la convergencia del método
        # no hacemos uso de esta pues solamente e trata de una sola iteracion del algoritmo
        # if abs(f_i1.item() - f_i.item()) > self.c1 :
        #     print("Se ha superado la tolerancia para la convergencia del método")
        #     return self.result
        # la evaluacion de tolerancia la podremos encontrar en el método NARNN.entrenamiento.entrena_lm 
            
        f_i = f_i1
        return self.result
            
    def rollback(self):
        """
        Reestablece los parametros originales de la red
        """
        for param, original_param in zip(self.red.parameters(), self.red_ant.parameters()):
            param.data.copy_(original_param.data)
            

    def calcula_perdida(self,*parametros):
        """
        Calcula la perdida entre la predicción de la red contra la salida esperada,
        teniendo como argumentos los parametros de la red. Así esta función es sobre la cual se
        obtiene el gradiente y la matriz hessiana.

        Args:
            *parametros: de la red (sus pesos y sesgos)
        """
        # print(">>Se calcula perdida...")
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

        # NOTA: Debido a que el calculo de la matriz hessiana sobre esta función no puede realizarse directamente
        # usando los parametros de la red (devuelve un tensor lleno de 0's), se recrea aqui mismo el funcionamiento 
        # de la red
        # url: https://stackoverflow.com/questions/64024312/how-to-compute-hessian-matrix-for-all-parameters-in-a-network-in-pytorch

        for entrada,salida_esperada in zip(self.entrada,self.salida_esperada):
            # Recrea el funcionamiento del paso de propagación hacia adelante 
            # de la red con la arquitectura que se muestra en NARNN.py
            l1 = tan_sigmoid(F.linear(entrada,n_params[0],n_params[1]))
            l2 = F.logsigmoid(F.linear(l1,n_params[2],n_params[3]))
            salida = F.linear(l2,n_params[4],n_params[5])
            #print("Pesos funcion: "+ str(n_params))
            #print("Pesos red: " + str([i for i in self.red.parameters()]))
            if(self.imprimir):
                print(f"entrada: {entrada}")
                print("--->Predicción: " + str(salida))
                print("--->Salida esperada: " + str(salida_esperada))
            
            # Convertir la lista de tensores a un solo tensor
            salidas_obtenidas = torch.cat([salidas_obtenidas,salida])
        loss = criterion(salidas_obtenidas,torch.tensor(self.salida_esperada))#devuelve la perdida
        
        return loss
    
    def step(self):
        """
        Paso de la optimización del algoritmo LM
        """
        x_n = self.aux_convierte_parametros() 

        #calculamos la matriz hessiana
        h = torch.autograd.functional.hessian(self.calcula_perdida, x_n) 

        #calculamos el gradiente de la funcion
        grad_f = torch.autograd.grad(self.calcula_perdida(x_n), x_n)[0]

        # calculamos la transpuesta del gradiente
        grad_f = torch.transpose(torch.unsqueeze(grad_f, 0),0, 1) 

        # multiplica un escalar por la matriz identidad del tamaño de h y se lo sumamos a h
        h_p = h+self.λ*torch.eye(h.size(1))
        #producto punto entre - la inversa de h_p y el gradiente de la red
        x_n1 = torch.matmul(-torch.inverse(h_p),grad_f) 
        #se le da la forma adecuada para que se pueda sumar con el vector de nuevos pesos
        x_n = x_n.reshape(h.size(1), 1)

        #se realiza la actualización a los parametros
        x_n = x_n + self.lr*x_n1 
        return self.asigna_parametros(torch.transpose(x_n,0,1)[0],reasignar=False)
        

    def asigna_parametros(self,*parametros,reasignar=False):
        """
        Convierte los parametros dados a una forma aceptable para self.red.parameters()

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
            self.aux_reasigna_parametros(n_params)
        return n_params
    
    def aux_reasigna_parametros(self,n_params):
        """
        Función auxiliar que asigna los argumentos dados a los parametros de la red

        Args:
            n_params: parametros nuevos a asignar.
        """
        i = 0
        for r_param in self.red.parameters():
            n_params[i].view(r_param.shape)#se le da la forma del parametro correspondiente a los parametros que llegan como entrada
            r_param.data = n_params[i]
            #r_param.data = torch.ones_like(n_params[i])
            #print("parametro: " + str(r_param))
            i = i+1
    
    def aux_convierte_parametros(self, n_params=[]):
        """
        Función auxiliar que convierte la entrada de los parametros de una red neuronal a un solo vector
        """
        if (len(n_params) == 0):
            return torch.cat([_.view(-1) for _ in self.red.parameters()], dim = 0)
        else:
            return torch.cat([_.view(-1) for _ in n_params], dim = 0)
        

# def step(self):
#         """
#         Paso de la optimización
#         """
#         #print(">>Paso...<<")
#         #concatena los paremetros de la red en un solo vector unidimensional
#         x_n = self.aux_convierte_parametros() 

#         #calculamos la matriz hessiana
#         h = torch.autograd.functional.hessian(self.calcula_perdida, x_n) 
#         #print("tamaño de h: " + str(h.shape))
#         #calculamos el gradiente de la funcion
#         grad_f = torch.autograd.grad(self.calcula_perdida(x_n), x_n)[0] 
#         self.result['grad'] = grad_f
#         self.result['hessian'] = h

#         # calculamos la transpuesta del gradiente
#         grad_f = torch.transpose(torch.unsqueeze(grad_f, 0),0, 1) 
#         #print("tamaño de grad " + str(grad_f.shape))

#         # multiplica un escalar por la matriz identidad del tamaño de h y se lo sumamos a h
#         h_p = h+self.λ*torch.eye(h.size(1))
#         #producto punto entre - la inversa de h_p y el gradiente de la red
#         x_n1 = torch.matmul(-torch.inverse(h_p),grad_f) 
#         #se le da la forma adecuada para que se pueda sumar con el vector de nuevos pesos
#         x_n = x_n.reshape(h.size(1), 1)

#         #print("x_n antes1: " + str(x_n))
#         #print("x_n1 antes: " + str(x_n1))
#         #print("lr*x_n1 antes: " + str(self.lr*x_n1))
#         #print("parametros de la red: " + str([i for i in self.red.parameters()]))
#         #print("--Pre-Actualización:-- " + str(x_n))
#         x_n = x_n + self.lr*x_n1 #se realiza la actualización a los parametros
#         #print("--Post-Actualización:-- " + str(x_n))
#         #print("transpuesta: " + str(torch.transpose(x_n,0,1)[0]))
#         return self.asigna_parametros(torch.transpose(x_n,0,1)[0],reasignar=False)
#         #print(">>Fin de paso<<")