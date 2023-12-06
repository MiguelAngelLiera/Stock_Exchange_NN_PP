import torch, numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from levenberg_marquardt import LM
from torch.utils.tensorboard import SummaryWriter

def error(modelo,input_data,target):
    return modelo(input_data)-target

criterion = nn.MSELoss()
writer = SummaryWriter('logs')
tolerancia = 0.001

def train(red,input_data, target, modelo):
    # Definir la función de pérdida y el optimizador
    
    optimizer = optim.LBFGS(red.parameters(), lr=0.1)
    def closure():
        optimizer.zero_grad()
        output = error(modelo,input_data,target)
        ##loss = criterion(output,target)
        loss = criterion(output, target)
        #loss = torch.sum(output ** 2)##criterion(output, target)
        loss.backward()
        return loss

    optimizer.step(closure)

def train_SGD(red,input_data, target):
    optimizer = optim.SGD(red.parameters(), lr=0.1, momentum=0.4)#,maximize=True)
    optimizer.zero_grad()
    output = red(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

def train_ASGD(red,input_data, target, modelo):
    optimizer = optim.ASGD(red.parameters(), lr=0.1)
    optimizer.zero_grad()
    output = modelo(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

#se trata de los conjuntos de todas las entradas y salidas para todas las redes
entradas_por_red = []
salidas_por_red = []
# Entrenar la red neuronal
def entrena(red,n_red,inputs,epocas=1000,t_ent = 8,t_sal = -1):
    """
    Entrena una red a partir de un conjunto de entradas y una salida
    """
    for i in range(epocas): #1000 epocas
        for i in inputs[n_red]:#por cada uno de los elementos del primer c. entrenamiento (el primero de los 6)(son 12 iteraciones)
            entradas = i[:, :t_ent]#se parten los primeros 8 días y se obtiene el noveno
            salida = i[:, t_sal]
            #for _ in range(100):# se entrena con esas entradas y esa salida
            # output = red(entradas)
            # loss = criterion(output, salida)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            train_SGD(red,entradas,salida)

def entrena_LM(red,n_red,inputs,epocas,lr,λ,t_ent = 8,t_sal = -1):
    """
    Entrena una red con el método de Levenverg-Marquardt 
    a partir de un conjunto de entradas y una salida
    """
    print("---INICIO DE ENTRENAMIENTO: entrena_LM_pred---")
    # print("paramtros antes: " + str([i for i in red.parameters()][0]))
    perdidas_totales = []
    s_original = []
    s_pred = []
    ventana_en_epoca = 1
    #epoca = 1
    for epoca in range(epocas): #1000 epocas
        print(f"---Inicio de epoca: {epoca + 1}--")
        ventana = 1
        
        for entrada in inputs[n_red]:#por cada uno de los elementos del primer c. entrenamiento (el primero de los 6)(son 12 iteraciones)
            # print(">>Ventana Actual: " + str(ventana))
            #entradas = entrada[:, :t_ent][0]#se parten los primeros 8 días y se obtiene el noveno
            entradas = entrada[:t_ent]
            #salida = entrada[:, t_sal]
            salida = entrada[t_sal].view(1)
            #print("Salida: " + str(salida))
            s_original.append(salida.item())

            lm = LM(red,entradas,salida,lr=lr,λ = λ)
            metricas = lm.exec()
            # print(perdidas)
            pred = red(entradas)
            s_pred.append(pred.item())
            #print(">>>metricas: " + str(metricas['grad'].norm()) + " ," + str(metricas['hessian'].norm()))
            writer.add_scalar(f'Gradiente de {n_red}', metricas['grad'].norm(), ventana_en_epoca)
            writer.add_scalar(f'Matriz Hessiana de {n_red}', metricas['hessian'].norm(), ventana_en_epoca)
            # for clave, loss in perdidas.items():
            #     perdidas_totales.append(loss)
            ventana_en_epoca = ventana_en_epoca + 1
            ventana = ventana + 1
        #print("paramtros final iteración: " + str([i for i in red.parameters()][0]))

        clave = 1
        # for loss in perdidas_totales:
        #     writer.add_scalar('Perdida', loss, clave)
        #     clave = clave +1
        #epoca = epoca + 1
        # print("s_original: " + str(s_original) + "tamaño: " + str(len(s_original)))
        # print("s_pred: " + str(s_pred) + "tamaño: " + str(len(s_pred)))
        perdida = criterion(torch.tensor(s_original),torch.tensor(s_pred))
        #print("<<Perdida: "+str(perdida.item()) + f" epoca: {epoca+1}")
        writer.add_scalar(f'Pérdida de entrenamiento de {n_red}', perdida, epoca+1)
        if (perdida.item() <= tolerancia):
            print(f"---epoca final: {epoca+1}--")
            break
    #writer.add_figure(f'Pérdida de entrenamiento de {n_red}', plt.gcf())
    writer.close()
    print("---FIN DE ENTRENAMIENTO: entrena_LM_pred---")


def entrena_LM_pred(red,n_red,inputs,epocas,lr,λ,t_ent = 8,t_sal = -1):
    """
    Entrena una red con el método de Levenverg-Marquardt 
    a partir de un conjunto de entradas y una salida
    Va actualizando los parametros de entrenamiento con los datos que va prediciendo
    """
    print("---INICIO DE ENTRENAMIENTO: entrena_LM_pred---")
    #print("paramtros antes: " + str([i for i in red.parameters()][0]))
    perdidas_totales = []
    
    ventana_en_epoca = 1
    #epoca = 1
    for epoca in range(epocas): #1000 epocas
        s_original = []
        s_pred = []
        ventana = 1
        print(f"---Inicio de epoca: {epoca+1}--")
        # print(inputs[n_red][0])
        serie = inputs[n_red][0][:t_ent]#primeros 8 elementos de la red
        for i in inputs[n_red]:#por cada uno de los elementos del primer c. entrenamiento (el primero de los 6)(son 12 iteraciones)
            
            # print("INICIO DE EPOCA...")
            # print(">>Ventana Actual: " + str(ventana))
            
            #entradas = i[:, :t_ent]#se parten los primeros 8 días y se obtiene el noveno
            entradas = serie[ventana-1:ventana+t_ent-1]
            print(f">>Entradas: {entradas}")
            salida = i[t_sal].view(1)
            print(">>Salida: " + str(salida))
            s_original.append(salida.item())
            #Core del algoritmo
            lm = LM(red,entradas,salida,lr=lr,λ = λ)
            metricas = lm.exec(sub_epocas = 1)

            pred = red(entradas)
            serie = torch.cat((serie,pred))# Se precidce el resultado con la red despues del paso y se integra a la serie
            s_pred.append(pred.item())
            writer.add_scalar(f'Gradiente de {n_red}', metricas['grad'].norm(), ventana_en_epoca)
            writer.add_scalar(f'Matriz Hessiana de {n_red}', metricas['hessian'].norm(), ventana_en_epoca)
            #print(perdidas)
            #print("paramtros red despues: " + str([i for i in red.parameters()][0]))
            
            # for clave, loss in perdidas.items():
            #     perdidas_totales.append(loss)
            ventana_en_epoca = ventana_en_epoca + 1
            ventana = ventana + 1
        #print("paramtros despues: " + str([i for i in red.parameters()][0]))

        # for clave, loss in perdidas_totales.items():
        #     print(f"Clave: {clave}, Valor: {loss}")
        clave = 1
        # for loss in perdidas_totales:
        #     writer.add_scalar('Perdida', loss, clave)
        #     clave = clave +1
        # print("s_original: " + str(s_original) + "tamaño: " + str(len(s_original)))
        # print("s_pred: " + str(s_pred) + "tamaño: " + str(len(s_pred)))
        perdida = criterion(torch.tensor(s_original),torch.tensor(s_pred))
        print(">>s_original: " + str(s_original))
        print(">>s_pred: " + str(s_pred))
        print("<<Perdida: "+str(perdida.item()) + f" epoca: {epoca+1}")
        writer.add_scalar(f'Pérdida de entrenamiento de {n_red}', perdida, epoca+1)
        # if (perdida.item() <= tolerancia):
        #     print(f"---epoca final: {epoca+1}--")
        #     break
        #epoca = epoca + 1
    #writer.add_figure(f'Pérdida de entrenamiento de {n_red}', plt.gcf())
    writer.close()
    print("---FIN DE ENTRENAMIENTO: entrena_LM_pred---")