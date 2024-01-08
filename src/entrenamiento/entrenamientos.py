import torch, numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from levenberg_marquardt import LM
from torch.utils.tensorboard import SummaryWriter
import io
import PIL.Image
from torchvision.transforms import ToTensor

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
            #entradas = entrada[:, :t_ent][0]
            
            #se parten los primeros 8 días y se obtiene el noveno
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
        writer.add_scalar(f'Pérdida de entrenamiento de la red: {n_red}', perdida, epoca+1)

        pinta_pesos(red,epoca)

        if (perdida.item() <= tolerancia):
            print(f"--Se supera la tolerancia. Epoca final: {epoca+1}--")
            break
    #writer.add_figure(f'Pérdida de entrenamiento de {n_red}', plt.gcf())
    print("---FIN DE ENTRENAMIENTO: entrena_LM---")


def entrena_LM_pred(red,n_red,inputs,epocas,lr,λ,batch_size = 1,t_ent = 8,t_sal = -1):
    """
    Entrena una red con el método de Levenverg-Marquardt 
    a partir de un conjunto de entradas y una salida
    Va actualizando los parametros de entrenamiento con los datos que va prediciendo
    """
    print("---INICIO DE ENTRENAMIENTO: entrena_LM_pred---")
    #print("paramtros antes: " + str([i for i in red.parameters()][0]))
    
    ventana_en_epoca = 1
    for epoca in range(epocas): #numero de apocas que se requiere que la red entrene
        s_original = [] #serie original
        s_pred = [] #serie a partir de predicciones
        ventana = 1
        print(f"---Inicio de epoca: {epoca+1}--")
        #se trata de la serie aue va prediciendo la red conforme se modifican los parametros de esta
        serie = inputs[n_red][0][:t_ent]#primeros 8 elementos de la red (inputs[numero de la red][primer batch][primeros t_ent elementos])
        for i in range(0,len(inputs[n_red]),batch_size):#por cada uno de los elementos del primer c. entrenamiento (el primero de los 6)(por cada uno son 12 iteraciones)
            
            # print("INICIO DE EPOCA...")
            # print(">>Ventana Actual: " + str(ventana))

            lote = inputs[n_red][i:i+batch_size]
            entradas_por_lote = []
            salidas_por_lote = []
            for ejemplar in lote:
                #entradas = i[:, :t_ent]#se parten los primeros 8 días y se obtiene el noveno
                entradas = serie[ventana-1:ventana+t_ent-1]
                print(f">>Entradas: {entradas}")
                #se agregan las entradas del ejemplar a las de todo el lote
                entradas_por_lote.append(entradas)
                salida = ejemplar[t_sal].view(1)
                print(">>Salida: " + str(salida))
                s_original.append(salida.item())
                salidas_por_lote.append(salida)

                pred = red(entradas) #prediccion despues de haber modificado los pesos
                serie = torch.cat((serie,pred))# Se precidce el resultado con la red despues del paso y se integra a la serie
                s_pred.append(pred.item())

                ventana = ventana + 1
            #Core/Nucleo del algoritmo
            print(f"entradas_por_lote: {entradas_por_lote}")
            print(f"salidas_por_lote: {salidas_por_lote}")
            lm = LM(red,entradas_por_lote,salidas_por_lote,lr=lr,λ = λ) #se modifica cada parametro de la red segun el batch que se le de (Entrada y salida predecida contra salida esperada)
            metricas = lm.exec(sub_epocas = 1)

            """for e in entradas_por_lote:
                pred = red(e) #prediccion despues de haber modificado los pesos
                serie = torch.cat((serie,pred))# Se precidce el resultado con la red despues del paso y se integra a la serie
                s_pred.append(pred.item())"""
            
            writer.add_scalar(f'Gradiente de {n_red}', metricas['grad'].norm(), ventana_en_epoca)
            writer.add_scalar(f'Matriz Hessiana de {n_red}', metricas['hessian'].norm(), ventana_en_epoca)
            #print(perdidas)
            #print("paramtros red despues: " + str([i for i in red.parameters()][0]))
            
            # for clave, loss in perdidas.items():
            #     perdidas_totales.append(loss)
            ventana_en_epoca = ventana_en_epoca + 1
            
        #print("paramtros despues: " + str([i for i in red.parameters()][0]))

        # for loss in perdidas_totales:
        #     writer.add_scalar('Perdida', loss, clave)
        #     clave = clave +1
        # print("s_original: " + str(s_original) + "tamaño: " + str(len(s_original)))
        # print("s_pred: " + str(s_pred) + "tamaño: " + str(len(s_pred)))
        perdida = criterion(torch.tensor(s_original),torch.tensor(s_pred))
        print(">>s_original: " + str(s_original))
        print(">>s_pred: " + str(s_pred))
        print("<<Perdida: "+str(perdida.item()) + f" epoca: {epoca+1}")
        writer.add_scalar(f'Pérdida de entrenamiento predictivo de la red: {n_red}', perdida, epoca+1)
        #writer.add_histogram(f'Serie de tiempo predicha para la epoca: {epoca+1}',torch.tensor(s_pred),epoca+1)

        
        
        pinta_pesos(red,epoca)

        plot_buf = gen_plot(s_original,s_pred,perdida.item())
        # for i, valor in enumerate(s_pred):
        #     writer.add_scalar(f'Serie de tiempo predicha para la epoca: {epoca+1}', valor, epoca+1)

        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)
        writer.add_image(f'Comportamiento de la serie de tiempo para la red: {0} durante el entrenamiento predictivo', image, epoca+1,dataformats='NCHW')

        if (perdida.item() <= tolerancia):
            print(f"---epoca final: {epoca+1}--")
            break
        #epoca = epoca + 1
    #writer.add_figure(f'Pérdida de entrenamiento de {n_red}', plt.gcf())
   
    print("---FIN DE ENTRENAMIENTO: entrena_LM_pred---")

def pinta_pesos(red, epoca):
    for nombre, parametro in red.named_parameters():
        s2 = 1 if parametro.dim() <= 1 else parametro.shape[1]
        imagen_parametro = parametro.detach().cpu().numpy().reshape((1,parametro.shape[0],s2 ,1))
        print(f"imagen_parametro: imagen_parametro.shape")
        writer.add_image(f'Pesos de la capa: {nombre} de la red {red}' , imagen_parametro, epoca+1, dataformats='NHWC')

def cerrar_escritor():
    writer.close()

def gen_plot(s_original,s_pred,perdida):
    """Create a pyplot plot and save to buffer."""
    plt.figure(figsize=(6, 4))
    plt.plot(s_original)
    plt.plot(s_pred,  label = f"Perdida: {float(perdida)}", color='#DA0C81')
    plt.title('Serie original contra Predicha')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf 