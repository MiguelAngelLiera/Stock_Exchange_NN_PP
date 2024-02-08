import torch, numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from .levenberg_marquardt import LM
from torch.utils.tensorboard import SummaryWriter
import io
import PIL.Image
from ....utilerias import utilerias as utls
from torchvision.transforms import ToTensor

def error(modelo,input_data,target):
    return modelo(input_data)-target

writer = SummaryWriter('logs/auto_regresivo')

class Entrenamiento:

    def __init__(self,red,n_red=0) -> None:
        """
        Constructor de la clase
        Args:
            red: instancia de la red a entrenar
        """
        self.red = red
        self.n_red = n_red
        self.criterion = nn.MSELoss()
        self.writer = writer
        self.tolerancia = 0.001

    def entrena_lm(self,inputs,epocas,lr,λ,batch_size = 1,decay_factor=0,t_ent = 8,t_sal = -1,e_predictivo = False):
        """
        Entrena una red con el método de Levenverg-Marquardt 
        a partir de un conjunto de entradas y una salida
        Va actualizando los parametros de entrenamiento con los datos que va prediciendo

            n_red:
            inputs: entradas de la red
            epocas: número de iteraciones del algoritmo
            lr: taza de aprendizaje que se aplicara como ponderación de la actualización de los parametros de la red
            λ: ponderación inicial de la matriz hessiana 
            batch_size: tamaño del batch para el entrenamiento
            decay_factor: ponderación del decaimiento de la taza de aprendizaje
            t_ent: tamaño de la entrada a la red neuronal
            t_sal: tamaño de la salida de la red neuronal
        """
        print("---INICIO DE ENTRENAMIENTO: entrena_LM---")

        lr_callback = CustomLearningRateScheduler(lr,decay_factor)
        lote_designado = 1 # lote elegido para empezar el decaimiento de lr
        for epoca in range(0,epocas+1): # número de apocas que se requiere que la red entrene
            print(f"---Inicio de epoca: {epoca}--")
            
            s_original, s_pred = self.entrena_por_lote(inputs,t_ent,t_sal,e_predictivo,batch_size,lote_designado,epoca,lr,λ,lr_callback)

            # print("s_original: " + str(s_original) + "tamaño: " + str(len(s_original)))
            # print("s_pred: " + str(s_pred) + "tamaño: " + str(len(s_pred)))
            # Se comparan las perdidas entre la señal original y la que predijo el algoritmo
            perdida = self.criterion(torch.tensor(s_original),torch.tensor(s_pred)) 

            # print("<<Perdida: "+str(perdida.item()) + f" epoca: {epoca}>>")
            self.writer.add_scalar(f'Perdida de entrenamiento predictivo de la red: {self.n_red}', float(perdida.item()), epoca)
            #writer.add_histogram(f'Serie de tiempo predicha para la epoca: {epoca}',torch.tensor(s_pred),epoca)

            self.pinta_pesos(epoca)
            plot_buf = utls.gen_plot(s_original,s_pred,perdida.item())

            image = PIL.Image.open(plot_buf)
            image = ToTensor()(image).unsqueeze(0)
            self.writer.add_image(f'Comportamiento de la serie de tiempo para la red: {0} durante el entrenamiento predictivo', image, epoca+1,dataformats='NCHW')

            if (perdida.item() <= self.tolerancia):
                print(f"-- Epoca final: {epoca+1} --")
                break
            lr_callback.reset()
        #writer.add_figure(f'Pérdida de entrenamiento de {n_red}', plt.gcf())
    
        print("---FIN DE ENTRENAMIENTO: entrena_LM---")

    def entrena_por_lote(self,lotes,t_ent,t_sal,e_predictivo,batch_size,lote_designado,epoca,lr,λ,lr_callback):
        s_original = [] # serie original
        s_pred = [] # serie a partir de predicciones
        ventana = 1 # ventana o batch
        n_lote = 1
        # se trata de la serie que va prediciendo la red conforme se modifican los parametros de esta
        serie = lotes[0][:t_ent] # primeros 8 elementos de la red (inputs[primer batch][primeros t_ent elementos])
        entradas_por_lote, salidas_por_lote  = [], [] # la conjunción de todas las entradas/salidas de los lotes
        for i in range(1,len(lotes)+1):# itera sobre el numero de lotes

                ejemplar = lotes[i-1]

                #se parten los primeros 8 días y se obtiene el noveno
                entradas = serie[ventana-1:ventana+t_ent-1] if e_predictivo else ejemplar[:t_ent] # se inicializan las entradas que va a tener la red
                #print(f">>Entradas: {entradas}")
                # se agregan las entradas del ejemplar a las de todo el lote
                entradas_por_lote.append(entradas)

                salida = ejemplar[t_sal].view(1)
                #print(f'>>Salida: {salida}')
                s_original.append(salida.item())
                # se agregan las entradas del ejemplar a las de todo el lote
                salidas_por_lote.append(salida)
                
                pred = self.red(entradas) #prediccion despues de haber modificado los pesos
                print(f"-----> Predicción pre entreno: {pred} ")
                serie = torch.cat((serie,pred))# Se precidce el resultado con la red despues del paso y se integra a la serie
                s_pred.append(pred.item())

                ventana = ventana + 1
                
                # Si ya se recorrió todo el lote
                if(i % batch_size == 0):

                    # print(f"entradas_por_lote: {entradas_por_lote}")
                    # print(f"salidas_por_lote: {salidas_por_lote}")
                        
                    # Vamos a evaluar el comportamiento del lr hasta ahora
                    # print(f"A comparar perdida actual: {torch.tensor(s_original[len(s_original)-t_ent:])} ,  {torch.tensor(s_pred[len(s_pred)-t_ent:])}")
                    perdida_actual = self.criterion(torch.tensor(s_original[len(s_original)-t_ent:]),torch.tensor(s_pred[len(s_pred)-t_ent:]))
                    # print(f"Perdida actual: {perdida_actual}")

                    if(perdida_actual <= 0.005 and lote_designado == n_lote):
                        lr_callback.decay_factor = lr_callback.decay_factor*0.8
                        # print(f">>nuevo factor: {lr_callback.decay_factor}")
                        lote_designado = lote_designado + 1

                    # Core/Nucleo del algoritmo
                    lm = LM(self.red,entradas_por_lote,salidas_por_lote,lr=lr,λ = λ) #se modifica cada parametro de la red segun el batch que se le de (Entrada y salida predecida contra salida esperada)
                    lm.exec()

                    # if(batch_size == 1):
                    #     print(f"prediccion post entreno: {self.red(entradas)}")
                    
                    lr = lr_callback.on_batch_begin(n_lote, logs={'loss': 0, 'epoca': epoca})
                    
                    # writer.add_scalar(f'Gradiente de {n_red}', metricas['grad'].norm(), ventana_en_epoca)
                    # writer.add_scalar(f'Matriz Hessiana de {n_red}', metricas['hessian'].norm(), ventana_en_epoca)
                    #ventana_en_epoca = ventana_en_epoca + 1

                    n_lote = n_lote+1
                    entradas_por_lote, salidas_por_lote  = [], []
        return s_original, s_pred

    def pinta_pesos(self,epoca):
        """
        Se encarga de representar en forma de matriz en escala de grises los pesos y sesgos de cada una de las capas de una red.
        
        Args:
            red: instancia de una red neuronal de la cual se pintaran los pesos.
            epoca: número de iteración actual del entrenamiento de la cual se toman los pesos.
        """
        for nombre, parametro in self.red.named_parameters():
            s2 = 1 if parametro.dim() <= 1 else parametro.shape[1]
            imagen_parametro = parametro.detach().cpu().numpy().reshape((1,parametro.shape[0],s2 ,1))
            #print(f"imagen_parametro: {imagen_parametro.shape}")
            self.writer.add_image(f'Pesos de la capa: {nombre} de la red {self.n_red}' , imagen_parametro, epoca+1, dataformats='NHWC')

    def cerrar_escritor(self):
        """
        Se encarga de cerrar el objeto escritor de TensorBoard
        """
        self.writer.close()

# def gen_plot(s_original,s_pred,perdida):
#     """Create a pyplot plot and save to buffer."""
#     plt.figure(figsize=(6, 4))
#     plt.plot(s_original)
#     plt.plot(s_pred,  label = f"Perdida: {float(perdida)}", color='#DA0C81')
#     plt.title('Serie original contra Predicha')
#     buf = io.BytesIO()
#     plt.savefig(buf, format='jpeg')
#     buf.seek(0)
#     return buf 
    
class CustomLearningRateScheduler():
    def __init__(self, initial_lr, decay_factor):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.iteration = 0  # Contador de iteraciones

    def on_batch_begin(self, batch, logs=None):
        #lr = self.initial_lr * (self.decay_factor ** self.iteration)
        lr = self.initial_lr / (1 + self.decay_factor * self.iteration)
        print(f"lr: {lr}, batch: {batch}")
        # if (logs['epoca'] == 1):
        #     self.writer.add_scalar("Learning Rate en cada batch: ",lr,batch)
        #print(red.summary())
        self.iteration += 1
        return lr
    
    def reset(self):
        self.iteration = 0
        print("Se resetea")
        return self.initial_lr
    
# def entrena_LM(red,n_red,inputs,epocas,lr,λ,t_ent = 8,t_sal = -1):
#     """
#     Entrena una red con el método de Levenverg-Marquardt 
#     a partir de un conjunto de entradas y una salida

#     Args:
#         red: instancia de la red a entrenar
#         n_red:
#         inputs: entradas de la red
#         epocas: número de iteraciones del algoritmo
#         lr: taza de aprendizaje que se aplicara como ponderación de la actualización de los parametros de la red
#         λ: ponderación inicial de la matriz hessiana 
#         t_ent: tamaño de la entrada a la red neuronal 
#         t_sal: tamaño de la salida de la red neuronal
#     """
#     print("---INICIO DE ENTRENAMIENTO: entrena_LM_pred---")
#     # print("paramtros antes: " + str([i for i in red.parameters()][0]))
#     s_original = []
#     s_pred = []
#     ventana_en_epoca = 1
#     for epoca in range(epocas): #1000 epocas
#         print(f"---Inicio de epoca: {epoca + 1}--")
#         ventana = 1
        
#         for entrada in inputs[n_red]:#por cada uno de los elementos del primer c. entrenamiento (el primero de los 6)(son 12 iteraciones)
#             # print(">>Ventana Actual: " + str(ventana))
#             #entradas = entrada[:, :t_ent][0]
            
#             #se parten los primeros 8 días y se obtiene el noveno
#             entradas = entrada[:t_ent]
#             #salida = entrada[:, t_sal]
#             salida = entrada[t_sal].view(1)
#             #print("Salida: " + str(salida))
#             s_original.append(salida.item())

#             lm = LM(red,entradas,salida,lr=lr,λ = λ)
#             metricas = lm.exec()
#             # print(perdidas)
#             pred = red(entradas)
#             s_pred.append(pred.item())
#             #print(">>>metricas: " + str(metricas['grad'].norm()) + " ," + str(metricas['hessian'].norm()))
#             writer.add_scalar(f'Gradiente de {n_red}', metricas['grad'].norm(), ventana_en_epoca)
#             writer.add_scalar(f'Matriz Hessiana de {n_red}', metricas['hessian'].norm(), ventana_en_epoca)
#             # for clave, loss in perdidas.items():
#             #     perdidas_totales.append(loss)
#             ventana_en_epoca = ventana_en_epoca + 1
#             ventana = ventana + 1
#         #print("paramtros final iteración: " + str([i for i in red.parameters()][0]))
#         #epoca = epoca + 1
#         # print("s_original: " + str(s_original) + "tamaño: " + str(len(s_original)))
#         # print("s_pred: " + str(s_pred) + "tamaño: " + str(len(s_pred)))
#         perdida = criterion(torch.tensor(s_original),torch.tensor(s_pred))
#         #print("<<Perdida: "+str(perdida.item()) + f" epoca: {epoca+1}")
#         writer.add_scalar(f'Pérdida de entrenamiento de la red: {n_red}', perdida, epoca+1)

#         pinta_pesos(red,epoca)

#         if (perdida.item() <= tolerancia):
#             print(f"--Se supera la tolerancia. Epoca final: {epoca+1}--")
#             break
#     #writer.add_figure(f'Pérdida de entrenamiento de {n_red}', plt.gcf())
#     print("---FIN DE ENTRENAMIENTO: entrena_LM---")