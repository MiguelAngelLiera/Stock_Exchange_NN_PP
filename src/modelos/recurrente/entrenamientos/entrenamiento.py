from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras import backend as K
from tensorflow import summary
import numpy as np
from keras.optimizers import SGD
from keras.losses import mean_squared_error
from torch.utils.tensorboard import SummaryWriter
import PIL.Image
from ....utilerias import utilerias as utls
from torchvision.transforms import ToTensor

s_vacia = ""
s_entr_pred = 'durante el entrenamiento predictivo'

def entrena(red,X_entrenamiento,y_entrenamiento,time_steps,refuerzo=0,lr=0.001,epocas=10,t_lote=1,callbacks = [], optimizador=SGD,loss='mean_squared_error',log_dir='logs'):
    
    writer = SummaryWriter(log_dir)
    #Se compila la red con los parametros de entrada de la función
    red.compile(optimizer=optimizador(learning_rate=lr),loss=loss)#SGD(learning_rate=1e-3)
    
    ts_cierre_s_pred = X_entrenamiento

    loss_m = []

    for epoca in range(epocas):  # Número de épocas
        ts_cierre_s_pred = X_entrenamiento[:time_steps] #:8 se toman los primeros 8 elementos del conjunto de entrenamiendo predictivo 

        n_ejemplar = 1 #número de ejemplar que ocupa en el lote
        n_lote = 1 
        x_lote = []

        for i in range(0,len(y_entrenamiento)):#time_steps+1

            # Obtener las actuales n características y las empaqueta en un ejemplar que pertenecera al lote 
            ejemplar = ts_cierre_s_pred[i:i+time_steps,0].reshape(time_steps,1)
            x_lote.append(ejemplar)

            # Predicción del modelo 
            prediccion = red(ejemplar.reshape(1,time_steps,1))

            # if((i+1)%refuerzo==0):
            #     # El refuerzo es un dato que tomamos del conjunto real y no de los que predice la red para 'ayudarle' al entrenamiento a que e corrija correctamente
            #     prediccion = np.array([[y_entrenamiento[i]]])
            
            print(f"Ejemplar x: {ejemplar}" + f" | y: {np.array(y_entrenamiento[i])} | " + f"Predicción actual: {prediccion}")

            # Agrega la predicción a la serie para que pueda predecir a partir de esta en el siguiente paso
            ts_cierre_s_pred = np.concatenate([ts_cierre_s_pred, prediccion])

            if(n_ejemplar == t_lote):# si el numero de ejemplar es igual al tamño del lote, es decir, el lote esta lleno, se puede entrenar la red

                lr = float(red.optimizer.lr)
                print(f"Lr que voy a aplicar en el lote es {lr}") # se comenta por popsible debug
                #print(f"Verdaderas salidas: {np.array( [y_entrenamiento[i-t_lote+1:i+1]])}")
                #print(f"PERDIDAAAA antes: {red.test_on_batch(np.array(x_lote),np.array( [y_entrenamiento[i-t_lote+1:i+1]]))}")

                # Se usa el método train_on_batch, que nos permite realizar un paso de la optimizacion, es de cir, un solo ajuste a los pesos
                # a partir de un lote lleno de ejemplares (caracteristicas) junto con sus respectivas etiquetas
                train = red.train_on_batch(np.array(x_lote), np.array( y_entrenamiento[i-t_lote+1: i+1]))

                # prediccion_post_entrenamiento = red(ejemplar.reshape(1,time_steps,1))
                #print(f"Predicción post entrenamiento : {prediccion_post_entrenamiento}")
                #print(f"PERDIDAAAA despues: {red.test_on_batch(np.array(x_lote),np.array( [y_entrenamiento[i-t_lote+1:i+1]]))}")
                #ts_cierre_s_pred_post_entreno = np.concatenate([ts_cierre_s_pred_post_entreno, prediccion])

                # regisrto de métricas en callbacks
                for callback in callbacks:
                    if hasattr(callback, "on_batch_begin"):
                        callback.on_batch_begin(n_lote, logs={'loss': train, 'epoca': epoca+1})  # Llamada al callback en cada lote

                x_lote = []
                n_ejemplar = 0
                
                n_lote = n_lote + 1
                
            n_ejemplar = n_ejemplar+1
            print(">Fin lote<")

        # Registra la perdida actual de todo el conjunto de entrenamiento contra lo que la red fue prediciendo en cada iteración    
        mse = np.mean(np.array(mean_squared_error(X_entrenamiento,ts_cierre_s_pred[:,0])))
        loss_m.append(mse)
        writer.add_scalar(f'Perdida de entrenamiento predictivo de la red: {red.name}', mse, epoca+1)
        
        plot_buf = utls.gen_plot(X_entrenamiento,ts_cierre_s_pred,mse)
        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)
        writer.add_image(f'Comportamiento de la serie de tiempo para la red: {red.name} durante el entrenamiento predictivo', image, epoca+1,dataformats='NCHW')

        #Se resetea el callback del learning rate
        for callback in callbacks:
            if hasattr(callback, "on_epoch_end"):
                callback.on_epoch_end(epoca+1) 
            if hasattr(callback, "reset"):
                callback.reset()
    writer.close()

    return loss_m

class CalendarizadorTasaAprendizaje(TensorBoard,Callback):
    def __init__(self, log_dir, initial_lr, decay_factor, red):
        super(CalendarizadorTasaAprendizaje, self).__init__()
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.iteration = 0  # Contador de iteraciones
        self.lote_designado = 1
        self.red=red
        self.writer = summary.create_file_writer(log_dir)

    def on_batch_begin(self, batch, logs=None):
        print(f"loss en el callback: {logs['loss']}, batch {batch}, lote_designado {self.lote_designado}")
        if (logs['loss'] <= 0.001 and batch == self.lote_designado):
            # El lote designado se trata del lote a partir del cual el lr empezara a decaer, esto para que los primeros lotes
            # tengae mayor prioridad (se les de mejor lr y en consecuencia aprenda la red mejor estos)
            self.lote_designado = self.lote_designado + 1
            self.decay_factor = self.decay_factor * 0.8
            print(f">>nuevo factor: {self.decay_factor}")
        #lr = self.initial_lr * (self.decay_factor ** self.iteration)
        lr = self.initial_lr / (1 + self.decay_factor * self.iteration)
        print(f"lr: {lr}, batch: {batch}")
        if (logs['epoca'] == 1):
            with self.writer.as_default():
                summary.scalar("Learning Rate en cada batch: ",lr,batch)
        #print(red.summary())
        K.set_value(self.red.optimizer.lr, lr)
        self.iteration += 1
    
    def reset(self):
        K.set_value(self.red.optimizer.lr, self.initial_lr)
        self.iteration = 0
        print("Se resetea")

class CalendarizadorPredicciones(TensorBoard):
    def __init__(self, log_dir, datos_validacion, y_original, e_predictivo= False):
        super().__init__()
        self.log_dir = log_dir
        self.writer = summary.create_file_writer(log_dir)
        self.datos_validacion = datos_validacion
        self.e_predictivo = e_predictivo
        self.y_original = y_original

    def on_epoch_end(self, epoch, logs=None):
        # Obtener las predicciones del modelo en el conjunto de validación
        y_pred = self.model.predict(self.datos_validacion)  # Ajusta esto según tus datos de validación
        y_pred = np.reshape(y_pred, (y_pred.shape[0]))
        #y_original = np.array(self.datos_validacion)
        # Log de las predicciones como un histograma en TensorBoard
        with self.writer.as_default():
            #for pred in y_pred:
             #   summary.scalar(f'Prediccion de la red {self.model.name} en cada epoca', pred, step=epoch)
            # np.reshape(self.y_original,(self.y_original.shape[0]))
            plot_buf = utls.gen_plot(self.y_original,y_pred,0)

            image = PIL.Image.open(plot_buf)
            image = ToTensor()(image).unsqueeze(0)
            image = image.permute(0, 2, 3, 1) #lo ordenamos en forma NHCW
            s_pred= 'predictivo'
            summary.image(f'Comportamiento de la serie de tiempo para la red: {self.model.name} durante el entrenamiento {s_pred if self.e_predictivo else s_vacia}', image, epoch+1)
            
        self.writer.flush()

class CalendarizadorPesos(TensorBoard):
    def __init__(self, log_dir, e_predictivo= False, genera_histogramas = False, red = 0):
        super().__init__()
        self.log_dir = log_dir
        self.writer = summary.create_file_writer(log_dir)
        self.e_predictivo = e_predictivo
        self.genera_histogramas = genera_histogramas
        if e_predictivo:
            self.model = red

    def on_epoch_end(self, epoca, logs=None):
        #obtenermos altura maxima de entre todos los pesos de la red
        altura_max = 0
        for capa in self.model.layers:
            for componente_de_peso in capa.get_weights():
                altura_max = componente_de_peso.shape[0] if componente_de_peso.shape[0] > altura_max else altura_max
                if componente_de_peso.ndim > 1:
                    altura_max = componente_de_peso.shape[1] if componente_de_peso.shape[1] > altura_max else altura_max

        imagen_total = np.empty((1, altura_max, 0, 1)) # se inicializa la imagen total de todos los componentes de cada capa
        # Obtener los pesos de la capa deseada
        for capa in self.model.layers:
            imagen_capa = np.empty((1, altura_max, 0, 1))

            for componente_de_peso in capa.weights: # Pesos de la primera capa LSTM
                if self.genera_histogramas:
                    with self.writer.as_default():    
                        summary.histogram(f"Peso: {componente_de_peso.name} de la red: {self.model.name}", data=componente_de_peso, step=epoca)
                imagen_parametro = np.empty((1, altura_max, 0, 1))

                anchura = 1 if componente_de_peso.numpy().ndim <= 1 else componente_de_peso.numpy().shape[1]
                altura = componente_de_peso.numpy().shape[0]

                if anchura > altura:
                    altura_t = altura
                    altura = anchura
                    anchura = altura_t

                # Concatena la imagen de los pesos de la componente de la capa con una linea blanca divisora
                imagen_parametro = np.concatenate((componente_de_peso.numpy().reshape((1,altura,anchura,1)), np.ones((1,altura, 1, 1))), axis=2)

                #Se extiende la longitud de la imagen a la altura máxima
                longitud_pad = ((0, 0), (0, altura_max-altura if altura_max > altura else 0), (0, 0), (0, 0))
                imagen_parametro = np.pad(imagen_parametro, longitud_pad, mode='constant',constant_values=1)

                imagen_capa = np.concatenate((imagen_capa, imagen_parametro), axis = 2)
                imagen_capa = np.concatenate((imagen_capa, np.ones((1, altura_max, 1, 1))), axis=2)

            with self.writer.as_default():
                if (imagen_capa.shape[2] != 0):
                    summary.image(f'Pesos de la red {self.model.name} la capa: {capa.name}', imagen_capa, epoca+1)#
            
            imagen_total = np.concatenate((imagen_total, imagen_capa), axis = 2)
        
            #imagen_total = imagen_total.resahpe(1,imagen_total.shape[1],1,imagen_total.shape[2]) #lo ordenamos en forma NHCW
        #print(f"imagen_total: {imagen_total}")
        with self.writer.as_default():
            summary.image(f'Pesos de la red {self.model.name} {s_entr_pred if self.e_predictivo else s_vacia}', imagen_total, epoca+1)#
        
           # summary.histogram(f"Peso de la red: ", data=8, step=epoca+1)
        #self.writer.flush()
            
# --------------- v1 entr. para redes neuronales lstm de manera auto-predictiva -------------------
            
            # ts_cierre_s_pred = c_entrenamiento_n

# loss_m = []
# for epoch in range(100):  # Número de épocas
#     ts_cierre_s_pred = c_entrenamiento_n[:time_steps]#se obtienen los primeros time_steps(8) elementos del trainig set
#     loss = []
#     X_train_c_pred = []
#     # print(f"grtrt: {ts_cierre_s_pred}")
#     for i in range(time_steps, N):
#         # Obtener las características y la etiqueta actual
#         x_actual = ts_cierre_s_pred[i-time_steps:i,0]
#         X_train_c_pred.append(x_actual)
#         x_actual = x_actual.reshape(1,time_steps,1)

#         y_actual = np.array([y_entrenamiento_n[i-time_steps]])

#         print(f"x_actual: {x_actual}")
#         print(f"y_actual: {y_actual}")
        
#         # Entrenar el modelo con las nuevas características y la etiqueta real
#         #loss.append(red.train_on_batch(x_actual, y_actual))

#         # Predicción del modelo
#         #prediccion = red.predict(x_actual)#.reshape(1,1,1)
#         prediccion = red(x_actual)
        
#         # Agregar la predicción a las características para el siguiente paso
#         # print(ts_cierre_s_pred)
#         print(f"prediccion: {prediccion}")
#         ts_cierre_s_pred = np.concatenate([ts_cierre_s_pred, prediccion])



#     # print(f"mean: {np.mean(np.array(loss))}")
#     # loss_m.append(np.mean(np.array(loss)))
#     X_train_c_pred = np.array(X_train_c_pred)
#     X_train_c_pred = np.reshape(X_train_c_pred, (X_train_c_pred.shape[0], X_train_c_pred.shape[1], 1))
#     history = red.fit(X_train_c_pred, y_entrenamiento_n, epochs=1, batch_size=32)
#     loss = history.history['loss']
#     loss_m.append(loss)
#     loss_m.append(mean_squared_error(c_entrenamiento_n,ts_cierre_s_pred[:,0]))