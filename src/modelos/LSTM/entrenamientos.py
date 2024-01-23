from keras.callbacks import Callback
from keras import backend as K
import numpy as np
from keras.optimizers import SGD
from keras.losses import mean_squared_error
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/lstm')
import PIL.Image
from ...utilerias import utilerias as utls
from torchvision.transforms import ToTensor

def entrena(red,c_entrenamiento_n,y_entrenamiento,time_steps,lr=0.01,epocas=10,t_lote=1,optimizador=SGD):

    red.compile(optimizer=SGD(learning_rate=lr),loss='mean_squared_error')#SGD(learning_rate=1e-3)

    # Definir el callback con la función de la tasa de aprendizaje
    lr_callback = CustomLearningRateScheduler(initial_lr=lr, decay_factor=0.5,red = red)#0.9
    ts_cierre_s_pred = c_entrenamiento_n

    loss_m = []
    print(f"y_entrenamiento: {y_entrenamiento}")
    for epoca in range(epocas):  # Número de épocas
        ts_cierre_s_pred = c_entrenamiento_n[:time_steps] #:8 se toman los primeros 8 elementos del conjunto de entrenamiendo predictivo 
        ts_cierre_s_pred_post_entreno = c_entrenamiento_n[:time_steps]
        loss = []
        n_ejemplar = 1
        n_lote = 1
        x_lote = []
        # print(f"grtrt: {ts_cierre_s_pred}")
        for i in range(0,len(y_entrenamiento)):#time_steps+1
            print(i)
            # Obtener las características y la etiqueta actual
            ejemplar = ts_cierre_s_pred[i:i+time_steps,0].reshape(time_steps,1)
            #print(ejemplar.reshape(1,time_steps,1))

            x_lote.append(ejemplar)

            # Predicción del modelo 
            #prediccion = red.predict(x_actual)#.reshape(1,1,1)
            prediccion = red(ejemplar.reshape(1,time_steps,1))
            
            # Agregar la predicción a las características para el siguiente paso
            # print(ts_cierre_s_pred)
            print(f"ejemplar: {ejemplar}")
            print(f"ejemplar: {ejemplar.shape}")
            print(f"y: {np.array( y_entrenamiento[i])}")
            print(f"Predicción : { prediccion}")
            ts_cierre_s_pred = np.concatenate([ts_cierre_s_pred, prediccion])
            

            if(n_ejemplar == t_lote):
                
                #print(f"y: {np.array( y_entrenamiento[i-t_lote+1:i+1]).reshape(t_lote,1)}")
                
                #print(f"x_lote: {x_lote}")
                lr = float(red.optimizer.lr)
                print(f"Lr que voy a aplicar en el lote: {n_lote} es {lr}")
                print(f"lote que voy a entrenar: {np.array(x_lote)}")
                print(f"verdaderas salidas: {np.array( [y_entrenamiento[i-t_lote+1:i+1]]).shape}")
                print(f"PERDIDAAAA antes: {red.test_on_batch(np.array(x_lote),np.array( [y_entrenamiento[i-t_lote+1:i+1]]))}")
                train = red.train_on_batch(np.array(x_lote), np.array( [y_entrenamiento[i-t_lote+1:i+1]]))
                
                # print(f"train: {train}")
                loss.append(train)#np.array(y_entrenamiento[i:i+t_lote])
                prediccion_post_entrenamiento = red(ejemplar.reshape(1,time_steps,1))
                print(f"Predicción post entrenamiento : { prediccion_post_entrenamiento}")
                print(f"PERDIDAAAA despues: {red.test_on_batch(np.array(x_lote),np.array( [y_entrenamiento[i-t_lote+1:i+1]]))}")
                #ts_cierre_s_pred_post_entreno = np.concatenate([ts_cierre_s_pred_post_entreno, prediccion])
                lr_callback.on_batch_begin(n_lote, logs={'loss': train, 'epoca': epoca+1})  # Llamada al callback en cada lote
                #red.optimizer.lr =
                x_lote = []
                n_ejemplar = 0
                
                n_lote = n_lote + 1
                
            n_ejemplar = n_ejemplar+1
            print(">>>>>>>>>>>>>>>Fin lote ")

            
        #print(f"mean: {np.mean(np.array(loss))}")
        #loss_m.append(np.mean(np.array(loss)))
        mse = np.mean(np.array(mean_squared_error(c_entrenamiento_n,ts_cierre_s_pred[:,0])))
        loss_m.append(mse)
        writer.add_scalar(f'Perdida de entrenamiento predictivo de la red: ', mse, epoca+1)
        
        
        plot_buf = utls.gen_plot(c_entrenamiento_n,ts_cierre_s_pred,mse)

        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)
        writer.add_image(f'Comportamiento de la serie de tiempo para la red: {0} durante el entrenamiento predictivo', image, epoca+1,dataformats='NCHW')
        lr_callback.reset()
    writer.close()

    return loss_m

class CustomLearningRateScheduler(Callback):
    def __init__(self, initial_lr, decay_factor, red):
        super(CustomLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.iteration = 0  # Contador de iteraciones
        self.lote_designado = 1
        self.red=red

    def on_batch_begin(self, batch, logs=None):
        print(f"loss en el callback: {logs['loss']}, batch {batch}, lote_designado {self.lote_designado}")
        if (logs['loss'] <= 0.01 and batch == self.lote_designado):
            self.lote_designado = self.lote_designado + 1
            self.decay_factor = self.decay_factor  * 0.8
            print(f">>nuevo factor: {self.decay_factor*0.8}")
        #lr = self.initial_lr * (self.decay_factor ** self.iteration)
        lr = self.initial_lr / (1 + self.decay_factor * self.iteration)
        print(f"lr: {lr}, batch: {batch}")
        if (logs['epoca'] == 1):
            writer.add_scalar("Learning Rate en cada batch: ",lr,batch)
        #print(red.summary())
        K.set_value(self.red.optimizer.lr, lr)
        self.iteration += 1
    
    def reset(self):
        K.set_value(self.red.optimizer.lr, self.initial_lr)
        self.iteration = 0
        print("Se resetea")