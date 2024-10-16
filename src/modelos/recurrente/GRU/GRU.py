from keras import Model
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import Dense

class red_GRU(Model):
    """
    Red Neuronal con Unidades Recurrentes Cerradas
    
    Estructura de la red:
        Entrada: los n valores anteriores de un instante de la serie.
        Arquitectura: 4 capas con 50 celdas LSTM e intercaladas 4 capas de desactivación (dropout)
        del 20% y finalmente una última capa densamente conectada que comprende una sola neurona. 
        Salida: un solo valor que representa la semana consecuente a las n de entrada.
    """
    def __init__(self,input_dim,output_dim):
        super().__init__()#red_GRU,self
        self.GRU1 = GRU(units=50,return_sequences=True,input_shape=(input_dim, 1))
        self.dropout1 = Dropout(0.2)
        self.GRU2 = GRU(units=50,return_sequences=True)
        self.dropout2 = Dropout(0.2)
        self.GRU3 = GRU(units=50,return_sequences=True)
        self.dropout3 = Dropout(0.2)
        self.GRU4 = GRU(units=50)
        self.dropout4 = Dropout(0.2)
        self.dense = Dense(units=output_dim)

    def call(self, inputs):
        """
        Define el comportamiento del modelo cuando se llama.
        """
        x = self.GRU1(inputs)
        x = self.dropout1(x)
        x = self.GRU2(x)
        x = self.dropout2(x)
        x = self.GRU3(x)
        x = self.dropout4(x)
        x = self.GRU4(x)
        x = self.dropout4(x)
        return self.dense(x)
