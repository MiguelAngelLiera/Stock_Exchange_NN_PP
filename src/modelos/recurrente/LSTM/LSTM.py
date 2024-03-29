from keras import Model
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

class red_LSTM(Model):
    """
    Red Neuronal con celdas de Memoria de Corto y Largo Plazo
    
    Estructura de la red:
        Entrada: los n valores anteriores de un instante de la serie.
        Arquitectura: 4 capas con 50 celdas LSTM e intercaladas 4 capas de desactivación (dropout)
        del 20% y finalmente una última capa densamente conectada que comprende una sola neurona. 
        Salida: un solo valor que representa la semana consecuente a las n de entrada.
    """
    def __init__(self,input_dim,output_dim):
        super().__init__()#red_LSTM,self
        self.LSTM1 = LSTM(units=50,return_sequences=True,input_shape=(input_dim, 1))
        self.dropout1 = Dropout(0.2)
        self.LSTM2 = LSTM(units=50,return_sequences=True)
        self.dropout2 = Dropout(0.2)
        self.LSTM3 = LSTM(units=50,return_sequences=True)
        self.dropout3 = Dropout(0.2)
        self.LSTM4 = LSTM(units=50)
        self.dropout4 = Dropout(0.2)
        self.dense = Dense(units=output_dim)

    def call(self, inputs):
        """
        Define el comportamiento del modelo cuando se llama.
        """
        x = self.LSTM1(inputs)
        x = self.dropout1(x)
        x = self.LSTM2(x)
        x = self.dropout2(x)
        x = self.LSTM3(x)
        x = self.dropout4(x)
        x = self.LSTM4(x)
        x = self.dropout4(x)
        return self.dense(x)

