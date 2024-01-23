
from keras import Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

# def LSTM(input_dim,output_dim=1):
#     #Se entrena con un aprendizaje por reforzamiento del profesor
#     red = Sequential()
#     red.add(LSTM(units=50,return_sequences=True,input_shape=(input_dim, 1)))#tiene un tamaño de entrada de 8 y de salida 1, input_shape = (8, 1)
#     red.add(Dropout(0.2))#Se apagan aleatoriamente el 20% de las neuronas de la capa anterior
#     red.add(LSTM(units=50,return_sequences=True))
#     red.add(Dropout(0.2))
#     red.add(LSTM(units=50,return_sequences=True))
#     red.add(Dropout(0.2))
#     red.add(LSTM(units=50))
#     red.add(Dropout(0.2))
#     red.add(Dense(units=output_dim))
#     return red

class red_LSTM(Model):
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
        x = self.LSTM1(inputs)
        x = self.dropout1(x)
        x = self.LSTM2(x)
        x = self.dropout2(x)
        x = self.LSTM3(x)
        x = self.dropout4(x)
        x = self.LSTM4(x)
        x = self.dropout4(x)
        return self.dense(x)

