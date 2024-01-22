
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

def LSTM(input_dim,output_dim=1):
    #Se entrena con un aprendizaje por reforzamiento del profesor
    red = Sequential()
    red.add(LSTM(units=50,return_sequences=True,input_shape=(input_dim, 1)))#tiene un tamaño de entrada de 8 y de salida 1, input_shape = (8, 1)
    red.add(Dropout(0.2))#Se apagan aleatoriamente el 20% de las neuronas de la capa anterior
    red.add(LSTM(units=50,return_sequences=True))
    red.add(Dropout(0.2))
    red.add(LSTM(units=50,return_sequences=True))
    red.add(Dropout(0.2))
    red.add(LSTM(units=50))
    red.add(Dropout(0.2))
    red.add(Dense(units=output_dim))
    return red
