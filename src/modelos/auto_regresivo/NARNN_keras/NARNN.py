from keras import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Dense


class NARNN(Model):
    def __init__(self,input_dim,output_dim):
        super().__init__()#red_Dense ,self
        self.Dense1 = Dense(units=10,activation='tanh',input_shape=(input_dim, 1))
        #self.dropout1 = Dropout(0.2)
        self.Dense2 = Dense(units=10,activation='sigmoid')
        #self.dropout2 = Dropout(0.2)
        self.Dense3 = Dense(units=output_dim)

    def call(self, inputs):
        x = self.Dense1(inputs)
        x = self.Dense2(x)
        return self.Dense3(x)