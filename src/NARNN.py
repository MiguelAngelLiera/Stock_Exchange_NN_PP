#Red Neuronal no lineal autoregresiva
#La estructura de la red es más que nada las entradas (los n valores anteriores de un instante de la serie) y la salida
#La arquitectura son 3 capas y la relu, luego la salida, la octava semana se le vuelve a meter al input y se obtiene la novena y as
# checar pk la dwt hace tan chiquitos los datos
import torch
import torch.nn as nn
import torch.nn.functional as F

class NARNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(NARNN, self).__init__()
        #self.hidden_dim = hidden_dim
        #self.num_layers = num_layers
        #self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)#capa lstm
        #self.fc1 = nn.Linear(hidden_dim, output_dim)#capa lineal
        self.fc1 = nn.Linear(input_dim,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,output_dim)

    def forward(self, x):
        """h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()#Crea tensores con las dimensiones especificadas
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc1(out[:, -1, :]) 
        return out"""
        tan_sigmoid = lambda a : F.tanh(F.sigmoid(a))
        x = tan_sigmoid(self.fc1(x))
        #print(y)
        #x = torch.sigmoid(self.fc1(x))
        #print(x)
        x = F.logsigmoid(self.fc2(x))
        x = self.fc3(x)
        return x