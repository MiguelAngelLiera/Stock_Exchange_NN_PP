import torch
import torch.nn as nn
import torch.nn.functional as F

class NARNN(nn.Module):
    """
    Red Neuronal no lineal autoregresiva
    Estructura de la red:
        Entradas (los n valores anteriores de un instante de la serie) y la salida
    La arquitectura son 3 capas y la relu, luego la salida, la novena semana se le vuelve a dar al a red como
    parte de la entrada en la siguiente iteración
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, nombre = 'NARNN'):
        super(NARNN, self).__init__()
        self.nombre = nombre
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
        x = F.logsigmoid(self.fc2(x))
        x = self.fc3(x)
        return x