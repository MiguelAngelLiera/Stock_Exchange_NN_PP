import torch.nn as nn
import torch.nn.functional as F

class NARNN(nn.Module):
    """
    Red Neuronal no Lineal Auto-regresiva
    
    Estructura de la red:
        Entrada: los n valores anteriores de un instante de la serie.
        Arquitectura: 3 capas densamente conectadas con 10 neuronas cada una. La primera con la función tangente-sigmode 
        y la segunda con logaritmo-sigmoide como funciones de activación. Luego la capa de salida,
        con una función lineal como activación comprende una sola neurona. 
        Salida: un solo valor que representa la semana consecuente a las n de entrada.
    """
    def __init__(self, t_entrada, t_salida, nombre = 'NARNN'):
        super(NARNN, self).__init__()
        self.nombre = nombre
        self.fc1 = nn.Linear(t_entrada,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,t_salida)

    def forward(self, x):
        """
        Paso de propagación hacia adelante de la red
        """
        tan_sigmoid = lambda a : F.tanh(F.sigmoid(a))
        x = tan_sigmoid(self.fc1(x))
        x = F.logsigmoid(self.fc2(x))
        x = self.fc3(x)
        return x