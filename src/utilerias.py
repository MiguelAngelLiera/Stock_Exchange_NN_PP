import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from levenberg_marquardt import LM

criterion = nn.MSELoss()

def normalizar(arr):
    """
    Normaliza cada uno de los elementos de un arreglo.
    """
    return np.vectorize(lambda x: (x-np.min(arr))/(np.max(arr)-np.min(arr)))(arr)

def desnormalizar(arr):
    """
    Elimina la normalización cada uno de los elementos de un arreglo.
    """
    return np.vectorize(lambda y: np.min(arr)*(-y) + np.min(arr) + np.max(arr)*y)(arr)

#Se descompone cada uno de los conjuntos de coeficientes en conjuntos de entrenamiento, prueba y validación
def generar_conjuntos(coeficientes,c_validacion,nivel_descomposicion):

    entrenamiento = [0] * (nivel_descomposicion + 1)
    prueba = [0] * (nivel_descomposicion + 1)
    validacion = [0] * (nivel_descomposicion + 1)

    for _ in range(len(coeficientes)):
        split_index = int(len(coeficientes[_]) * 0.7)

        train_data = coeficientes[_][:split_index]
        test_data = coeficientes[_][split_index:]
        val_data = []

        if (c_validacion):
            split_index = int(len(test_data) * 0.5)
            val_data = test_data
            test_data = test_data[:split_index]
            val_data = val_data[split_index:]
        
        entrenamiento[_] = train_data
        prueba[_] = test_data
        validacion[_] = val_data

    return entrenamiento,prueba,validacion

def forma_entrada(a, input_size):
    """
    Dado un arreglo, parte a este correspondiendo a la entrada de la red neuronal.
    """
    subarreglos = []
    for i in range(0, len(a), input_size):
        subarreglo = torch.Tensor(a[i:i+input_size]).unsqueeze(0)
        subarreglos.append(subarreglo)
    if(len(subarreglos[-1]) != input_size):
        subarreglos = subarreglos[:-1]#elimina la ultima entrada en caso de que no tenga el taño correcto
    return subarreglos



#se trata de los conjuntos de todas las entradas y salidas para todas las redes
entradas_por_red = []
salidas_por_red = []
# Entrenar la red neuronal


def genera_prediccion(c_pruebas,red):
    """
    Genera prediccion cada n días, usando los datos que se le dan, no los que predice
    """
    serie = torch.tensor([])
    for _ in c_pruebas:
        predicted_output = red(_[:, :8])
        #print("Salida predecida:")
        #print(predicted_output)
        #print(_[:, :8])
        serie = torch.cat((serie, _[:, :8], predicted_output), dim=1)

    return serie

# def entrena(red,n_red,inputs,t_ent = 8,t_sal = -1):
#     """
#     Entrena una red a partir de un conjunto de entradas y una salida
#     """
#     #print("Entrena")
#     for i in range(1000): #010 epocas
#         for i in inputs[n_red]:#por cada uno de los elementos del primer c. entrenamiento (el primero de los 6)(son 12 iteraciones)
#             #print("i:")
#             #print(i)
#             entradas = i[:, :t_ent]#se parten los primeros 8 días y se obtiene el noveno
#             #print(entradas)
#             salida = i[:, t_sal]
#             #print(salida)
#             train(entradas,salida,red)

# # Función de entrenamiento
# def train(input_data, target, modelo):
#     #target_data = torch.tensor([1.0]).unsqueeze(0)   # Valor objetivo
#     # Definir la función de pérdida y el optimizador
#     criterion = nn.MSELoss()
#     optimizer = optim.LBFGS(modelo.parameters(), lr=0.1)
#     def closure():
#         optimizer.zero_grad()
#         output = modelo(input_data)
#         loss = criterion(output, target)
#         loss.backward()
#         return loss.numpy()

#     optimizer.step(closure)

def error(modelo,input_data,target):
    return modelo(input_data)-target


def train(red,input_data, target, modelo):
    #target_data = torch.tensor([1.0]).unsqueeze(0)   # Valor objetivo
    # Definir la función de pérdida y el optimizador
    
    optimizer = optim.LBFGS(red.parameters(), lr=0.1)
    def closure():
        optimizer.zero_grad()
        output = error(modelo,input_data,target)
        ##loss = criterion(output,target)
        loss = criterion(output, target)
        #loss = torch.sum(output ** 2)##criterion(output, target)
        loss.backward()
        return loss

    optimizer.step(closure)

def train_SGD(red,input_data, target):
    optimizer = optim.SGD(red.parameters(), lr=0.1, momentum=0.4)#,maximize=True)
    optimizer.zero_grad()
    output = red(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

def train_ASGD(red,input_data, target, modelo):
    optimizer = optim.ASGD(red.parameters(), lr=0.1)
    optimizer.zero_grad()
    output = modelo(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

#se trata de los conjuntos de todas las entradas y salidas para todas las redes
entradas_por_red = []
salidas_por_red = []
# Entrenar la red neuronal
def entrena(red,n_red,inputs,epocas=1000,t_ent = 8,t_sal = -1):
    """
    Entrena una red a partir de un conjunto de entradas y una salida
    """
    for i in range(epocas): #1000 epocas
        for i in inputs[n_red]:#por cada uno de los elementos del primer c. entrenamiento (el primero de los 6)(son 12 iteraciones)
            entradas = i[:, :t_ent]#se parten los primeros 8 días y se obtiene el noveno
            salida = i[:, t_sal]
            #for _ in range(100):# se entrena con esas entradas y esa salida
            # output = red(entradas)
            # loss = criterion(output, salida)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            train_SGD(red,entradas,salida)

def entrena_LM(red,n_red,inputs,epocas=1000,t_ent = 8,t_sal = -1):
    """
    Entrena una red con el método de Levenverg-Marquardt 
    a partir de un conjunto de entradas y una salida
    """
    for i in range(epocas): #1000 epocas
        for i in inputs[n_red]:#por cada uno de los elementos del primer c. entrenamiento (el primero de los 6)(son 12 iteraciones)
            entradas = i[:, :t_ent]#se parten los primeros 8 días y se obtiene el noveno
            salida = i[:, t_sal]
            #for _ in range(100):# se entrena con esas entradas y esa salida
            # output = red(entradas)
            # loss = criterion(output, salida)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            lm = LM(red,entradas,salida)
            lm.exec()

def genera_salida(vect,tam,red):
    #print(vect)
    print(len(vect))
    c = vect[:tam]
    o = vect[:tam]
    copia_prueba_0 = vect.copy()
    for i in range(len(vect)-(tam-1)):
        print(">>>>>>i: " + str(i))
        print("Nueva Entrada:")
        print(torch.Tensor(c))
        predicted_output = red(torch.Tensor(c))
        print("Prediction: " + str(predicted_output.item()))
        if(i+tam < len(vect)):
            copia_prueba_0[i+tam]=predicted_output.item()
            c = np.concatenate((np.array(copia_prueba_0[i+1:i+tam]),np.array([predicted_output.item()])))
            o = np.concatenate((np.array(o),np.array([predicted_output.item()])))
            #print([predicted_output.item()])
            #print(prueba[0][i:i+7])
            print(np.concatenate((np.array(copia_prueba_0[i+1:i+tam]),np.array([predicted_output.item()]))))

    return o
