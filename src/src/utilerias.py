import torch, numpy as np
import torch.nn as nn
import torch.optim as optim

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

# Función de entrenamiento
def train(red,input_data, target, modelo):
    #target_data = torch.tensor([1.0]).unsqueeze(0)   # Valor objetivo
    # Definir la función de pérdida y el optimizador
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(red.parameters(), lr=0.1)
    def closure():
        optimizer.zero_grad()
        output = modelo(input_data)
        loss = criterion(output, target)
        loss.backward()
        return loss

    optimizer.step(closure)

#se trata de los conjuntos de todas las entradas y salidas para todas las redes
entradas_por_red = []
salidas_por_red = []
# Entrenar la red neuronal
def entrena(inputs,red,n_red,t_ent = 8,t_sal = -1):
    """
    Entrena una red a partir de un conjunto de entradas y una salida
    """
    for i in range(100): #010 epocas
        for i in inputs[n_red]:#por cada uno de los elementos del primer c. entrenamiento (el primero de los 6)(son 12 iteraciones)
            entradas = i[:, :t_ent]#se parten los primeros 8 días y se obtiene el noveno
            salida = i[:, t_sal]
            #for _ in range(100):# se entrena con esas entradas y esa salida
            # output = red(entradas)
            # loss = criterion(output, salida)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            train(red,entradas,salida,red)

def genera_prediccion(c_pruebas,red):
    serie = torch.tensor([])
    for _ in c_pruebas:
        predicted_output = red(_[:, :8])
        #print(predicted_output)
        #print("AAAAA")
        #print(_[:, :8])
        serie = torch.cat((serie, _[:, :8], predicted_output), dim=1)

    return serie
