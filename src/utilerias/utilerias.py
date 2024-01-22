# import sys
# import os
#print(os.path.dirname(str(sys.path[0])))
# sys.path.append(os.path.dirname(str(sys.path[0])))
#print(sys.path)
from matplotlib import pyplot as plt
import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
#from levenberg_marquardt import LM
import os
import io
from ..modelos.DWT_Auto_regresivo.NARNN import NARNN
from keras.models import Sequential
# from torch.utils.tensorboard import SummaryWriter

# criterion = nn.MSELoss()
# writer = SummaryWriter('logs')
# tolerancia = 0.001

def normalizar(arr):
    """
    Normaliza cada uno de los elementos de un arreglo.
    """
    return np.vectorize(lambda x: (x-np.min(arr))/(np.max(arr)-np.min(arr)))(arr)

# def desnormalizar(arr):
#     """
#     Elimina la normalización cada uno de los elementos de un arreglo.
#     """
#     return np.vectorize(lambda y: np.min(arr)*(-y) + np.min(arr) + np.max(arr)*y)(arr)

def desnormalizar(arr_normalizado, max_original, min_original):
    """
    Desnormaliza un arreglo de Python que se normalizó utilizando la fórmula
    def normalizar(arr):
        """
    return arr_normalizado * (max_original - min_original) + min_original

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

def corrimiento_t_1(a, input_size):
    """
    Dado un arreglo, parte a este correspondiendo a la entrada de la red neuronal y a un elemento de prueba
    (si la entrada de la red es de n, el tamaño del sub-arreglo es de tamaño n+1), hacidendo un corrimiento
    temporal de 1.
    : input_size: es el tamaño de los sub-arreglos que queremos crear
    """
    subarreglos = []
    #print("aaaaaaa: " + str(a))
    for i in range(len(a)-input_size+1):
        subarreglo = torch.Tensor(a[i:i+input_size])
        
        subarreglos.append(subarreglo)
        #print("subarreglo:"+ str(subarreglo))#s[-1].shape[0]
        # if(subarreglos[-1].shape[0] != input_size):
        #     subarreglos = subarreglos[:-1]#elimina la ultima entrada en caso de que no tenga el taño correcto
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
        # print("Salida predecida:" + str(predicted_output))
        #print(_[:, :8])
        serie = torch.cat((serie, _[:, :8], predicted_output), dim=1)

    return serie

def genera_prediccion_1(c_pruebas,red,t_ent):
    """
    Genera prediccion cada n días, usando los datos que se le dan, no los que predice
    """
    serie = c_pruebas[0][:t_ent].clone().detach()#obtiene los primeros 8 datos del conjunto de prueba
    
    for _ in c_pruebas:
        #print("entrada: " + str(_[:, :8]))
        predicted_output = red(_[:t_ent])
        # print("Salida predecida:" + str(predicted_output))
        serie = torch.cat((serie, predicted_output))#concatena la salida predicha con los datos predichos anteriores
    #     print("serie: " + str(serie))
    return serie

def genera_prediccion_predictiva(datos_iniciales,t_ent,t_datos,red):
    """
    Genera prediccion cada n días, usando los datos que predice
    :t_datos: tamaño del conjunto de datos
    """
    #serie = torch.tensor(c_pruebas[0][:, :t_ent][0].clone().detach())#obtiene los primeros 8 datos del conjunto de prueba
    serie = datos_iniciales
    ventana = 1
    for _ in range(t_datos):
        if isinstance(red, NARNN):
            predicted_output = red(serie[ventana-1:ventana-1+t_ent].clone().detach())
            
            #print("Salida predecida:" + str(predicted_output))
            serie = torch.cat((serie, predicted_output))#concatena la salida predicha con los datos predichos anteriores
        #print("serie: " + str(serie))
        if isinstance(red, Sequential):
            predicted_output = red.predict(np.array(serie[ventana-1:ventana-1+t_ent]).reshape(1, *red.layers[0].input_shape[1:]))
            serie = np.concatenate((serie, predicted_output.reshape(1))) 
            # print(f"serie: {serie}")
        ventana = ventana + 1
    return serie

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

def eliminar_archivos_registro(carpeta):
    # Obtenemos la lista de archivos en la carpeta logs
    files = os.listdir(carpeta)
    
    # Filtramos los archivos que tienen el sufijo "events"
    archivos_registro = [archivo for archivo in files if archivo.startswith("events")]
    
    # Eliminamos los archivos de registro
    for archivo in archivos_registro:
        os.remove(f"{carpeta}/" + archivo)

def clear_tensorboard_database():
    # Obtiene la ruta del directorio de logs
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    print(f"logs_dir: {logs_dir}")
    # Elimina la base de datos interna de TensorBoard
    db_path = os.path.join(logs_dir, "events.db")
    print(f"db_path: {db_path}")
    if os.path.exists(db_path):
        os.remove(db_path)

def take(rec, take=0):
    """
    Permite recortar los datos a solo los :take: centrales
    """
    rec_len = len(rec)
    if take > 0 and take < rec_len:
        left_bound = right_bound = (rec_len-take) // 2
        if (rec_len-take) % 2:
            # right_bound must never be zero for indexing to work
            right_bound = right_bound + 1

        return rec[left_bound:-right_bound] 
    
def gen_plot(s_original,s_pred,perdida):
    """Create a pyplot plot and save to buffer."""
    plt.figure(figsize=(6, 4))
    plt.plot(s_original)
    plt.plot(s_pred,  label = f"Perdida: {float(perdida)}", color='#DA0C81')
    plt.title('Serie original contra Predicha')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.close()
    buf.seek(0)
    return buf 

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

# def error(modelo,input_data,target):
#     return modelo(input_data)-target


# def train(red,input_data, target, modelo):
#     #target_data = torch.tensor([1.0]).unsqueeze(0)   # Valor objetivo
#     # Definir la función de pérdida y el optimizador
    
#     optimizer = optim.LBFGS(red.parameters(), lr=0.1)
#     def closure():
#         optimizer.zero_grad()
#         output = error(modelo,input_data,target)
#         ##loss = criterion(output,target)
#         loss = criterion(output, target)
#         #loss = torch.sum(output ** 2)##criterion(output, target)
#         loss.backward()
#         return loss

#     optimizer.step(closure)

# def train_SGD(red,input_data, target):
#     optimizer = optim.SGD(red.parameters(), lr=0.1, momentum=0.4)#,maximize=True)
#     optimizer.zero_grad()
#     output = red(input_data)
#     loss = criterion(output, target)
#     loss.backward()
#     optimizer.step()

# def train_ASGD(red,input_data, target, modelo):
#     optimizer = optim.ASGD(red.parameters(), lr=0.1)
#     optimizer.zero_grad()
#     output = modelo(input_data)
#     loss = criterion(output, target)
#     loss.backward()
#     optimizer.step()

# #se trata de los conjuntos de todas las entradas y salidas para todas las redes
# entradas_por_red = []
# salidas_por_red = []
# # Entrenar la red neuronal
# def entrena(red,n_red,inputs,epocas=1000,t_ent = 8,t_sal = -1):
#     """
#     Entrena una red a partir de un conjunto de entradas y una salida
#     """
#     for i in range(epocas): #1000 epocas
#         for i in inputs[n_red]:#por cada uno de los elementos del primer c. entrenamiento (el primero de los 6)(son 12 iteraciones)
#             entradas = i[:, :t_ent]#se parten los primeros 8 días y se obtiene el noveno
#             salida = i[:, t_sal]
#             #for _ in range(100):# se entrena con esas entradas y esa salida
#             # output = red(entradas)
#             # loss = criterion(output, salida)

#             # optimizer.zero_grad()
#             # loss.backward()
#             # optimizer.step()
#             train_SGD(red,entradas,salida)

# def entrena_LM(red,n_red,inputs,epocas=1,t_ent = 8,t_sal = -1):
#     """
#     Entrena una red con el método de Levenverg-Marquardt 
#     a partir de un conjunto de entradas y una salida
#     """
#     print("---INICIO DE ENTRENAMIENTO: entrena_LM_pred---")
#     # print("paramtros antes: " + str([i for i in red.parameters()][0]))
#     perdidas_totales = []
#     s_original = []
#     s_pred = []
#     #epoca = 1
#     for epoca in range(epocas): #1000 epocas
#         print(f"---Inicio de epoca: {epoca + 1}--")
#         ventana = 1
#         for entrada in inputs[n_red]:#por cada uno de los elementos del primer c. entrenamiento (el primero de los 6)(son 12 iteraciones)
#             # print(">>Ventana Actual: " + str(ventana))
#             #entradas = entrada[:, :t_ent][0]#se parten los primeros 8 días y se obtiene el noveno
#             entradas = entrada[:t_ent]
#             #salida = entrada[:, t_sal]
#             salida = entrada[t_sal]
#             # print("Entradass: " + str(entradas))
#             s_original.append(salida.item())

#             lm = LM(red,entradas,salida)
#             perdidas = lm.exec()
#             # print(perdidas)
#             pred = red(entradas)
#             s_pred.append(pred.item())

#             for clave, loss in perdidas.items():
#                 perdidas_totales.append(loss)
#             ventana = ventana + 1
#         #print("paramtros final iteración: " + str([i for i in red.parameters()][0]))

#         clave = 1
#         # for loss in perdidas_totales:
#         #     writer.add_scalar('Perdida', loss, clave)
#         #     clave = clave +1
#         #epoca = epoca + 1
#         # print("s_original: " + str(s_original) + "tamaño: " + str(len(s_original)))
#         # print("s_pred: " + str(s_pred) + "tamaño: " + str(len(s_pred)))
#         perdida = criterion(torch.tensor(s_original),torch.tensor(s_pred))
#         # print("<<Perdida: "+str(perdida.item()))
#         writer.add_scalar('Perdida', perdida, ventana)
#         if (perdida.item() <= tolerancia):
#             print(f"---epoca final: {epoca+1}--")
#             break
#     writer.close()
#     print("---FIN DE ENTRENAMIENTO: entrena_LM_pred---")


# def entrena_LM_pred(red,n_red,inputs,epocas=1,t_ent = 8,t_sal = -1):
#     """
#     Entrena una red con el método de Levenverg-Marquardt 
#     a partir de un conjunto de entradas y una salida
#     Va actualizando los parametros de entrenamiento con los datos que va prediciendo
#     """
#     print("---INICIO DE ENTRENAMIENTO: entrena_LM_pred---")
#     #print("paramtros antes: " + str([i for i in red.parameters()][0]))
#     perdidas_totales = []
#     s_original = []
#     s_pred = []
#     #epoca = 1
#     for epoca in range(epocas): #1000 epocas
#         ventana = 1
#         print(f"---Inicio de epoca: {epoca+1}--")
#         # print(inputs[n_red][0])
#         serie = inputs[n_red][0][:t_ent]#primeros 8 elementos de la red
#         for i in inputs[n_red]:#por cada uno de los elementos del primer c. entrenamiento (el primero de los 6)(son 12 iteraciones)
            
#             # print("INICIO DE EPOCA...")
#             # print(">>Ventana Actual: " + str(ventana))
#             # print("serie: " + str(serie))
#             #entradas = i[:, :t_ent]#se parten los primeros 8 días y se obtiene el noveno
#             entradas = serie[ventana-1:ventana+t_ent-1]
            
#             salida = i[t_sal]
#             # print("Entradass: " + str(entradas))
#             s_original.append(salida.item())
#             #Core del algoritmo
#             lm = LM(red,entradas,salida)
#             perdidas = lm.exec()

#             pred = red(entradas)
#             serie = torch.cat((serie,pred))# Se precidce el resultado con la red despues del paso y se integra a la serie
#             s_pred.append(pred.item())
#             #print(perdidas)
#             #print("paramtros red despues: " + str([i for i in red.parameters()][0]))
            
#             for clave, loss in perdidas.items():
#                 perdidas_totales.append(loss)
#             ventana = ventana + 1
#         #print("paramtros despues: " + str([i for i in red.parameters()][0]))

#         # for clave, loss in perdidas_totales.items():
#         #     print(f"Clave: {clave}, Valor: {loss}")
#         clave = 1
#         # for loss in perdidas_totales:
#         #     writer.add_scalar('Perdida', loss, clave)
#         #     clave = clave +1
#         # print("s_original: " + str(s_original) + "tamaño: " + str(len(s_original)))
#         # print("s_pred: " + str(s_pred) + "tamaño: " + str(len(s_pred)))
#         perdida = criterion(torch.tensor(s_original),torch.tensor(s_pred))
#         # print("<<Perdida: "+str(perdida.item()))
#         writer.add_scalar('Perdida', perdida, ventana)
#         if (perdida.item() <= tolerancia):
#             print(f"---epoca final: {epoca+1}--")
#             break
#         #epoca = epoca + 1
#     writer.add_figure('Pérdida de entrenamiento', plt.gcf(), global_step=10)
#     writer.close()
#     print("---FIN DE ENTRENAMIENTO: entrena_LM_pred---")


