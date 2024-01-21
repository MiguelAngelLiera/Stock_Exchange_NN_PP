#Grupo Financiero Inbursa
import pywt, csv, numpy as np
import utilerias.utilerias as utils
from modelos.DWT_Auto_regresivo.NARNN import NARNN 

# Abrir el archivo CSV en modo lectura
# with open('cierre.csv', newline='') as csvfile:

#     # Crear un lector de CSV
#     cierre = np.array(list(csv.reader(csvfile, delimiter=',')))

# cierre = cierre.transpose()
# cierre = np.delete(cierre, 0)
# print(cierre)

#     # Iterar por cada fila del archivo
#     #for fila in cierre:
#         # Imprimir la fila completa
#         #print(fila)


# (cA, cD) = pywt.dwt(cierre.tolist(), 'db1')
# print(cA)
# print(cD)