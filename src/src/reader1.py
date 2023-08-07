#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv, numpy as np
     
def leer_archivo(ruta_archivo):
    with open(ruta_archivo, newline='') as csvfile:

        # Crear un lector de CSV
        cierre = np.array(list(csv.reader(csvfile, delimiter=',')))
        
    cierre = cierre.transpose()
    cierre = np.delete(cierre, 0)
    cierre = np.flip(cierre, axis= 0)

    return cierre


# In[2]:



     

