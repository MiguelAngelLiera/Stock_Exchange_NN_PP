import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def prueba_KS(m1,m2,modo_UD):
    M1 = np.load(f'errores/ACTINVRB/{m1}.npy')
    M2 = np.load(f'errores/ACTINVRB/{m2}.npy')

    M1_dist = np.load(f'distribuciones/ACTINVRB/{m1}.npy')
    M2_dist = np.load(f'distribuciones/ACTINVRB/{m2}.npy')

    ts_ks_statistic, ts_ks_p_value = ks_2samp(M1, M2)
    os_ks_statistic, os_ks_p_value = ks_2samp(M1, M2,modo_UD)

    plt.title(f'Distribución de errores entre {m1} y {m2}')
    plt.text(0.75, 0.1, f"KS: {round(ts_ks_statistic,4)}, Valor p: {round(ts_ks_p_value,6)}", fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
    plt.text(0.75, 0.05, f"KS: {round(os_ks_statistic,4)}, Valor p: {round(os_ks_p_value,6)}", fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)

    plt.plot(M1_dist, color = '#451952', label = f'{m1}') #Señal original
    plt.plot(M2_dist, color='#AE445A', label = f'{m2}') #Señal predicha

    plt.legend()
    plt.show()

    if ts_ks_p_value < 0.05:
        print("Se rechaza la hipotesis nula y se acepta la alternativa: \n Los errores no comparten la misma distribución, existe una diferencia estadísticamente significativa")
    else:
        print("Se acepta la hipotesis nula \n Los errores comparten la misma distribución, no existe una diferencia estadísticamente significativa")

    if os_ks_p_value < 0.05:
        print(f"Se rechaza la hipotesis nula y se acepta la alternativa: \n el modelo {m1} reporta un menor error estocastico")
    else:
        print(f"Se acepta la hipotesis nula \n el modelo {m2} reporta un menor error estocastico")