import pywt

def multilevel_dwt(data,wavelet,levels,mode = 'constant'):
    """
    Calcula la transformada de ondícula (wavelet) de un conjunto de datos o serie de tiempo y devuelve sus componentes
    [IMPORTANTE] cAn, cDn hacen referencia a los coeficientes de aprocimación y detalle al nivel n respectivamente, que surgen de la descomposición de la serie por 
    la transformada, mientras que An y Dn son los componentes de aproximación y detalle al nivel n, se generan al aplicar una reconstrucción parcial (upcoef) de 
    los coeficientes y son útiles para, al sumar cada uno de estos componentes, obtener la señal original.
    Args:
        data: serie o señal a descomponer
        wavelet: función con la cual se realizara la convolución, y en consecuencia la descomposición
        levels: nivel de la descomposición
        mode:
    """
    n = len(data)
    componentes = []
    At = data

    for l in range(1,levels+1):
        # descompone la serie de tiempo en un siguiente nivel y obtiene los coeficientes de detalle y aproximación
        (cAt, cDt) = pywt.dwt(At, wavelet, mode)
        # componentes de Aproximación 
        At = pywt.upcoef('a', cAt, wavelet, take = n)
        # componentes de Detalle
        Dt = pywt.upcoef('d', cDt, wavelet, take = n)
        componentes[:0] = [Dt]
        if (l == levels):
            componentes[:0] = [At]
    return componentes

