import pywt

def multilevel_dwt(data,wavelet,levels,mode = 'constant'):
    n = len(data)
    componentes = []
    At = data

    for l in range(1,levels+1):
        (cAt, cDt) = pywt.dwt(At, wavelet, mode)
        At = pywt.upcoef('a', cAt, wavelet, take = n)
        # print(len(pywt.upcoef('a', cA1, 'bior3.5')))
        Dt = pywt.upcoef('d', cDt, wavelet, take = n)
        componentes[:0] = [Dt]
        if (l == levels):
            componentes[:0] = [At]
    return componentes

