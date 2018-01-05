# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage.filters import gaussian_filter

'''
- can be imported by algorithms for 2d inversion as an input distribution
- define a density function e.g. f(T1,T2) = Gaussian peak + ridge
- e.g. dim(F(T1,T2)) = 100x100
'''

#def centered_gaussian(sigma, Nx, Ny):
#    trueF = np.zeros((Nx, Ny))
#    for i in range(Nx):
#        for j in range(Ny):
#            trueF[i, j] = gaussian(sigma, 50, 50, i, j)
#    trueF /= trueF.sum()
#    return trueF
    
def gaussian(sigma, T1, T2, tau1, tau2):
    px = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(tau1-T1)**2/(2*sigma**2))
    py = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(tau2-T2)**2/(2*sigma**2))
    return px*py

def single_gauss(Nx=100, Ny=100, T1=0.7, T2=0.2, sigmaGauss=3.0):
    T1min = 1e-3
    T1max = 10.0
    T2min = 1e-3
    T2max = 10.0
    T1Arr = np.logspace(np.log10(T1min),
                        np.log10(T1max),
                        Nx, endpoint=True,
                        base=10.0)
    T2Arr = np.logspace(np.log10(T2min),
                        np.log10(T2max),
                        Ny,
                        endpoint=True,
                        base=10.0)
    print("T1 = %.2f s" % T1)
    print("T2 = %.2f s" % T2)
    T1_index = np.where(T1Arr > T1)[0][0]
    T2_index = np.where(T2Arr > T2)[0][0]
    trueF = np.zeros([Nx, Ny])
    for i in range(Nx):
        for j in range(Ny):
            trueF[i, j] = gaussian(sigmaGauss, T1_index, T2_index, i, j)
    trueF /= trueF.sum()
    return trueF, T1Arr, T2Arr

def ridge_gauss(Nx=100, Ny=100, sigmaGauss=3.0, sigmaRidge=2):
    # sigmas are given in pixels
    T1min = 1e-3
    T1max = 10.0
    T2min = 1e-3
    T2max = 10.0
    T1Arr = np.logspace(np.log10(T1min),
                        np.log10(T1max),
                        Nx, endpoint=True,
                        base=10.0)
    T2Arr = np.logspace(np.log10(T2min),
                        np.log10(T2max),
                        Ny,
                        endpoint=True,
                        base=10.0)
    T1 = 0.7  # seconds #T1Arr[50]
    T2 = 0.2  # seconds #T2Arr[40]
    print("T1 = %.2f s" % T1)
    print("T2 = %.2f s" % T2)
    T1_index = np.where(T1Arr > T1)[0][0]
    T2_index = np.where(T2Arr > T2)[0][0]
    trueF = np.zeros([Nx, Ny])
    for i in range(Nx):
        for j in range(Ny):
            trueF[i, j] = gaussian(sigmaGauss, T1_index, T2_index, i, j)
    xUpper = 50
    xLower = 17
    m = 0.5
    n = 30
    F = np.zeros([Nx, Ny])
    for i in range(xLower, xUpper, 1):
        j = int(m * i + n)
        F[i, j] = 1.0
    Ridge = gaussian_filter(F, sigma=sigmaRidge)
    Ridge = Ridge/max(Ridge.ravel())
    trueF = trueF/max(trueF.ravel())
    trueF = trueF + Ridge.T
    trueF = trueF/np.sum(trueF.ravel())
    return trueF, T1Arr, T2Arr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    trueF, T1Arr, T2Arr = ridge_gauss(Nx=100,
                                      Ny=100,
                                      sigmaGauss=3,
                                      sigmaRidge=2)
    fig = plt.figure()
    ax = fig.add_axes([.1, .1, .8, .8])
    ax.pcolor(T2Arr,
              T1Arr,
              trueF,
              vmin=0.0,
              vmax=max(trueF.ravel()))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.plot(T1Arr, T2Arr, 'w--')
    ax.set_xlabel(r'$T_{2}\,/\, \mathrm{s}$')
    ax.set_ylabel(r'$T_{1}\,/\,\mathrm{s}$')
    plt.show()
