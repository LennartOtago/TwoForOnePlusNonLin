import numpy as np
import matplotlib as mpl
#from puwr import tauint
#from importetFunctions import *
import time
import pickle as pl


#import matlab.engine
from functions import *
#from errors import *
from scipy import constants, optimize
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
#import tikzplotlib

import pandas as pd
from numpy.random import uniform, normal, gamma
import scipy as scy
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

#mpl.rc('text.latex', preamble=r"\boldmath")

""" for plotting figures,
PgWidth in points, either collumn width page with of Latex"""
def scientific(x, pos):
    # x:  tick value
    # pos: tick position
    return '%.e' % x
scientific_formatter = FuncFormatter(scientific)

fraction = 1.5
dpi = 300
PgWidthPt = 245
#PgWidthPt = 1/0.3 *fraction * 421/4 #phd
defBack = mpl.get_backend()
mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 12,#1/0.3 *fraction *
                     'text.usetex': True,
                     'font.family' : 'serif',
                     'font.serif'  : 'cm',
                     'text.latex.preamble': r'\usepackage{bm, amsmath}'})

""" for plotting histogram and averaging over lambda """
n_bins = 40

""" for MwG"""
burnIn = 50

betaG = 1e-10# 1e-18#
betaD = 1e3#9e3#1e-3#1e-10#1e-22#  # 1e-4

import numpy as np


dir = '/home/lennartgolks/PycharmProjects/firstModelCheckPhD/'
dir = '/Users/lennart/PycharmProjects/firstModelCheckPhD/'
dir = '/Users/lennart/PycharmProjects/TTDecomposition/'
B_inv_A_trans_y0 = np.loadtxt(dir + 'B_inv_A_trans_y0.txt')
VMR_O3 = np.loadtxt(dir + 'VMR_O3.txt')
pressure_values = np.loadtxt(dir + 'pressure_values.txt')
temp_values = np.loadtxt(dir + 'temp_values.txt')
height_values = np.loadtxt(dir + 'height_values.txt')
A = np.loadtxt(dir + 'AMat.txt')
APress = np.loadtxt(dir + 'AP.txt')
ATemp = np.loadtxt(dir + 'AT.txt')
APressTemp = np.loadtxt(dir + 'APT.txt')
gamma0 = np.loadtxt(dir + 'gamma0.txt')
y = np.loadtxt(dir + 'nonLinDataY.txt')

RealMap = np.loadtxt(dir + 'RealMap.txt')
L = np.loadtxt(dir + 'GraphLaplacian.txt')
theta_scale_O3 = np.loadtxt(dir + 'theta_scale_O3.txt')
tang_heights_lin = np.loadtxt(dir + 'tan_height_values.txt')
A_lin = np.loadtxt(dir +'ALinMat.txt')
m, n = A_lin.shape
ind = 623

tol = 1e-8
SpecNumMeas, SpecNumLayers= A_lin.shape
y =  y.reshape((SpecNumMeas,1))
height_values = height_values.reshape((SpecNumLayers,1))
newA = RealMap @ A
newATy = np.matmul(newA.T, y)
newATA = np.matmul(newA.T,newA)


Ax =np.matmul(newA, VMR_O3 * theta_scale_O3)
AxPT =np.matmul(RealMap@ APressTemp, pressure_values.reshape((n, 1))/temp_values.reshape((n, 1)) )
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin)

ax1.plot(AxPT, tang_heights_lin)
ax1.scatter(y, tang_heights_lin)
ax1.plot(y, tang_heights_lin)
plt.show()

def MinLogMargPostFirst(params):#, coeff):
    tol = 1e-8
    # gamma = params[0]
    # delta = params[1]
    gam = params[0]
    lamb = params[1]
    if lamb < 0  or gam < 0:
        return np.nan

    #ATA = np.matmul(A.T,A)
    Bp = newATA + lamb * L

    #y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
    #ATy = np.matmul(A.T, y)
    B_inv_A_trans_y, exitCode = gmres(Bp, newATy[:,0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A, L,  lamb)
    F = f(newATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( betaD *  lamb * gam + betaG *gam)


gammaMin0, lam0 = optimize.fmin(MinLogMargPostFirst, [gamma0,(np.var(VMR_O3) * theta_scale_O3) /gamma0 ])
mu0 = 0
print(lam0)
print('delta:' + str(lam0*gammaMin0))
print('gamma:' + str(gammaMin0))
##

def pressFunc(x, b1, b2, h0, p0):
    b = np.ones(len(x))
    b[x>h0] = b2
    b[x<=h0] = b1
    return -b * (x - h0) + np.log(p0)

popt, pcov = scy.optimize.curve_fit(pressFunc, height_values[:,0], np.log(pressure_values), p0=[-2e-2,-2e-2, 18, 15])

print(popt)
def log_postTP(params, means, sigmas, popt, A, y, height_values, gamma0):
    n = len(height_values)
    h0Mean = means[0]
    h0Sigm = sigmas[0]

    h1Mean = means[1]
    h1Sigm = sigmas[1]

    h2Mean = means[2]
    h2Sigm = sigmas[2]

    h3Mean = means[3]
    h3Sigm = sigmas[3]

    h4Mean = means[4]
    h4Sigm = sigmas[4]

    h5Mean = means[5]
    h5Sigm = sigmas[5]

    a0Mean = means[6]
    a0Sigm = sigmas[6]

    a1Mean = means[7]
    a1Sigm = sigmas[7]

    a2Mean = means[8]
    a2Sigm = sigmas[8]

    a3Mean = means[9]
    a3Sigm = sigmas[9]

    a4Mean = means[10]
    a4Sigm = sigmas[10]

    b0Mean = means[11]
    b0Sigm = sigmas[11]

    sigmaGrad1 = sigmas[12]
    sigmaGrad2 = sigmas[13]
    sigmaH = sigmas[14]
    sigmaP = sigmas[15]


    h0 = params[0]
    h1 = params[1]
    h2 = params[2]
    h3 = params[3]
    h4 = params[4]
    h5 = params[5]
    a0 = params[6]
    a1 = params[7]
    a2 = params[8]
    a3 = params[9]
    a4 = params[10]
    b0 = params[11]

    b1 = params[12]
    b2 = params[13]
    h0P =params[14]
    p0 = params[15]
    gam = gamma0#params[15]
    paramT = [h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, b0]
    paramP = [b1, b2, h0P, p0]
    # postDatT = - gamma0 * np.sum((y - A @ (1 / temp_func(height_values, *paramT).reshape((n, 1)))) ** 2)
    # postDatP = gamma0 * 1e-3 * np.sum((y - A @ pressFunc(height_values[:, 0], *paramP).reshape((n, 1))) ** 2)
    PT = np.exp(pressFunc(height_values[:, 0], *paramP).reshape((n, 1))) /temp_func(height_values, *paramT).reshape((n, 1))
    #postDat = + SpecNumMeas / 2  * np.log(gam) - 0.5 * gam * np.sum((y - A @ PT ) ** 2)- betaG * gam
    postDat = - 0.5 * gam * np.sum((y - A @ PT) ** 2)

    #postDat = 0
    Values =     - ((h0 - h0Mean) / h0Sigm) ** 2 - ((h1 - h1Mean) / h1Sigm) ** 2 - (
                (h2 - h2Mean) / h2Sigm) ** 2 - (
                        (h3 - h3Mean) / h3Sigm) ** 2 - ((h4 - h4Mean) / h4Sigm) ** 2 - (
                        (h5 - h5Mean) / h5Sigm) ** 2 - ((a0 - a0Mean) / a0Sigm) ** 2 - (
                        (a1 - a1Mean) / a1Sigm) ** 2 - ((a2 - a2Mean) / a2Sigm) ** 2 \
                - ((a3 - a3Mean) / a3Sigm) ** 2 - ((a4 - a4Mean) / a4Sigm) ** 2 - ((b0 - b0Mean) / b0Sigm) ** 2 \
                - ((popt[0] - b1) / sigmaGrad1) ** 2 - ((popt[1] - b2) / sigmaGrad2) ** 2 - (
                            (popt[3] - p0) / sigmaP) ** 2 \
                - ((popt[2] - h0P) / sigmaH) ** 2

    return postDat + 0.5 * Values
means = np.zeros(16)
sigmas = np.zeros(16)

means[0] = 11
means[1] = 20
means[2] = 32
means[3] = 47
means[4] = 51
means[5] = 71
means[6] = -6.5
means[7] = 1
means[8] = 2.8
means[9] = -2.8
means[10] = -2
means[11] = 288.15
means[12] = popt[0]
means[13] = popt[1]
means[14] = popt[2]
means[15] = popt[3]

sigmas[0] = 0.5 * 0.1
sigmas[1] = 3 * 0.1
sigmas[2] = 1 * 0.1
sigmas[3] = 2 #* 0.1
sigmas[4] = 2 #* 0.1
sigmas[5] = 2 #* 0.1
sigmas[6] = 0.01 * 0.1
sigmas[7] = 0.01
sigmas[8] = 0.1
sigmas[9] = 0.01
sigmas[10] = 0.01
sigmas[11] = 2 * 0.1

#sigmas[0:12] = sigmas[0:12] * 0.02


sigmaP = 0.025
sigmaH = 0.5
sigmaGrad1 = 0.005
sigmaGrad2 = 0.01

sigmas[12] = sigmaGrad1
sigmas[13] = sigmaGrad2
sigmas[14] = sigmaH
sigmas[15] = sigmaP
#

newAPT = RealMap @ APressTemp

log_post = lambda params: -log_postTP(params, means, sigmas, popt, newAPT, y, height_values, gamma0)

import glob
dir = '/home/lennartgolks/PycharmProjects/TTDecomposition/'
dim = len(glob.glob(dir + 'ttSQcoreTP*.txt'))

univarGrid = [None] * dim
for i in range(0, dim):
    univarGrid[i] = np.loadtxt(dir+'uniVarGridTP' +str(i)+ '.txt')
    print(i)
    print(univarGrid[i][0])
    print(univarGrid[i][-1])
gridSize =100
factor= 50
univarGrid = [np.linspace(means[0] - sigmas[0] *3*factor, means[0] + sigmas[0] * 3* factor, gridSize),
              np.linspace(means[1] - sigmas[1] * factor, means[1] + sigmas[1] *factor, gridSize),
              np.linspace(means[2] - sigmas[2] * factor, means[2] + sigmas[2] * factor, gridSize),
              np.linspace(means[3] - sigmas[3] * factor, means[3] + sigmas[3] * factor, gridSize),
              np.linspace(means[4] - sigmas[4] * factor, means[4] + sigmas[4] * factor, gridSize),
              np.linspace(means[5] - sigmas[5] * factor, means[5] + sigmas[5] * factor, gridSize),
              np.linspace(means[6] - sigmas[6] * factor, means[6] + sigmas[6] * factor, gridSize),
              np.linspace(means[7] - sigmas[7] * factor, means[7] + sigmas[7] * factor, gridSize),
              np.linspace(means[8] - sigmas[8] * factor, means[8] + sigmas[8] * factor, gridSize),
              np.linspace(means[9] - sigmas[9] * factor, means[9] + sigmas[9] * factor, gridSize),
              np.linspace(means[10]- sigmas[10]* factor, means[10]+ sigmas[10] * factor, gridSize),
              np.linspace(means[11]- sigmas[11]*1.5* factor, means[11]+ sigmas[11] * factor, gridSize),
              np.linspace(0.140, 0.16, gridSize),
              np.linspace(0.11, 0.15, gridSize),
              np.linspace(33.9, 36, gridSize),
              np.linspace(5.5, 7, gridSize)]

# def MargPostSupp(Params):
#     list = []
#     list.append(Params[0] > 0)
#     list.append(univarGrid[1][-1] > Params[1] > univarGrid[1][0])
#     list.append(univarGrid[2][-1] > Params[2] > univarGrid[2][0])
#     list.append(univarGrid[3][-1] > Params[3] > univarGrid[3][0])
#     list.append(univarGrid[4][-1] > Params[4] > univarGrid[4][0])
#     list.append(univarGrid[5][-1] > Params[5] > univarGrid[5][0])
#     list.append(univarGrid[6][0] < Params[6] < univarGrid[6][-1])
#     list.append(univarGrid[7][-1] > Params[7] > univarGrid[7][0])
#     list.append(univarGrid[8][-1] > Params[8] > univarGrid[8][0])
#     list.append(univarGrid[9][0] < Params[9] < univarGrid[9][-1])
#     list.append(univarGrid[10][0] < Params[10] < univarGrid[10][-1])
#     list.append(univarGrid[11][-1] > Params[11] > univarGrid[11][0])
#     list.append(univarGrid[12][-1] > Params[12] > univarGrid[12][0])
#     list.append(univarGrid[13][-1] > Params[13] > univarGrid[13][0])
#     list.append(univarGrid[14][-1] >Params[14] > univarGrid[14][0])
#     list.append(univarGrid[15][-1] >Params[15] > univarGrid[15][0])
#     #list.append(1 > Params[16] > 0)
#
#     return all(list)
def MargPostSupp(Params):
    list = []
    list.append(Params[0] > 0)
    list.append(Params[1] > 0)
    list.append(Params[2] > 0)
    list.append(Params[3] > 0)
    list.append(Params[4] > 0)
    list.append(Params[5] > 0)
    list.append(Params[6] < 0)
    list.append(Params[7] > 0)
    list.append(Params[8] > 0)
    list.append(Params[9] < 0)
    list.append(Params[10] < 0)
    list.append(Params[11] > 0)
    list.append(1 >Params[12] > 0)
    list.append(1 >Params[13] > 0)
    list.append(Params[14] > 0)
    list.append(Params[15] > 0)
    #list.append(1 > Params[16] > 0)

    return all(list)
x0 = np.append(means, gamma0)
x0 = means
xp0 = 0.9999999 * x0
dim = len(x0)
burnIn = 10000
tWalkSampNum = 3000000
MargPost = pytwalk.pytwalk(n=dim, U=log_post, Supp=MargPostSupp)

#print(" Support of Starting points:" + str(MargPostSupp(x0)) + str(MargPostSupp(xp0)))
MargPost.Run(T=tWalkSampNum + burnIn, x0=x0, xp0=xp0)

SampParas = MargPost.Output

fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)#tight_layout = True,
#axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
for i in range(0,3):

    axs[i].hist(SampParas[burnIn:,i],bins=n_bins)
    axs[i].axvline(means[i], color = 'red')
    axs[i].set_yticklabels([])


axs[0].set_xlabel('$h_0$')
axs[1].set_xlabel('$h_1$')
axs[2].set_xlabel('$h_2$')

fig.savefig('TempPostHistSamp0.svg')



fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)#tight_layout = True,

for i in range(3,6):
    axs[i-3].hist(SampParas[burnIn:,i],bins=n_bins)
    axs[i-3].axvline(means[i], color='red')
    axs[i-3].set_yticklabels([])

axs[0].set_xlabel('$h_3$')
axs[1].set_xlabel('$h_4$')
axs[2].set_xlabel('$h_5$')

fig.savefig('TempPostHistSamp1.svg')



fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)

for i in range(6,9):
    axs[i-6].hist(SampParas[burnIn:,i],bins=n_bins)
    axs[i-6].axvline(means[i], color='red')
    axs[i-6].set_yticklabels([])
axs[0].set_xlabel('$a_0$')
axs[1].set_xlabel('$a_1$')
axs[2].set_xlabel('$a_2$')

fig.savefig('TempPostHistSamp2.svg')


fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
for i in range(9,12):

    axs[i-9].hist(SampParas[burnIn:,i],bins=n_bins)
    axs[i-9].axvline(means[i], color='red')
    axs[i-9].set_yticklabels([])

axs[0].set_xlabel('$a_3$')
axs[1].set_xlabel('$a_4$')
axs[2].set_xlabel('$b_0$')
#axs[3].set_xlabel('$\gamma$')

fig.savefig('TempPostHistSamp3.svg')

fig, axs = plt.subplots(5,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
for i in range(12,dim):
    axs[i-12].hist(SampParas[burnIn:, i], bins=n_bins)
    axs[i-12].axvline(means[i], color='red')
    axs[i-12].set_yticklabels([])
axs[4].plot(range(0,tWalkSampNum+1),SampParas[burnIn:, -1])
axs[0].set_xlabel('$b_1$')
axs[1].set_xlabel('$b_2$')
axs[2].set_xlabel('$h_0$')
axs[3].set_xlabel('$p_0$')
#axs[4].set_xlabel('$\gamma$')
fig.savefig('PressPostHistSamp4.svg')
plt.show()
print(np.mean(SampParas[burnIn:, -1]))
print('done')
##
TrueCol = [50/255,220/255, 0/255]
tests = 100

indcies = np.random.randint(low=burnIn, high=burnIn+tWalkSampNum, size=tests)

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
axs.plot( pressure_values,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0 ,linewidth = 1.5, markersize =7)
for r in range(0, tests):

    Sol = np.exp(pressFunc(height_values[:,0], *SampParas[indcies[r], 12:- 1]))
    axs.plot( Sol ,height_values, markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', markersize =2, linewidth =0.5)


axs.set_xlabel(r'pressure in hPa')

axs.set_ylabel(r'height in km')
plt.savefig('PressPostMeanSigm.svg')
plt.show()
fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
axs.plot( temp_values,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0 ,linewidth = 1.5, markersize =7)
for r in range(0, tests):

    Sol = temp_func(height_values[:,0], *SampParas[indcies[r], :12])
    axs.plot( Sol ,height_values , markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', markersize =2, linewidth =0.5)

axs.set_xlabel(r'temperature in K ')

axs.set_ylabel(r'height in km')
plt.savefig('TempPostMeanSigm.svg')
plt.show()