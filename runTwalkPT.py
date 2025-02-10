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
#dir = '/Users/lennart/PycharmProjects/firstModelCheckPhD/'
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
y = np.loadtxt(dir + 'dataY.txt')
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
ATy = np.matmul(A.T, y)
ATA = np.matmul(A.T,A)
Ax =np.matmul(A, VMR_O3 * theta_scale_O3)

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin)
ax1.scatter(y, tang_heights_lin)
ax1.plot(y, tang_heights_lin)
#plt.show()

def MinLogMargPostFirst(params):#, coeff):
    tol = 1e-8
    # gamma = params[0]
    # delta = params[1]
    gam = params[0]
    lamb = params[1]
    if lamb < 0  or gam < 0:
        return np.nan

    #ATA = np.matmul(A.T,A)
    Bp = ATA + lamb * L

    #y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
    #ATy = np.matmul(A.T, y)
    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[:,0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

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
    PT = pressFunc(height_values[:, 0], *paramP).reshape((n, 1)) /temp_func(height_values, *paramT).reshape((n, 1))
    #postDat = + SpecNumMeas / 2  * np.log(gam) - 0.5 * gam * np.sum((y - A @ PT ) ** 2)
    postDat = + SpecNumMeas / 2 * np.log(gam) - 0.5 * gam * np.sum((y - A @ PT) ** 2)

    #postDat = 0
    Values =     - ((h0 - h0Mean) / h0Sigm) ** 2 - ((h1 - h1Mean) / h1Sigm) ** 2 - (
                (h2 - h2Mean) / h2Sigm) ** 2 - (
                        (h3 - h3Mean) / h3Sigm) ** 2 - ((h4 - h4Mean) / h4Sigm) ** 2 - (
                        (h5 - h5Mean) / h5Sigm) ** 2 - ((a0 - a0Mean) / a0Sigm) ** 2 - (
                        (a1 - a1Mean) / a1Sigm) ** 2 - ((a2 - a2Mean) / a2Sigm) ** 2 \
                - ((a3 - a3Mean) / a3Sigm) ** 2 - ((a4 - a4Mean) / a4Sigm) ** 2 - ((b0 - b0Mean) / b0Sigm) ** 2 \
                - ((popt[0] - b1) / sigmaGrad1) ** 2 - ((popt[1] - b2) / sigmaGrad2) ** 2 - (
                            (popt[3] - p0) / sigmaP) ** 2 \
                - ((popt[2] - h0P) / sigmaH) ** 2 - betaG * gam

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

sigmas[0] = 0.5
sigmas[1] = 3
sigmas[2] = 1
sigmas[3] = 2
sigmas[4] = 2
sigmas[5] = 2
sigmas[6] = 0.01
sigmas[7] = 0.01
sigmas[8] = 0.1
sigmas[9] = 0.01
sigmas[10] = 0.01
sigmas[11] = 2

sigmaP = 0.25
sigmaH = 0.5
sigmaGrad1 = 0.005
sigmaGrad2 = 0.01

sigmas[12] = sigmaGrad1
sigmas[13] = sigmaGrad2
sigmas[14] = sigmaH
sigmas[15] = sigmaP


log_post = lambda params: -log_postTP(params, means, sigmas, popt, A, y, height_values, gamma0)
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
xp0 = 1.01 * x0
dim = len(x0)
tWalkSampNum = 100000
MargPost = pytwalk.pytwalk(n=dim, U=log_post, Supp=MargPostSupp)

#print(" Support of Starting points:" + str(MargPostSupp(x0)) + str(MargPostSupp(xp0)))
MargPost.Run(T=tWalkSampNum + burnIn, x0=x0, xp0=xp0)

SampParas = MargPost.Output

fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)#tight_layout = True,
#axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
for i in range(0,3):

    axs[i].hist(SampParas[:,i],bins=n_bins)
    axs[i].axvline(means[i], color = 'red')
    axs[i].set_yticklabels([])


axs[0].set_xlabel('$h_0$')
axs[1].set_xlabel('$h_1$')
axs[2].set_xlabel('$h_2$')

#fig.savefig('TempPostHistSamp0.svg')



fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)#tight_layout = True,

for i in range(3,6):
    axs[i-3].hist(SampParas[:,i],bins=n_bins)
    axs[i-3].axvline(means[i], color='red')
    axs[i-3].set_yticklabels([])

axs[0].set_xlabel('$h_3$')
axs[1].set_xlabel('$h_4$')
axs[2].set_xlabel('$h_5$')

#fig.savefig('TempPostHistSamp1.svg')



fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)

for i in range(6,9):
    axs[i-6].hist(SampParas[:,i],bins=n_bins)
    axs[i-6].axvline(means[i], color='red')
    axs[i-6].set_yticklabels([])
axs[0].set_xlabel('$a_0$')
axs[1].set_xlabel('$a_1$')
axs[2].set_xlabel('$a_2$')

#fig.savefig('TempPostHistSamp2.svg')


fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
for i in range(9,12):

    axs[i-9].hist(SampParas[:,i],bins=n_bins)
    axs[i-9].axvline(means[i], color='red')
    axs[i-9].set_yticklabels([])

axs[0].set_xlabel('$a_3$')
axs[1].set_xlabel('$a_4$')
axs[2].set_xlabel('$b_0$')
#axs[3].set_xlabel('$\gamma$')

#fig.savefig('TempPostHistSamp3.svg')

fig, axs = plt.subplots(5,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
for i in range(12,dim):
    axs[i-12].hist(SampParas[:, i], bins=n_bins)
    axs[i-12].axvline(means[i], color='red')
    axs[i-12].set_yticklabels([])

axs[0].set_xlabel('$b_1$')
axs[1].set_xlabel('$b_2$')
axs[2].set_xlabel('$h_0$')
axs[3].set_xlabel('$p_0$')
axs[4].set_xlabel('$\gamma$')
plt.show()

print('done')
