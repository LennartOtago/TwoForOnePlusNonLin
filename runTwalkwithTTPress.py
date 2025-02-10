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

import glob
ttDir = '/home/lennartgolks/PycharmProjects/TTDecomposition/'
dim = len(glob.glob(ttDir + 'ttTraincoreLogP*.txt'))
maxRank = 1
TTCore = [None] * dim
univarGrid = [None] * dim
for i in range(0, dim):
    filename = ttDir + 'ttTraincoreLogP' +str(i)+ '.txt'
    f = open(filename)
    header = f.readline()
    matSha =header[2:-1].split(',')
    TTCore[i] = np.loadtxt(filename).reshape( (int(matSha[0]),int(matSha[1]),int(matSha[2])), order = 'F' )

    if int(matSha[2]) > maxRank:
        maxRank = int(matSha[2])
    univarGrid[i] = np.loadtxt(ttDir + 'uniVarGridLopP' +str(i)+ '.txt')

dim = len(univarGrid)
gridSize = len(univarGrid[0])



def MargPostSupp(Params):
    list = []
    list.append(univarGrid[0][-1] > Params[0] > univarGrid[0][0])
    list.append(univarGrid[1][-1] >Params[1] > univarGrid[1][0])
    list.append(univarGrid[2][-1] >Params[2] > univarGrid[2][0])  # 6.5)
    list.append(univarGrid[3][-1] >Params[3] > univarGrid[3][0])  # 5.5)
    list.append(univarGrid[4][-1] > Params[4] > univarGrid[4][0])  # 5.5)
    # list.append(Params[0] > Params[1])
    return all(list)


def LogPostfromTT(Params, TTCore, univarGrid):
    r_k, d, r_kpls1 = TTCore[0].shape
    PDFApprox = np.zeros((1,r_kpls1))
    #find indices
    ind = np.argmin(abs(univarGrid[0] - Params[0]))
    if Params[0] < univarGrid[0][ind]:
        ind -= 1
    PDFApprox = ((Params[0] - univarGrid[0][ind]) / (univarGrid[0][ind + 1] - univarGrid[0][ind])) * TTCore[0][:,
                                                                                                     ind + 1, :] + (
                        (univarGrid[0][ind + 1] - Params[0]) / (
                        univarGrid[0][ind + 1] - univarGrid[0][ind])) * TTCore[0][:, ind, :]

    currPDFAprrox = np.copy(PDFApprox)
    for i in range(1,len(Params)):
        ind =np.argmin(abs(univarGrid[i] - Params[i]))
        #print(ind)
        #print(i)
        if Params[i] < univarGrid[i][ind] or ind+1 == len(univarGrid[i]):
            ind -= 1
        PDFApprox = ((Params[i] - univarGrid[i][ind]) / (univarGrid[i][ind + 1] - univarGrid[i][ind])) * TTCore[i][:, ind + 1, :] + (
                                             (univarGrid[i][ind + 1] - Params[i]) / (
                                             univarGrid[i][ind + 1] - univarGrid[i][ind])) * TTCore[i][:, ind, :]
        currPDFAprrox = currPDFAprrox @ PDFApprox

    return currPDFAprrox


TTlogPost = lambda Params: LogPostfromTT(Params, TTCore, univarGrid)
tWalkSampNum = 300000
MargPost = pytwalk.pytwalk(n=dim, U=TTlogPost, Supp=MargPostSupp)
startTime = time.time()
# x0 = popt * 1.1
x0 = np.append(popt,gamma0)
xp0 = 1.01 * x0
TTlogPost(x0)
# print(" Support of Starting points:" + str(MargPostSupp(x0)) + str(MargPostSupp(xp0)))
MargPost.Run(T=tWalkSampNum + burnIn, x0=x0, xp0=xp0)
elapsedtWalkTime = time.time() - startTime
print('Elapsed Time for t-walk: ' + str(elapsedtWalkTime))
# MargPost.Ana()
# MargPost.SavetwalkOutput("MargPostDat.txt")
SampParas = MargPost.Output

plotDim = 5
fig, axs = plt.subplots(plotDim,1, tight_layout = True)
for i in range(0,plotDim):
    axs[i].hist(SampParas[:,i],bins=n_bins)
#axs[4].hist(SampParas[:,4],bins=n_bins)
axs[0].set_xlabel('$b_1$')
axs[1].set_xlabel('$b_2$')
axs[2].set_xlabel('$h_0$')
axs[3].set_xlabel('$p_0$')
axs[4].set_xlabel('$\gamma$')
#fig.savefig('pressHistRes.svg')
plt.show()
print('done')
