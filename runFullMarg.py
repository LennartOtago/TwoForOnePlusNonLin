import numpy as np
import scipy as scy
import time, pytwalk
import os
import matplotlib.pyplot as plt
from pathlib import Path
cwd = os.getcwd()
path = Path(cwd)
parentDir = str( path.parent.absolute())

def g(A, L, l):
    """ calculate g"""
    B = np.matmul(A.T, A) + l * L
    # Bu, Bs, Bvh = np.linalg.svd(B)
    upL = scy.linalg.cholesky(B)
    # return np.sum(np.log(Bs))
    return 2 * np.sum(np.log(np.diag(upL)))

def f(ATy, y, B_inv_A_trans_y):
    return np.matmul(y[:, 0].T, y[:, 0]) - np.matmul(ATy[:, 0].T, B_inv_A_trans_y)



def temp_func(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,a5,a6,b0):
    a = np.ones(x.shape)
    b = np.ones(x.shape)

    a[x < h0] = a0
    a[h0 <= x] = a1
    a[h1 <= x] = a2
    a[h2 <= x] = a3
    a[h3 <= x] = a4
    a[h4 <= x ] = a5
    a[h5 <= x ] = a6
    #a[h6 <= x ] = 0

    b[x < h0] = b0
    b[h0 <= x] = b0 + h0 * a0
    b[h1 <= x] = b0 + (h1 - h0) * a1 + h0 * a0
    b[h2 <= x] = a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h3 <= x ] = a3 * (h3-h2) + a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h4 <= x ] = a4 * (h4 -h3) + a3 * (h3-h2) + a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h5 <= x ] = a5 * (h5 -h4) + a4 * (h4 -h3) + a3 * (h3-h2) + a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    #b[h6 <= x ] = a4 * (h6-h5) + a3 * (h5-h4) + a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0


    h = np.ones(x.shape)
    h[x < h0] = 0
    h[h0 <= x] = h0
    h[h1 <= x] = h1
    h[h2 <= x] = h2
    h[h3 <= x] = h3
    h[h4 <= x] = h4
    h[h5 <= x] = h5
    #h[h6 <= x] = h6
    return a * (x - h) + b

def pressFunc(x, b, p0):
    return np.exp(-b * x  + np.log(p0))

def forward_substitution(L, b):
    # Get number of rows
    n = L.shape[0]

    # Allocating space for the solution vector
    y = np.zeros_like(b, dtype=np.double);

    # Here we perform the forward-substitution.
    # Initializing  with the first row.
    y[0] = b[0] / L[0, 0]

    # Looping over rows in reverse (from the bottom  up),
    # starting with the second to last row, because  the
    # last row solve was completed in the last step.
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y

def back_substitution(U, y):
    # Number of rows
    n = U.shape[0]

    # Allocating space for the solution vector
    x = np.zeros_like(y, dtype=np.double)

    # Here we perform the back-substitution.
    # Initializing with the last row.
    x[-1] = y[-1] / U[-1, -1]

    # Looping over rows in reverse (from the bottom up),
    # starting with the second to last row, because the
    # last row solve was completed in the last step.
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]

    return x


def lu_solve(L, U, b):

    y = forward_substitution(L, b)

    return back_substitution(U, y)



def FullMarg(params, means, sigmas, A, L, y, height_values):
    n = len(height_values)
    m = len(y)

    h1Mean = means[6]
    h1Sigm = sigmas[6]

    h2Mean = means[4]
    h2Sigm = sigmas[4]

    h3Mean = means[10]
    h3Sigm = sigmas[10]

    h4Mean = means[12]
    h4Sigm = sigmas[12]

    h5Mean = means[14]
    h5Sigm = sigmas[14]

    a0Mean = means[7]
    a0Sigm = sigmas[7]

    a1Mean = means[5]
    a1Sigm = sigmas[5]

    a2Mean = means[8]
    a2Sigm = sigmas[8]

    a3Mean = means[9]
    a3Sigm = sigmas[9]

    a4Mean = means[11]
    a4Sigm = sigmas[11]

    a5Mean = means[13]
    a5Sigm = sigmas[13]

    a6Mean = means[15]
    a6Sigm = sigmas[15]

    b0Mean = means[1]
    b0Sigm = sigmas[1]

    h0Mean = means[2]
    h0Sigm = sigmas[2]

    sigmaGrad1 = sigmas[3]
    bmean = means[3]

    sigmaP = sigmas[0]
    pmean = means[0]
    betaD = 1e-35
    betaG = 1e-35
    gam0 = 2e15
    gamSig = 1e15
    lamb0 = 2000
    lambSig = 1000


    lamb = params[0]
    gam = params[1]
    h1 = params[8]
    h2 = params[6]
    h3 = params[12]
    h4 = params[14]
    h5 = params[16]
    a0 = params[9]
    a1 = params[7]
    a2 = params[10]
    a3 = params[11]
    a4 = params[13]
    a5 = params[15]
    a6 = params[17]
    b0 = params[3]
    h0 = params[4]
    b = params[5]
    p0 = params[2]
    paramT = [h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, a5, a6, b0]
    paramP = [b, p0]
    PT = pressFunc(height_values[:, 0], *paramP).reshape((n, 1)) / temp_func(height_values, *paramT).reshape((n, 1))

    CurrA = A * PT.T
    G = g(CurrA, L, lamb)
    ATy = CurrA.T @ y
    Bp = CurrA.T @ CurrA + lamb * L
    LowTri = np.linalg.cholesky(Bp)
    UpTri = LowTri.T
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[:,0])
    F = f(ATy, y, B_inv_A_trans_y)
    priors = ( - ((h0 - h0Mean) / h0Sigm) ** 2 -  ((h1 - h1Mean) / h1Sigm) ** 2 - ((h2 - h2Mean) / h2Sigm) ** 2 - (
                   (h3 - h3Mean) / h3Sigm) ** 2 -  ((h4 - h4Mean) / h4Sigm) ** 2
                   - ((h5 - h5Mean) / h5Sigm) ** 2  - ((a0 - a0Mean) / a0Sigm) ** 2
                   - ((a1 - a1Mean) / a1Sigm) ** 2 - ((a2 - a2Mean) / a2Sigm) ** 2
                   - ((a3 - a3Mean) / a3Sigm) ** 2 - ((a4 - a4Mean) / a4Sigm) ** 2
                  - ((a6 - a6Mean) / a6Sigm) ** 2 - ((a5 - a5Mean) / a5Sigm) ** 2
                  - ((b0 - b0Mean) / b0Sigm) ** 2
                   - ((pmean - p0) / sigmaP) ** 2 - ((bmean - b) / sigmaGrad1) ** 2) #- 2 * gam[j] * 1e-10
    gamLamPrior = n/2 * np.log(lamb) + (m/2 + 1) * np.log(gam) -  ( betaD *  lamb * gam + betaG *gam)
    #gamLamPrior =  n/2 * np.log(lamb) + m/2 * np.log(gam)  - 0.5* ((gam - gam0) / gamSig) ** 2 - 0.5* ((lamb - lamb0) / lambSig) ** 2
    PrevMarg =   - 0.5 * G - 0.5 * gam * F

    return  PrevMarg + 0.5 * priors + gamLamPrior


dir = parentDir + '/TTDecomposition/'

Aplain = np.loadtxt(dir + 'APlainMat.txt')
means = np.loadtxt(dir + 'PTMeans.txt')
sigmas = np.loadtxt(dir + 'PTSigmas.txt')

L = np.loadtxt(dir + 'GraphLaplacian.txt')
y = np.loadtxt(dir + 'nonLinDataY.txt')
y = y.reshape((len(y),1))
height_values = np.loadtxt(dir + 'height_values.txt')
height_values = height_values.reshape((len(height_values),1))

import glob

dim = len(glob.glob(dir + 'uniVarGridFull*.txt'))

univarGrid = [None] * dim
for i in range(0, dim):
    univarGrid[i] = np.loadtxt(dir+'uniVarGridFull' +str(i)+ '.txt')
gridSize = 40  # 150#15
factor =4 # 1.5
univarGrid = [np.linspace(1, 7000, gridSize),
                np.linspace(0.1e15, 6e15, gridSize),
                np.linspace(means[0] - sigmas[0] * factor, means[0] + sigmas[0] * factor, gridSize),
                np.linspace(means[1] - sigmas[1] *factor, means[1] + sigmas[1]*factor, gridSize),
                np.linspace(means[2] - sigmas[2] * factor, means[2] + sigmas[2]* factor, gridSize),
                np.linspace(means[3] - sigmas[3] * factor, means[3] + sigmas[3] * factor, gridSize),
                np.linspace(means[4] - sigmas[4]  * factor, means[4] + sigmas[4] * factor, gridSize),
                np.linspace(means[5] - sigmas[5] * factor, means[5] + sigmas[5] * factor, gridSize),
                np.linspace(means[6] - sigmas[6] * factor, means[6] + sigmas[6] *factor, gridSize),
                np.linspace(means[7] - sigmas[7] * factor, means[7] + sigmas[7] * factor, gridSize),
                np.linspace(means[8] - sigmas[8] * factor, means[8] + sigmas[8] * factor, gridSize),
                np.linspace(means[9] - sigmas[9] * factor, means[9] + sigmas[9] * factor, gridSize),
                np.linspace(means[10] - sigmas[10] * factor, means[10] + sigmas[10] * factor, gridSize),
                np.linspace(means[11] - sigmas[11] * factor, means[11] + sigmas[11] * factor, gridSize),
                np.linspace(means[12] - sigmas[12] * factor, means[12] + sigmas[12] * factor, gridSize),
                np.linspace(means[13]- sigmas[13]*factor, means[13]+ sigmas[13] *factor, gridSize),
                np.linspace(means[14] - sigmas[14] * factor, means[14] + sigmas[14] * factor, gridSize),
                np.linspace(means[15]- sigmas[15]  * factor, means[15]+ sigmas[15] * factor, gridSize)]



def MargPostSupp(Params):
    list = []
    #list.append( Params[5] >Params[4] >Params[3] >Params[2] >Params[1] >Params[0] )
    list.append(univarGrid[0][0] < Params[0] < univarGrid[0][-1])
    list.append(univarGrid[1][0] < Params[1] < univarGrid[1][-1])
    list.append(univarGrid[2][0] < Params[2] < univarGrid[2][-1])
    list.append(univarGrid[3][0] < Params[3] < univarGrid[3][-1])
    list.append(univarGrid[4][0] < Params[4] < univarGrid[4][-1])
    list.append(univarGrid[5][0] < Params[5] < univarGrid[5][-1])
    #list.append(univarGrid[6][-1] > Params[6] > univarGrid[6][0])
    list.append(univarGrid[6][0] < Params[6] < univarGrid[6][-1])
    list.append(univarGrid[7][0] < Params[7] < univarGrid[7][-1])
    list.append(univarGrid[8][0] < Params[8] < univarGrid[8][-1])
    list.append(univarGrid[9][0] < Params[9] < univarGrid[9][-1])
    list.append(univarGrid[10][0] < Params[10] < univarGrid[10][-1])
    list.append(univarGrid[11][0] < Params[11] < univarGrid[11][-1])
    list.append(univarGrid[12][0] < Params[12] < univarGrid[12][-1])
    #list.append(means[12]-4*sigmas[12] < Params[12] < means[12]+4*sigmas[12])
    list.append(univarGrid[13][0] < Params[13] < univarGrid[13][-1])
    list.append(univarGrid[14][0] < Params[14] < univarGrid[14][-1])
    #list.append(means[15] - 4 * sigmas[15] < Params[15] < means[15] + 4 * sigmas[15])
    list.append(univarGrid[15][0] < Params[15] < univarGrid[15][-1])
    list.append(univarGrid[16][0] < Params[16] < univarGrid[16][-1])
    list.append(univarGrid[17][0] < Params[17] < univarGrid[17][-1])
    return all(list)
#
# def MargPostSupp(Params):
#     list = []
#     return all(list)

##
# burnIn = 20
# SampParas = np.loadtxt('SampParasFull.txt')
#
# fig, axs = plt.subplots(4,1, tight_layout = True)#tight_layout = True,
# #axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# for i in range(0,4):
#
#     axs[i].hist(SampParas[burnIn:,i])
#     axs[i].set_yticklabels([])
#
# fig, axs = plt.subplots(4, 1, tight_layout=True)  # tight_layout = True,
# # axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# for i in range(4, 8):
#     axs[i-4].hist(SampParas[burnIn:, i])
#     axs[i-4].set_yticklabels([])
#
#
# fig, axs = plt.subplots(4,1, tight_layout = True)#tight_layout = True,
# #axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# for i in range(8,12):
#
#     axs[i-8].hist(SampParas[burnIn:,i])
#     axs[i-8].set_yticklabels([])
#
# fig, axs = plt.subplots(5,1, tight_layout = True)#tight_layout = True,
# #axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# for i in range(12,17):
#
#     axs[i-12].hist(SampParas[burnIn:,i])
#     axs[i-12].set_yticklabels([])
#
#
#
# plt.show(block= True)





##

gamLam0 = [2500 ,2e15]
x0 = np.append(gamLam0 , means)
gamLam0 = [2000 ,2.5e15]
xp0 =  np.append(gamLam0,means)+ 1e-3
dim = len(x0)
burnIn = 2000
tWalkSampNum = 5000000
log_post = lambda params: -FullMarg(params, means, sigmas, Aplain, L, y, height_values)

MargPost = pytwalk.pytwalk(n=dim, U=log_post, Supp=MargPostSupp)
print(" Support of Starting points:" + str(MargPostSupp(x0)) + str(MargPostSupp(xp0)))
##
#print(" Support of Starting points:" + str(MargPostSupp(x0)) + str(MargPostSupp(xp0)))
startTime = time.time()
MargPost.Run(T=tWalkSampNum + burnIn, x0=x0, xp0=xp0)
elapsTime = time.time() - startTime
print(f'elapsed time : {elapsTime/60}')

SampParas = MargPost.Output
np.savetxt('SampParasFull.txt',SampParas,  fmt = '%.30f')

fig, axs = plt.subplots(4,1, tight_layout = True)#tight_layout = True,
#axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
for i in range(0,4):

    axs[i].hist(SampParas[burnIn:,i])
    axs[i].set_yticklabels([])

fig, axs = plt.subplots(4, 1, tight_layout=True)  # tight_layout = True,
# axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
for i in range(4, 8):
    axs[i-4].hist(SampParas[burnIn:, i])
    axs[i-4].set_yticklabels([])


fig, axs = plt.subplots(4,1, tight_layout = True)#tight_layout = True,
#axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
for i in range(8,12):

    axs[i-8].hist(SampParas[burnIn:,i])
    axs[i-8].set_yticklabels([])

fig, axs = plt.subplots(5,1, tight_layout = True)#tight_layout = True,
#axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
for i in range(12,17):

    axs[i-12].hist(SampParas[burnIn:,i])
    axs[i-12].set_yticklabels([])



plt.show(block= True)
