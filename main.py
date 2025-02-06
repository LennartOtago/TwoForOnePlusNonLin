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

betaG = 1e-4# 1e-18#
betaD = 1e3#9e3#1e-3#1e-10#1e-22#  # 1e-4

import numpy as np


dir = '/home/lennartgolks/PycharmProjects/firstModelCheckPhD/'
dir = '/Users/lennart/PycharmProjects/firstModelCheckPhD/'
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

ATy = np.matmul(A.T, y)
ATA = np.matmul(A.T,A)
Ax =np.matmul(A, VMR_O3 * theta_scale_O3)
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin)
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
##
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(temp_values, height_values)
# plt.show()
#
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(Ax, tang_heights_lin)
# ax1.scatter(y, tang_heights_lin)
# ax1.plot(y, tang_heights_lin)
# plt.show()
#print(1/np.var(y))


##
"""update A so that O3 profile is constant"""
#O3_Prof = np.mean(VMR_O3) * np.ones(SpecNumLayers)

# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# #ax1.plot(O3_Prof, height_values, linewidth = 2.5, label = 'my guess', marker = 'o')
# ax1.plot(VMR_O3, height_values, linewidth = 2.5, label = 'true profile', marker = 'o')
# #ax1.plot(O3, heights, linewidth = 2.5, label = 'true profile', marker = 'o')
#
# ax1.set_ylabel('Height in km')
# ax1.set_xlabel('Volume Mixing Ratio of Ozone')
# # ax2 = ax1.twiny()
# # ax2.scatter(y, tang_heights_lin ,linewidth = 2, marker =  'x', label = 'data' , color = 'k')
# # ax2.set_xlabel(r'Spectral radiance in $\frac{W cm}{m^2  sr} $',labelpad=10)# color =dataCol,
#
# ax1.legend()
# # plt.savefig('DataStartTrueProfile.png')
# plt.show()



##

def pressFunc(x, b1, b2, h0, p0):
    b = np.ones(len(x))
    b[x>h0] = b2
    b[x<=h0] = b1
    return -b * (x - h0) + np.log(p0)

popt, pcov = scy.optimize.curve_fit(pressFunc, height_values, np.log(pressure_values), p0=[-2e-2,-2e-2, 18, 15])


# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(pressure_values,height_values, linewidth = 2)
# #ax1.plot(np.exp(pressFunc(height_values[:,0], -0.12,-0.2)), height_values, linewidth = 2)
# ax1.plot(np.exp(pressFunc(height_values[:,0], *popt)), height_values[:,0], linewidth = 2)
# ax1.axhline(y=popt[2])
# ax1.axvline(x=popt[3])
# ax1.set_xlabel(r'Pressure in hPa ')
# ax1.set_ylabel('Height in km')
# #ax1.set_xscale('log')
# plt.savefig('samplesPressure.png')
# plt.show()

##
''' t-walk for temperature start'''


h0 = 11
h1 = 20
h2 = 32
h3 = 47
h4 = 51
h5 = 71


a0 = -6.5
a1 = 1
a2 = 2.8
a3 = -2.8
a4 = -2
b0 = 288.15
# #b1 = 288.15 + h0 * a0
# #
# fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
# ax1.plot(temp_values, height_values, linewidth=5, label='true T', color='green', zorder=0)
# ax1.plot(temp_func(height_values,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,b0), height_values, linewidth=2, label='reconst', color='red', zorder=1)
# #plt.savefig('TemperatureSamp.png')
# plt.show()
# #
# #
# A, theta_scale_T = composeAforTemp(A_lin, pressure_values, VMR_O3, ind, temp_values)
#
# def log_post_temp(Params):
#
#     n = SpecNumLayers
#     m = SpecNumMeas
#     h0 = Params[0]
#     h1 = Params[1]
#     h2 = Params[2]
#     h3 = Params[3]
#     h4 = Params[4]
#     h5 = Params[5]
#     a0 = Params[6]
#     a1 = Params[7]
#     a2 = Params[8]
#     a3 = Params[9]
#     a4 = Params[10]
#     b0 = Params[11]
#
#     h0Mean = 11
#     h0Sigm = 0.5
#
#     h1Mean = 20
#     h1Sigm = 3
#
#     h2Mean = 32
#     h2Sigm = 1
#
#     h3Mean = 47
#     h3Sigm = 2
#
#     h4Mean = 51
#     h4Sigm = 2
#
#     h5Mean = 71
#     h5Sigm = 2
#
#     a0Mean = -6.5
#     a0Sigm = 0.01
#
#     a1Mean = 1
#     a1Sigm = 0.01
#
#     a2Mean = 2.8
#     a2Sigm = 0.1
#
#     a3Mean = -2.8
#     a3Sigm = 0.01
#
#     a4Mean = -2
#     a4Sigm = 0.01
#
#     b0Mean = 288.15
#     b0Sigm = 2
#
#
#     return gamma * np.sum((y - A @ (1/temp_func(height_values,*Params).reshape((n,1)))) ** 2) + ((h0-h0Mean)/h0Sigm)**2 + ((h1-h1Mean)/h1Sigm)**2 + ((h2-h2Mean)/h2Sigm)**2 +  ((h3-h3Mean)/h3Sigm)**2+  ((h4-h4Mean)/h4Sigm)**2+ ((h5-h5Mean)/h5Sigm)**2+  ((a0-a0Mean)/a0Sigm)**2+  ((a1-a1Mean)/a1Sigm)**2+ ((a2-a2Mean)/a2Sigm)**2\
#         + ((a3-a3Mean)/a3Sigm)**2 + ((a4-a4Mean)/a4Sigm)**2+ ((b0-b0Mean)/b0Sigm)**2
#
#
#
# def MargPostSupp_temp(Params):
#     list = []
#     return all(list)


# MargPost = pytwalk.pytwalk(n=10, U=log_post_temp, Supp=MargPostSupp_temp)
# x0 = np.array([h0,h1,h2,h3,h4,a0,a1,a2,a3,b0])
# xp0 = 1.01 * x0
# TempBurnIn = 5000
# TempWalkSampNum = 100000
# MargPost.Run(T=TempWalkSampNum + TempBurnIn, x0=x0, xp0=xp0)
##
# TempSamps = MargPost.Output
# paraSamp = 100#
# TempResults = np.zeros((paraSamp,n))
# randInd = np.random.randint(low = burnIn, high= burnIn+TempWalkSampNum, size = paraSamp)
#
# fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
# ax1.plot(temp_values, height_values, linewidth=5, label='true T', color='green', zorder=0)
#
# for p in range(0,paraSamp):
#     TempResults[p] = temp_func(height_values[:,0], *TempSamps[randInd[p],0:-1])
#     ax1.plot(TempResults[p], height_values[:,0], linewidth=0.2, label='reconst', zorder=1)
#
# temp_Prof = np.mean(TempResults,0)
# ax1.plot(temp_Prof, height_values, marker='>', color="k", label='sample mean', zorder=2, linewidth=0.5,markersize=5)
# plt.show()

''' t-walk for temperature end'''
## accept data set

def gamDist(x, mean, sigma):
    return (2* np.pi * sigma**2)**(-0.5) * np.exp(-0.5 * (x - mean) ** 2 / sigma** 2)

def twoNormDist(X, Y, meanX, sigX, meanY, sigY):
    Mat = np.zeros((len(Y),len(X)))
    for i in range(0,len(X)):
        for j in range(0,len(Y)):
             Mat[j,i] = 1/(2 * np.pi * sigX * sigY) * np.exp(
                -0.5 * (X[i] - meanX) ** 2 / sigX ** 2) * np.exp(-0.5 * (Y[j] - meanY) ** 2 / sigY ** 2)
    return Mat


def SingtwoNormDist(X, Y, meanX, sigX, meanY, sigY):
    return 1/(2 * np.pi * sigX * sigY) * np.exp(
                -0.5 * (X - meanX) ** 2 / sigX ** 2) * np.exp(-0.5 * (Y - meanY) ** 2 / sigY ** 2)

Y = np.linspace(gammaMin0*0.5, gammaMin0*1.5,100)

X = np.linspace(1e-4, 3e-4)

Z = twoNormDist(X, Y, 1.9e-4, 2e-5, gammaMin0, gammaMin0 * 0.05)/ np.sum(twoNormDist(X, Y, 1.9e-4, 2e-5, gammaMin0, gammaMin0 * 0.05))
# fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
# plt.pcolormesh(X,Y,Z)
# #plt.imshow(Mat, cmap=mpl.cm.hot)
# plt.colorbar()
# plt.show()
# x = np.linspace(0, 3e-6)
# fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
# #ax1.plot(x,gamDist(x, gamma, gamma*0.2)/np.sum(gamDist(x, gamma, gamma*0.2)), linewidth=5, color='green', zorder=0)
# #ax1.plot(xdel,gamDist(xdel, 1.8e-4,2.5e-6)/np.sum(gamDist(xdel, 1.8e-4,2.5e-6)), linewidth=5, color='green', zorder=0)
# ax1.plot(x,gamDist(x, 9e-7,3.5e-7), linewidth=5, color='green', zorder=0)
#
# plt.show()
##
'''do the sampling'''
def pressFunc(x, b1, b2, h0, p0):
    b = np.ones(len(x))
    b[x>h0] = b2
    b[x<=h0] = b1
    return np.exp(-b * (x - h0) + np.log(p0))

def Parabel(x, h0, a0, d0):

    return a0 * np.power((h0-x),2 )+ d0

##
# tests = 30
# for t in range(0,tests):
#
#     A, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind)
#     Ax = np.matmul(A, VMR_O3 * theta_scale_O3)
#     y, gamma = add_noise(Ax, SNR)  # 90 works fine
#     y = y.reshape((m,1))
#     #y = np.loadtxt('dataYtest002.txt').reshape((SpecNumMeas, 1))
#     ATy = np.matmul(A.T, y)
#     ATA = np.matmul(A.T, A)
#     print(1/np.var(y[0:12]))

##

# dim = 3
# maxRank = 1
# univarGrid = [None] * dim
# TTCore = [None] * dim
# for dimCount in range(0, dim):
#     univarGrid[dimCount] = np.loadtxt('detLGrid' + str(dimCount) + '.txt')
#     filename = 'ttdetLCore' + str(dimCount) + '.txt'
#     file = open(filename)
#     header = file.readline()
#     matSha = header[2:-1].split(',')
#
#     TTCore[dimCount] = np.loadtxt(filename).reshape((int(matSha[0]), int(matSha[1]), int(matSha[2])), order='F')
#     if int(matSha[2]) > maxRank:
#         maxRank = int(matSha[2])

tests = 1
for t in range(0,tests):

    A, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind)
    Ax = np.matmul(A, VMR_O3 * theta_scale_O3)
    #y, gamma = add_noise(Ax, SNR)  # 90 works fine
    #y = y.reshape((m,1))
    #y = np.loadtxt('dataYtest003.txt').reshape((SpecNumMeas, 1))
    ATy = np.matmul(A.T, y)
    ATA = np.matmul(A.T, A)


    def MinLogMargPostFirst(params):  # , coeff):
        tol = 1e-8
        # gamma = params[0]
        # delta = params[1]
        gam = params[0]
        lamb = params[1]
        if lamb < 0 or gam < 0:
            return np.nan

        # ATA = np.matmul(A.T,A)
        Bp = ATA + lamb * L

        # y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
        # ATy = np.matmul(A.T, y)
        B_inv_A_trans_y, exitCode = gmres(Bp, ATy[:, 0], tol=tol, restart=25)
        if exitCode != 0:
            print(exitCode)

        G = g(A, L, lamb)
        F = f(ATy, y, B_inv_A_trans_y)

        return -n / 2 * np.log(lamb) - (m / 2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F + (
                    betaD * lamb * gam + betaG * gam)


    gammaMin0, lam0 = optimize.fmin(MinLogMargPostFirst, [gamma0, (np.var(VMR_O3) * theta_scale_O3) / gamma0])

    #np.savetxt('data/dataYtest' + str(t).zfill(3) + '.txt', y, header = 'Data y including noise', fmt = '%.15f')



    SampleRounds = 2
    round = 1

    print(np.mean(VMR_O3))

    number_samples =1500
    recov_temp_fit = temp_values#np.mean(temp_values) * np.ones((SpecNumLayers,1))
    recov_press = pressure_values#np.mean(pressure_values) * np.ones((SpecNumLayers,1))#1013 * np.exp(-np.mean(grad) * height_values[:,0])
    Results = np.zeros((SampleRounds, len(VMR_O3)))
    TempResults = np.zeros((SampleRounds, len(VMR_O3)))
    PressResults = np.zeros((SampleRounds, len(VMR_O3)))
    deltRes = np.zeros((SampleRounds,3))
    gamRes = np.zeros(SampleRounds)

    burnInDel = 100
    tWalkSampNumDel = 100000

    tWalkSampNum = 10000
    burnInT =100
    burnInMH =100

    deltRes[0,:] = np.array([ 30,1e-6, 7.5e-5])#lam0 * gamma0*0.4])
    gamRes[0] = gammaMin0
    SetDelta = Parabel(height_values,*deltRes[0,:])
    SetGamma =  gamRes[0]
    TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
    TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
    Diag = np.eye(n) * np.sum(TriU + TriL, 0)

    L_d = -TriU + Diag - TriL
    L_d[0, 0] = 2 * L_d[0, 0]
    L_d[-1, -1] = 2 * L_d[-1, -1]

    B0 = (ATA + 1 / gamRes[0] *  L_d)
    B_inv_A_trans_y0, exitCode = gmres(B0, ATy[:,0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    PressResults[0, :] = pressure_values
    TempResults[0,:] = temp_values.reshape(n)
    Results[0,:] = VMR_O3

    def MargPostSupp(Params):
        list = []
        list.append(1e-9 < Params[0] < 2.4e-8 )
        list.append(28 < Params[1] < 36)
        list.append(1e-8 < Params[2] < 5.5e-6)
        list.append(1e-7 < Params[3] < 1e-3)
        return all(list)


    def log_post(Params):
        tol = 1e-8
        n = SpecNumLayers
        m = SpecNumMeas

        gam = Params[0]
        h1 = Params[1]
        a0 = Params[2]
        d0 = Params[3]
        #detL = getDetL([h1,a0,d0], univarGrid, TTCore, maxRank)

        delta = Parabel(height_values,h1, a0, d0)
        TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * delta
        TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * delta.T
        Diag = np.eye(n) * np.sum(TriU + TriL, 0)

        L_d = -TriU + Diag - TriL
        L_d[0, 0] = 2 * L_d[0, 0]
        L_d[-1, -1] = 2 * L_d[-1, -1]

        try:
            upTriL = scy.linalg.cholesky(L_d)
            detL = 2 * np.sum(np.log(np.diag(upTriL)))
        except scy.linalg.LinAlgError:
            try:
                L_du, L_ds, L_dvh = np.linalg.svd(L_d)
                detL = np.sum(np.log(L_ds))
            except np.linalg.LinAlgError:
                print("SVD did not converge, use scipy.linalg.det()")
                detL = np.log(scy.linalg.det(L_d))

        Bp = ATA + 1/gam * L_d

        B_inv_A_trans_y, exitCode = gmres(Bp, ATy[:,0], x0 = B_inv_A_trans_y0, tol=tol, restart=25)
        if exitCode != 0:
            print(exitCode)

        G = g(A, L_d,  1/gam)
        F = f(ATy, y,  B_inv_A_trans_y)
        alphaD =  1
        alphaG = 1
        hMean = [[31.35]]#height_values[VMR_O3[:] == np.max(VMR_O3[:])]
        # hMean = 25

        d0Mean =0.8e-4
        betaG = 1e-10
        betaD = 1e-10
        aMean = 1.6e-6
        aStd = 0.5e-6
        # + betaG *gam
        # + 0.5 * ((gam - gamma0) / ( - (m / 2 - n / 2) * np.log(gam)gamma0 * 0.01)) ** 2
        return -3.5e2 - 0.5 * detL + 0.5 * G + 0.5 * gam * F  + 0.5 * ((Params[1] - hMean) / 1) ** 2+ 0.5 * ((a0 - aMean) / (aStd)) ** 2 + betaG *gam + betaD * d0


    startTime = time.time()

    # A, theta_scale_O3 = composeAforO3(A_lin, TempResults[round - 1, :].reshape((n, 1)), PressResults[round - 1, :], ind)
    # ATy = np.matmul(A.T, y)
    # ATA = np.matmul(A.T, A)
    # dim = 4
    # MargPost = pytwalk.pytwalk(n=dim, U=log_post, Supp=MargPostSupp)
    # x0 = np.array([SetGamma, *deltRes[round - 1, :]])
    # xp0 = 1.0001 * x0
    #
    # MargPost.Run(T=tWalkSampNumDel + burnInDel, x0=x0, xp0=xp0)
    #
    # Samps = MargPost.Output

    # A, theta_scale = composeAforPress(A_lin, TempResults[round - 1, :].reshape((n, 1)), VMR_O3, ind)
    # SampParas = tWalkPress(height_values, A, y, popt, tWalkSampNum, burnInT, SetGamma)

    # mean, delta, tint, d_tint = tauint(Samps[1+burnInDel:,:-1].reshape((dim, 1, tWalkSampNumDel)), 0)
    # print(2 * tint)

    # A, theta_scale_T = composeAforTemp(A_lin, PressResults[round - 1, :], Results[round-1, :], ind, temp_values)
    #
    # TempBurnIn = 2500
    # TempWalkSampNum = 500000
    # TempSamps = tWalkTemp(height_values, A, y, TempWalkSampNum, TempBurnIn, SetGamma, SpecNumLayers, h0, h1, h2, h3, h4, h5, a0, a1, a2, a3,a4, b0)

    for round in range(1,SampleRounds):

        # A, theta_scale_O3 = composeAforO3(A_lin, TempResults[round-1, :].reshape((n, 1)), PressResults[round-1, :], ind)
        # ATy = np.matmul(A.T, y)
        # ATA = np.matmul(A.T, A)
        #
        # MargPost = pytwalk.pytwalk(n=4, U=log_post, Supp=MargPostSupp)
        # x0 = np.array([SetGamma, *deltRes[round-1, :]])
        # xp0 = 1.0001 * x0
        #
        # MargPost.Run(T=tWalkSampNumDel + burnInDel, x0=x0, xp0=xp0)
        #
        # Samps = MargPost.Output

        # MWGRand = burnIn + np.random.randint(low=0, high=tWalkSampNumDel)
        # SetGamma = Samps[MWGRand,0]
        # SetDelta = Parabel(height_values, *Samps[MWGRand,1:-1])
        #
        # TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
        # TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
        # Diag = np.eye(n) * np.sum(TriU + TriL, 0)
        #
        # L_d = -TriU + Diag - TriL
        # L_d[0, 0] = 2 * L_d[0, 0]
        # L_d[-1, -1] = 2 * L_d[-1, -1]
        # SetB = SetGamma * ATA +  L_d
        #
        # W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)) )
        # v_1 = np.sqrt(SetGamma) * A.T @ W.reshape((m,1))
        # W2 = np.random.multivariate_normal(np.zeros(len(L)), L_d )
        # v_2 = W2.reshape((n,1))
        #
        # RandX = (SetGamma * ATy + v_1 + v_2)
        # O3_Prof, exitCode = gmres(SetB, RandX[0::, 0], tol=tol)
        # Results[round, :] = O3_Prof / theta_scale_O3
        # deltRes[round, :] = np.array([Samps[MWGRand, 1:-1]])
        # gamRes[round] = SetGamma

        Results[round, :] = VMR_O3
        #print(np.mean(O3_Prof))
        print(popt)
        #SetGamma = 3.5e-9
        tWalkSampNum = 100000
        A, theta_scale = composeAforPress(A_lin, TempResults[round-1, :].reshape((n,1)), Results[round, :], ind)
        SampParas = tWalkPress(height_values, A, y, popt, tWalkSampNum, burnInT, SetGamma)

        randInd = np.random.randint(low=0, high=tWalkSampNum)

        sampB1 = SampParas[burnInT + randInd,0]
        sampB2 = SampParas[burnInT + randInd, 1]
        sampA1 = SampParas[burnInT + randInd, 2]
        sampA2 = SampParas[burnInT + randInd, 3]

        PressResults[round, :] = pressFunc(height_values, sampB1, sampB2, sampA1, sampA2)

        PressResults[round, :] = pressure_values

        # A, theta_scale_T = composeAforTemp(A_lin, PressResults[round,:], Results[round, :], ind, temp_values)
        #
        # TempBurnIn = 2500
        # TempWalkSampNum = 75000
        # TempSamps = tWalkTemp(height_values, A, y, TempWalkSampNum, TempBurnIn, SetGamma, SpecNumLayers, h0, h1, h2, h3, h4, h5, a0, a1, a2, a3,a4, b0)

        # randInd = np.random.randint(low=0, high=TempWalkSampNum)
        # h0 = TempSamps[TempBurnIn + randInd, 0]
        # h1 = TempSamps[TempBurnIn + randInd, 1]
        # h2 = TempSamps[TempBurnIn + randInd, 2]
        # h3 = TempSamps[TempBurnIn + randInd, 3]
        # h4 = TempSamps[TempBurnIn + randInd, 4]
        # h5 = TempSamps[TempBurnIn + randInd, 5]
        # a0 = TempSamps[TempBurnIn + randInd, 6]
        # a1 = TempSamps[TempBurnIn + randInd, 7]
        # a2 = TempSamps[TempBurnIn + randInd, 8]
        # a3 = TempSamps[TempBurnIn + randInd, 9]
        # a4 = TempSamps[TempBurnIn + randInd, 10]
        # b0 = TempSamps[TempBurnIn + randInd, 11]
        #
        # TempResults[round, :] = temp_func(height_values,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,b0).reshape(n)

        TempResults[round, :] = temp_values.reshape(n)


    print('elapsed time:' + str(time.time() - startTime))
    np.savetxt('data/deltRes'+ str(t).zfill(3) +'.txt', deltRes, fmt = '%.15f', delimiter= '\t')
    np.savetxt('data/gamRes'+ str(t).zfill(3) +'.txt', gamRes, fmt = '%.15f', delimiter= '\t')
    np.savetxt('data/O3Res'+ str(t).zfill(3) +'.txt', Results/theta_scale_O3, fmt = '%.15f', delimiter= '\t')
    np.savetxt('data/PressRes'+ str(t).zfill(3) +'.txt', PressResults, fmt = '%.15f', delimiter= '\t')
    np.savetxt('data/TempRes'+ str(t).zfill(3) +'.txt', TempResults, fmt = '%.15f', delimiter= '\t')

print('finished')


fig, axs = plt.subplots(5,1, tight_layout = True)
for i in range(0,5):
    axs[i].hist(SampParas[:,i],bins=n_bins)
axs[0].set_xlabel('$b_1$')
axs[1].set_xlabel('$b_2$')
axs[2].set_xlabel('$h_0$')
axs[3].set_xlabel('$p_0$')
axs[4].set_xlabel('$\gamma$')
#fig.savefig('pressHistRes.svg')
plt.show()


fig3, ax1 = plt.subplots(tight_layout=True, figsize=set_size(245, fraction=fraction))

ax1.plot(pressure_values, height_values, label='true pressure', color = 'green', marker ='o', zorder =1, markersize=10)
tests = 100
sampB1 = SampParas[np.random.randint(low=burnInT, high=tWalkSampNum, size=tests), 0]
sampB2 = SampParas[np.random.randint(low=burnInT, high=tWalkSampNum, size=tests), 1]
sampA1 = SampParas[np.random.randint(low=burnInT, high=tWalkSampNum, size=tests), 2]
sampA2 = SampParas[np.random.randint(low=burnInT, high=tWalkSampNum, size=tests), 3]
for r in range(0, tests):

    Sol = pressFunc(height_values, sampB1[r], sampB2[r], sampA1[r], sampA2[r])

    ax1.plot(Sol, height_values, marker='+', color='r', zorder=0, linewidth=0.5,
             markersize=5)
#PressProf = np.mean(PressResults[1:],0)
#ax1.plot(PressProf, height_values, marker='>', color="k", label='sample mean', zorder=2, linewidth=0.5,markersize=5)

#ax1.plot(2500 * np.exp(-np.mean(grad) * height_values[:,0]),height_values[:,0])
ax1.set_xlabel(r'pressure in hPa ')
ax1.set_ylabel('height in km')
ax1.legend()
#plt.savefig('samplesPressure.png')
plt.show()

print('plot')
##
# fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)#tight_layout = True,
# #axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# for i in range(0,3):
#     print(i)
#
#     axs[i].hist(TempSamps[:,i],bins=n_bins)
#     # axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
#     # labels = axs[i].get_yticklabels()
#     # labels[0] = ' '
#     axs[i].set_yticklabels([])
#
#
# axs[0].set_xlabel('$h_0$')
# axs[1].set_xlabel('$h_1$')
# axs[2].set_xlabel('$h_2$')
#
# #axs[3].set_xlabel('$h_3$')
# #axs[4].set_xlabel('$h_4$')
# #axs[5].set_xlabel('$h_5$')
# #axs[6].set_xlabel('$a_0$')
# fig.savefig('tempHistRes0.svg')
#
# plt.show()
#
# fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)#tight_layout = True,
#
# for i in range(3,6):
#     axs[i-3].hist(TempSamps[:,i],bins=n_bins)
#     # labels = axs[i-3].get_yticklabels()
#     # labels[0] = ' '
#     # axs[i-3].set_yticklabels(labels)
#     axs[i-3].set_yticklabels([])
#
# axs[0].set_xlabel('$h_3$')
# axs[1].set_xlabel('$h_4$')
# axs[2].set_xlabel('$h_5$')
# #axs[6].set_xlabel('$a_0$')
# fig.savefig('tempHistRes1.svg')
#
# plt.show()
#
#
# fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
#
# for i in range(6,9):
#     axs[i-6].hist(TempSamps[:,i],bins=n_bins)
#     # labels = axs[i-6].get_yticklabels()
#     # labels[0] = ' '
#     # axs[i-6].set_yticklabels(labels)
#     axs[i-6].set_yticklabels([])
# axs[0].set_xlabel('$a_0$')
# axs[1].set_xlabel('$a_1$')
# axs[2].set_xlabel('$a_2$')
# # axs[2].set_xlabel('$a_3$')
# # axs[3].set_xlabel('$a_4$')
# # axs[4].set_xlabel('$b_0$')
# # axs[5].set_xlabel('$\gamma$')
#
# fig.savefig('tempHistRes2.svg')
# plt.show()
#
#
# fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
# for i in range(9,12):
#     # axs[i-9].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
#     axs[i-9].hist(TempSamps[:,i],bins=n_bins)
#     # labels = axs[i-9].get_yticklabels()
#     # labels[0] = ' '
#     # axs[i-9].set_yticklabels(labels)
#     axs[i-9].set_yticklabels([])
# # axs[0].set_xlabel('$a_1$')
# # axs[1].set_xlabel('$a_2$')
# axs[0].set_xlabel('$a_3$')
# axs[1].set_xlabel('$a_4$')
# axs[2].set_xlabel('$b_0$')
# #axs[5].set_xlabel('$\gamma$')
#
# fig.savefig('tempHistRes3.svg')
# plt.show()
# fig_width_in, fig_height_in = set_size(PgWidthPt, fraction=fraction)
# fig, axs = plt.subplots(1,1, figsize=(fig_width_in, fig_height_in/2.7), tight_layout = True)
# axs.ticklabel_format(axis='y', style='sci',scilimits=(0,0) )
# axs.hist(TempSamps[:, -2], bins=n_bins)
# axs.set_yticklabels([])
# axs.set_xlabel('$\gamma$')
# fig.savefig('tempHistRes4.svg')
# plt.show()
##
fig, axs = plt.subplots()#figsize = (7,  2))
# We can set the number of bins with the *bins* keyword argument.
axs.hist(gamRes,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoGam)))
axs.set_title('$\gamma$')
#axs.set_title(str(len(new_gam)) + r' $\gamma$ samples, the noise precision')
#axs.set_xlabel(str(len(new_gam)) + ' effective $\gamma$ samples')
axs.axvline(x=gamma, color = 'r')
#tikzplotlib.save("HistoResults1.tex",axis_height='3cm', axis_width='7cm')
#plt.close()
fig.savefig('gamHistRes.svg')
plt.show()

fig, axs = plt.subplots(4,1,figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(deltRes[:,2],bins=n_bins, color = 'k')
axs[0].set_xlabel('$\delta_0$ samples')
axs[1].hist(deltRes[:,1],bins=n_bins, color = 'k')
axs[1].set_xlabel('$a$ samples')
axs[2].hist(deltRes[:,0],bins=n_bins, color = 'k')
axs[2].set_xlabel('$h_0$ samples')
axs[3].hist(gamRes,bins=n_bins, color = 'k')
axs[3].set_xlabel('$\gamma$ samples')
axs[3].axvline(x=gamma, color = 'r')
fig.savefig('allHistoRes.svg')
plt.show()


##

# deltRes = np.loadtxt('deltRes.txt', delimiter= '\t')
# gamRes = np.loadtxt('gamRes.txt', delimiter= '\t')
# VMR_O3 = np.loadtxt('VMR_O3.txt', delimiter= '\t')
# O3Res = np.loadtxt('O3Res.txt', delimiter= '\t')
# PressResults = np.loadtxt('PressRes.txt', delimiter= '\t')
# Results = O3Res  * theta_scale_O3
# SampleRounds = len(gamRes)

##
plt.close('all')
DatCol =  'gray'
ResCol = "#1E88E5"
TrueCol = [50/255,220/255, 0/255]



fig3, ax2 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
line3 = ax2.scatter(y, tang_heights_lin, label  = r'data $\bm{y}$', zorder = 0, marker = '*', color =DatCol )#,linewidth = 5

ax1 = ax2.twiny()

ax1.plot(VMR_O3,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=1 ,linewidth = 1.5, markersize =7)

for r in range(1,SampleRounds):
    Sol = Results[r, :]

    ax1.plot(Sol,height_values,marker= '+',color = ResCol, zorder = 0, linewidth = 0.5, markersize = 5,label = r'$\bm{x} \sim \pi(\bm{x}|\bm{y}, \bm{\theta})$')
    # with open('Samp' + str(n) +'.txt', 'w') as f:
    #     for k in range(0, len(Sol)):
    #         f.write('(' + str(Sol[k]) + ' , ' + str(height_values[k]) + ')')
    #         f.write('\n')
O3_Prof = np.mean(Results[1:],0)

ax1.plot(O3_Prof, height_values, marker='>', color="k", label='sample mean', zorder=2, linewidth=0.5,
             markersize=5)

ax1.set_xlabel(r'ozone volume mixing ratio ')

ax2.set_ylabel('(tangent) height in km')
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.set_ylim([height_values[0], height_values[-1]])

ax2.set_xlabel(r'spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,

ax2.tick_params(colors = DatCol, axis = 'x')
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.spines[:].set_visible(False)
#ax2.spines['top'].set_color(pyTCol)

legend = ax1.legend(handles = [handles[-3], handles2[0], handles[0]])# loc='lower right', framealpha = 0.2,fancybox=True)#, bbox_to_anchor=(1.01, 1.01), frameon =True)

#ax1.legend()
fig3.savefig('O3Results.svg')
plt.savefig('O3Results.png')
plt.show()


relErr = np.linalg.norm(O3_Prof - VMR_O3)/np.linalg.norm(VMR_O3) * 100

print(f'relative Error: {relErr:.2f} %')
##
fig3, ax1 = plt.subplots(tight_layout=True, figsize=set_size(245, fraction=fraction))
#ax1.plot(press, heights, label='true press.')
ax1.plot(pressure_values, height_values, label='true pressure', color = TrueCol, marker ='o', zorder =1, markersize=10)
#ax1.plot(recov_press, height_values, linewidth=2.5, label='samp. press. fit')  #
for r in range(0, SampleRounds):
    Sol = PressResults[r, :]

    ax1.plot(Sol, height_values, marker='+', color=ResCol, zorder=0, linewidth=0.5,
             markersize=5)
PressProf = np.mean(PressResults[1:],0)
ax1.plot(PressProf, height_values, marker='>', color="k", label='sample mean', zorder=2, linewidth=0.5,
         markersize=5)

#ax1.plot(2500 * np.exp(-np.mean(grad) * height_values[:,0]),height_values[:,0])
ax1.set_xlabel(r'pressure in hPa ')
ax1.set_ylabel('height in km')
ax1.legend()
plt.savefig('samplesPressure.png')
plt.show()
##

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))

for r in range(0, SampleRounds):
    Sol = Parabel(height_values, *deltRes[r, :])

    ax1.plot(Sol, height_values, marker='+', color=ResCol, zorder=0, linewidth=0.5)
ax1.set_xlabel(r'$\delta$ ')
ax1.set_ylabel('height in km')
#plt.savefig('DeltaSamp.png')
plt.savefig('DeltaSamp.svg')
plt.show()

##
fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
for r in range(0, SampleRounds):
    Sol = TempResults[r, :]

    ax1.plot(Sol, height_values, marker='+', color=ResCol, zorder=0, linewidth=0.5,
             markersize=5)

TempProf = np.mean(TempResults[1:], 0)
ax1.plot(TempProf, height_values, marker='>', color="k", label='sample mean', zorder=2, linewidth=0.5,
         markersize=5)

ax1.plot(temp_values, height_values, linewidth=5, label='true T', color='green', zorder=0)
ax1.legend()
plt.savefig('TemperatureSamp.png')
plt.show()

#tikzplotlib.save("FirstRecRes.pgf")
print('done')



# def Parabel(x, h0, a0, d0):
#
#     return a0 * np.power((h0-x),2 )+ d0
#
#
#
# def oneParabeltoConst(x, h0, a0, d0):
#     a = np.ones(x.shape)
#     a[x <= h0] = a0
#     a[x > h0] = 0#-a1
#     return a * (h0 -x)**2 + d0
#
#
#
# B0 = ATA + lam0 * L
# B_inv_A_trans_y0, exitCode = gmres(B0, ATy[:,0], tol=tol, restart=25)
#
# B0u, B0s, B0vh = np.linalg.svd(B0)
# cond_B0 = np.max(B0s)/np.min(B0s)
# print("Condition Number B0: " + str(orderOfMagnitude(cond_B0)))
#
# def log_post(Params):
#     tol = 1e-8
#     n = SpecNumLayers
#     m = SpecNumMeas
#     # gamma = params[0]
#     # delta = params[1]
#     gam = Params[0]
#     h1 = Params[1]
#     a0 = Params[2]
#     # h0 = Params[2]
#
#
#     # mean = Params[1]
#     # w = Params[2]
#     # skewP = Params[3]
#     # scale = Params[4]
#     d0 = Params[3]
#     #a1 = Params[4]
#     delta = Parabel(height_values,h1, a0, d0)
#     #delta = oneParabeltoConst(height_values,h1, a0, d0)
#     #delta = simpleDFunc(height_values, h1,a0, d0)
#     #delta = twoParabel(height_values, a0, 0, h1, 0)
#     #delta = skew_norm_pdf(height_values, 16, 50, 8, 9.5e-05, 3.7e-03)
#     #delta = skew_norm_pdf(height_values[:,0],mean,w,skewP, scale, d0)
#     TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * delta
#     TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * delta.T
#     Diag = np.eye(n) * np.sum(TriU + TriL, 0)
#
#     L_d = -TriU + Diag - TriL
#     L_d[0, 0] = 2 * L_d[0, 0]
#     L_d[-1, -1] = 2 * L_d[-1, -1]
#
#     try:
#         L_du, L_ds, L_dvh = np.linalg.svd(L_d)
#         detL = np.sum(np.log(L_ds))
#     except np.linalg.LinAlgError:
#         print("SVD did not converge, use scipy.linalg.det()")
#         detL = np.log(scy.linalg.det(L_d))
#
#     Bp = ATA + 1/gam * L_d
#
#     B_inv_A_trans_y, exitCode = gmres(Bp, ATy[:,0], x0= B_inv_A_trans_y0, tol=tol, restart=25)
#     if exitCode != 0:
#         print(exitCode)
#
#     G = g(A, L_d,  1/gam)
#     F = f(ATy, y,  B_inv_A_trans_y)
#     alphaD = 1
#     alphaG = 1
#     #hMean = tang_heights_lin[y[:,0] == np.max(y[:,0])]
#     #hMean = tang_heights_lin[Ax == np.max(Ax)]
#     hMean = height_values[VMR_O3[:] == np.max(VMR_O3[:])]
#     #hMean = 25
#     alphaA1 = (lam0 * gamma0*0.75) / (hMean - np.min(height_values))**2
#     alphaA2 = (lam0 * gamma0*0.75) / (hMean - np.max(height_values)) ** 2
#     if alphaA2 < alphaA1:
#         alphaA = 1/alphaA2
#     else:
#         alphaA = 1/alphaA1
#     #sigmaP = 100
#     #return - (0.5 + alphaD - 1 ) * np.sum(np.log(delta/gam))  - (m/2+1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( 1e1 *  np.sum(delta) + 1e2 *gam)+ ((8 - mean)/sigmaP) ** 2 + (( 1.7e-03 - d0)/1e-3) ** 2 + (( 5 - skewP)/10) ** 2 +(( 4.2e-05 - scale)/1e-4) ** 2 +(( 50 - w)/20) ** 2
#     #return - (0.5 + alphaD - 1 ) * np.sum(np.log(delta/gam))  - (m/2+1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( 1e4 *  np.sum(delta)/n + betaG *gam)+ 0.5 * ((20 -Params[1])/25) ** 2 + 0.5* (( 1e-4 - Params[2])/2e-4) ** 2
#     #return - (0.5* n)  * np.log(1/gam) - 0.5 * np.sum(np.log(L_ds)) - (alphaD - 1) * np.log(d0) - (m/2+1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( 1e4 * d0 + betaG *gam) - 0 * np.log(Params[2]) + 1e3* Params[2] - 0.1*  np.log(Params[1]) + 1e-4* Params[1]
#     #return - (0.5* n)  * np.log(1/gam) - 0.5 * np.sum(np.log(L_ds)) - (alphaD - 1) * np.log(d0) - (m/2+1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( 3e4 * d0 + 1e5 *gam) - 11 * np.log(Params[1]) + 5e-1* Params[1] - 0.2*  np.log(Params[2]) + 5e7* Params[2]
#     #return - (m/2 - n/2 + alphaG -1) * np.log(gam) - 0.5 * np.sum(np.log(L_ds)) - (alphaD - 1) * np.log(d0) + 0.5 * G + 0.5 * gam * F +  (1/(lam0 * gamma0*1e-1) * d0 +7e9 *gam) - 0.3 * np.log(Params[1]) +1e-3 * Params[1] #- 0*  np.log(Params[2]) + 1e7* Params[2]
#     return - (m/2 - n/2 + alphaG -1) * np.log(gam) - 0.5 * detL - (alphaD - 1) * np.log(d0) + 0.5 * G + 0.5 * gam * F +  (1/(gamma0*lam0*0.4) * d0 + betaG *gam)  - 0*  np.log(Params[2]) + alphaA* Params[2]- 0 * np.log(Params[1]) + 0.5* ((Params[1]-hMean)/2)**2
#
#
#
# def MargPostSupp(Params):
#     list = []
#     list.append(Params[0] > 0)
#     list.append(height_values[-1]> Params[1] >height_values[0])
#     list.append(Params[2] > 0)
#     list.append(lam0 * gamma0 >Params[3] > 0)
#     return all(list)
#
#
# MargPost = pytwalk.pytwalk(n=4, U=log_post, Supp=MargPostSupp)
# # startTime = time.time()
# #x0 = np.array([gamma, 8, 50, 5, 4.2e-05,1.7e-03])
# x0 = np.array([gamma,29, 5e-7, lam0 * gamma0*0.4])
# xp0 = 1.01 * x0
# burnIn = 500
# tWalkSampNum = 20000
# MargPost.Run(T=tWalkSampNum + burnIn, x0=x0, xp0=xp0)
#
# Samps = MargPost.Output
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(Samps[:,0], bins = 50)
# ax1.axvline(x=gamma, color = 'r')
# plt.show()
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(Samps[:,1], bins = 50)
# #ax1.axvline(x=popt[1], color = 'r')
# plt.show()
#
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(Samps[:,2], bins = 50)
#
# plt.show()
#
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(Samps[:,3], bins = 50)
# plt.show()
# ##
# fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
#
# for p in range(burnIn, tWalkSampNum,500):
#     Sol = Parabel(height_values, *Samps[p, 1:-1])
#
#     ax1.plot(Sol, height_values, linewidth=0.5)
# ax1.set_xlabel(r'$\delta$ ')
# ax1.set_ylabel('Height in km')
#
#
# plt.show()
#
# ##
# xm = np.mean(Samps[:,2])
#
# hMean = 27.5  # tang_heights_lin[y[:,0] == np.max(y[:,0])]
# alphaA1 = (lam0 * gamma0 * 0.6) / (hMean - np.min(height_values)) ** 2
# alphaA2 = (lam0 * gamma0 * 0.6) / (hMean - np.max(height_values)) ** 2
# if alphaA2 < alphaA1:
#     alphaA = 1 / alphaA2
# else:
#     alphaA = 1 / alphaA1
# def normalprior(x):
#     sigma =2
#
#
#     return 1/sigma * np.exp(-0.5 * ((x - xm)/(sigma))**2)
#
# def expDelta(x, a,b,d0):
#     # a = 4
#     # b = 1e-1
#     # d0 = 50
#     return x**a * np.exp(-b * x) + d0
# xTry = np.linspace(0,3*(xm),100)
# fig3, ax1 = plt.subplots()
# #ax1.scatter(xTry, normalprior(xTry) , color = 'r')
# ax1.scatter(xTry, expDelta(xTry,0,alphaA,0) , color = 'r')
# ax1.axvline(x=xm, color = 'r')
# #ax1.scatter(expDelta(height_values,4,1e-1,50), height_values, color = 'r')
# plt.show()
#
# ##
# #ds = oneParabeltoConst(height_values,np.mean(Samps[:,1]),np.mean(Samps[:,2])-1.6e-7,np.mean(Samps[:,3])+1e-5)
# ds = Parabel(height_values,np.mean(Samps[:,1]),np.mean(Samps[:,2]),np.mean(Samps[:,3]))
#
# fig3, ax1 = plt.subplots()
# ax1.scatter(ds,height_values, color = 'r')
# #ax1.scatter(paraDs,height_values, color = 'b')
# plt.show()
#
#
#
# n = SpecNumLayers
# m = SpecNumMeas
# paraSamp = 100#n_bins
# NewResults = np.zeros((paraSamp,n))
# #SetDelta = skewDsTry #ds
# SetGamma = gamma
# randInd = np.random.randint(low=burnIn, high=tWalkSampNum+burnIn, size = paraSamp)
# for p in range(paraSamp):
#     SetGamma = Samps[randInd[p],0]
#     #SetDelta = twoParabel(height_values,Samps[randInd[p],1], 0, Samps[randInd[p],2],0)
#     #SetDelta = simpleDFunc(height_values, Samps[randInd[p], 1], Samps[randInd[p], 2],  Samps[randInd[p], 2])
#     #SetDelta = oneParabeltoConst(height_values, Samps[randInd[p], 1], Samps[randInd[p], 2],  Samps[randInd[p], 3])
#     SetDelta = Parabel(height_values, Samps[randInd[p], 1], Samps[randInd[p], 2],  Samps[randInd[p], 3])
#     #SetDelta = ds
#     Mu = np.zeros((n,1))
#     #Mu = 0.3e-6 * theta_scale_O3
#     TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
#     TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
#     Diag = np.eye(n) * np.sum(TriU + TriL, 0)
#
#     L_d = -TriU + Diag - TriL
#     L_d[0, 0] = 2 * L_d[0, 0]
#     L_d[-1, -1] = 2 * L_d[-1, -1]
#     SetB = SetGamma * ATA +  L_d
#
#     W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)) )
#     v_1 = np.sqrt(SetGamma) * A.T @ W.reshape((m,1))
#     W2 = np.random.multivariate_normal(np.zeros(len(L)), L_d )
#     v_2 = W2.reshape((n,1))
#
#     RandX = (SetGamma * ATy + L_d @ Mu + v_1 + v_2)
#     NewResults[p,:], exitCode = gmres(SetB, RandX[0::, 0], tol=tol)
#     print(np.mean(NewResults[p,:]))
# ResCol = "#1E88E5"
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# #ax1.plot(Res/theta_scale_O3, height_values, linewidth = 2.5, label = 'my guess', marker = 'o')
#
# for p in range(0, paraSamp):
#     Sol = NewResults[p, :] / theta_scale_O3
#     ax1.plot(Sol, height_values, marker='+', color=ResCol, zorder=1, linewidth=0.5, markersize=5)
#
# ax1.plot(VMR_O3, height_values, linewidth = 2.5, label = 'true profile', marker = 'o', color = "k")
# O3_Prof = np.mean(NewResults,0)/ theta_scale_O3
#
# ax1.plot(O3_Prof, height_values, marker='>', color="k", zorder=2, linewidth=0.5,
#              markersize=5)
# ax1.set_ylabel('Height in km')
# ax1.set_xlabel('Volume Mixing Ratio of Ozone')
# ax2 = ax1.twiny()
# ax2.scatter(y, tang_heights_lin ,linewidth = 2, marker =  'x', label = 'data' , color = 'k')
# ax2.set_xlabel(r'Spectral radiance in $\frac{W cm}{m^2  sr} $',labelpad=10)# color =dataCol,
# ax1.legend()
# #plt.savefig('DataStartTrueProfile.png')
# plt.show()
#
#
# print('bla')
