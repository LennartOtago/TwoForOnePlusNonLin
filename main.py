import numpy as np
import matplotlib as mpl
#from importetFunctions import *
import time
import pickle as pl
import matlab.engine
from functions import *
#from errors import *
from scipy import constants, optimize
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
#import tikzplotlib
plt.rcParams.update({'font.size': 18})
import pandas as pd
from numpy.random import uniform, normal, gamma
import scipy as scy
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

#mpl.rc('text.latex', preamble=r"\boldmath")

""" for plotting figures,
PgWidth in points, either collumn width page with of Latex"""
fraction = 1.5
dpi = 300
PgWidthPt = 245
defBack = mpl.get_backend()

""" for plotting histogram and averaging over lambda """
n_bins = 20

""" for MwG"""
burnIn = 50

betaG =1e-4# 1e-18#
betaD = 1e3#9e3#1e-3#1e-10#1e-22#  # 1e-4

""" for B_inve"""
tol = 1e-8

df = pd.read_excel('ExampleOzoneProfiles.xlsx')
#print the column names
print(df.columns)

#get the values for a given column
press = df['Pressure (hPa)'].values #in hectpascal or millibars
O3 = df['Ozone (VMR)'].values

minInd = 7
maxInd = 42
pressure_values = press[minInd:maxInd]
VMR_O3 = O3[minInd:maxInd]
scalingConstkm = 1e-3

def height_to_pressure(p0, x, dx):
    R = constants.gas_constant
    R_Earth = 6371  # earth radiusin km
    grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
    temp = get_temp(x)
    return p0 * np.exp(-28.97 * grav / temp / R * dx  )
##
calc_press = np.zeros((len(press)+1,1))
calc_press[0] = 1013.25
calc_press[1:] = press.reshape((len(press),1)) #hPa
actual_heights = np.zeros(len(press)+1)
try_heights = np.logspace(0,2.2,1000)
try_heights[0] = 0

for i in range(1,len(calc_press)):
    #k = 0
    for j in range(0, len(try_heights)-1):
        curr_press = height_to_pressure(calc_press[i-1], actual_heights[i-1], try_heights[j] - actual_heights[i-1])
        next_press = height_to_pressure(calc_press[i-1], actual_heights[i-1], try_heights[j+1] - actual_heights[i-1])
        #print(curr_press)
        if abs(calc_press[i]-curr_press) < abs(calc_press[i]-next_press):
            next_press = height_to_pressure(calc_press[i - 1], actual_heights[i - 1],
                                            try_heights[j - 1] - actual_heights[i - 1])

            if abs(calc_press[i]-curr_press) > abs(calc_press[i]-next_press):
                actual_heights[i] = try_heights[j-1]
                k = j-1
            else:
                actual_heights[i] = try_heights[j]
                k = j
            break

print('got heights')



'''fit pressure'''
#efit, dfit, cfit,
cfit, bfit, afit = np.polyfit(actual_heights, np.log(calc_press), 2)


def pressFunc(a,b,c,d,e,x):
    #a[0] = pressure_values[0]*1.75e1
    return np.exp( e * x**4 + d * x**3 + c * x**2 + b * x + a)


# fig, axs = plt.subplots(tight_layout=True)
# plt.plot(calc_press,actual_heights)
# plt.plot(pressFunc(afit, bfit, cfit, 0, 0,actual_heights),actual_heights)
# plt.show()
heights = actual_heights[1:]
##
# https://en.wikipedia.org/wiki/Pressure_altitude
# https://www.weather.gov/epz/wxcalc_pressurealtitude
#heights = 145366.45 * (1 - ( press /1013.25)**0.190284 ) * 0.3048 * scalingConstkm

SpecNumLayers = len(VMR_O3)
height_values = heights[minInd:maxInd].reshape((SpecNumLayers,1))
np.savetxt('height_values.txt',height_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('pressure_values.txt',pressure_values, fmt = '%.15f', delimiter= '\t')
temp_values = get_temp_values(height_values)
""" analayse forward map without any real data values"""

MinH = height_values[0]
MaxH = height_values[-1]
R_Earth = 6371 # earth radiusin km
ObsHeight = 500 # in km

''' do svd for one specific set up for linear case and then exp case'''

#find best configuration of layers and num_meas
#so that cond(A) is not inf
#exp case first
SpecNumMeas = 45
SpecNumLayers = len(height_values)

n = SpecNumLayers
m = SpecNumMeas

# find minimum and max angle in radians
# min and max angle are defined by the height values of the retrived profile
MaxAng = np.arcsin((height_values[-1]+ R_Earth) / (R_Earth + ObsHeight))
MinAng = np.arcsin((height_values[0] + R_Earth) / (R_Earth + ObsHeight))

#find best configuration of layers and num_meas
#so that cond(A) is not inf
# coeff = 1/np.log(SpecNumMeas)
# meas_ang = (MinAng) + (MaxAng - MinAng) * coeff * np.log( np.linspace(1, int(SpecNumMeas) , SpecNumMeas ))

# coeff = 1/(SpecNumMeas)
# meas_ang = (MinAng) + (MaxAng - MinAng) * np.exp(- coeff *4* np.linspace(0, int(SpecNumMeas) -1 , SpecNumMeas ))
# meas_ang = np.flip(meas_ang)

meas_ang = np.linspace(MinAng, MaxAng, SpecNumMeas)
meas_ang = np.arange(MinAng, MaxAng, 0.0009)
#meas_ang = np.arange(MinAng, MaxAng, 0.0004)
SpecNumMeas = len(meas_ang)
m = SpecNumMeas

A_lin, tang_heights_lin, extraHeight = gen_sing_map(meas_ang,height_values,ObsHeight,R_Earth)


fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.scatter(range(0,SpecNumMeas), tang_heights_lin)
plt.show()

ATA_lin = np.matmul(A_lin.T,A_lin)
#condition number for A
A_lin = A_lin
A_linu, A_lins, A_linvh = np.linalg.svd(A_lin)
cond_A_lin =  np.max(A_lins)/np.min(A_lins)
print("normal: " + str(orderOfMagnitude(cond_A_lin)))



#to test that we have the same dr distances
tot_r = np.zeros((SpecNumMeas,1))
#calculate total length
for j in range(0, SpecNumMeas):
    tot_r[j] = 2 * (np.sqrt( ( extraHeight + R_Earth)**2 - (tang_heights_lin[j] +R_Earth )**2) )
print('Distance through layers check: ' + str(np.allclose( sum(A_lin.T,0), tot_r[:,0])))





#taylor exapnsion for f to do so we need y (data)

##
''' load data and pick wavenumber/frequency'''
#check absoprtion coeff in different heights and different freqencies
filename = 'tropical.O3.xml'

N_A = constants.Avogadro # in mol^-1
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
R_gas = N_A * k_b_cgs # in ..cm^3

files = '634f1dc4.par' #/home/lennartgolks/Python /Users/lennart/PycharmProjects

my_data = pd.read_csv(files, header=None)
data_set = my_data.values

size = data_set.shape
wvnmbr = np.zeros((size[0],1))
S = np.zeros((size[0],1))
F = np.zeros((size[0],1))
g_air = np.zeros((size[0],1))
g_self = np.zeros((size[0],1))
E = np.zeros((size[0],1))
n_air = np.zeros((size[0],1))
g_doub_prime= np.zeros((size[0],1))


for i, lines in enumerate(data_set):
    wvnmbr[i] = float(lines[0][5:15]) # in 1/cm
    S[i] = float(lines[0][16:25]) # in cm/mol
    F[i] = float(lines[0][26:35])
    g_air[i] = float(lines[0][35:40])
    g_self[i] = float(lines[0][40:45])
    E[i] = float(lines[0][46:55])
    n_air[i] = float(lines[0][55:59])
    g_doub_prime[i] = float(lines[0][155:160])


#load constants in si annd convert to cgs units by multiplying
h = scy.constants.h #* 1e7#in J Hz^-1
c_cgs = constants.c * 1e2# in m/s
k_b_cgs = constants.Boltzmann #* 1e7#in J K^-1
#T = temp_values[0:-1] #in K
N_A = constants.Avogadro # in mol^-1



mol_M = 48 #g/mol for Ozone
#ind = 293
ind = 623
#pick wavenumber in cm^-1
v_0 = wvnmbr[ind][0]#*1e2
#wavelength
lamba = 1/v_0
f_0 = c_cgs*v_0
print("Frequency " + str(np.around(v_0*c_cgs/1e9,2)) + " in GHz")

C1 =2 * scy.constants.h * scy.constants.c**2 * v_0**3 * 1e8
C2 = scy.constants.h * scy.constants.c * 1e2 * v_0  / (scy.constants.Boltzmann * temp_values )
#plancks function
Source = np.array(C1 /(np.exp(C2) - 1) ).reshape((SpecNumLayers,1))

#differs from HITRAN, implemented as in Urban et al
T_ref = 296 #K usually
p_ref = pressure_values[0]




'''weighted absorption cross section according to Hitran and MIPAS instrument description
S is: The spectral line intensity (cm^−1/(molecule cm^−2))
f_broad in (1/cm^-1) is the broadening due to pressure and doppler effect,
 usually one can describe this as the convolution of Lorentz profile and Gaussian profile
 VMR_O3 is the ozone profile in units of molecule (unitless)
 has to be extended if multiple gases are to be monitored
 I multiply with 1e-4 to go from cm^2 to m^2
 '''
f_broad = 1
w_cross =   f_broad * 1e-4 * VMR_O3#np.mean(VMR_O3) * np.ones((SpecNumLayers,1))
#w_cross[0], w_cross[-1] = 0, 0

#from : https://hitran.org/docs/definitions-and-units/
HitrConst2 = 1.4387769 # in cm K

# internal partition sum
Q = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ temp_values)
Q_ref = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ 296)
LineInt = S[ind,0] * Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ temp_values)/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ temp_values))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))
LineIntScal =  Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ temp_values)/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ temp_values))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))

''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''

#take linear
num_mole = 1 / ( scy.constants.Boltzmann )#* temp_values)

AscalConstKmToCm = 1e3
#1e2 for pressure values from hPa to Pa


scalingConst = 1e11
# A_scal_T = pressure_values.reshape((SpecNumLayers,1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm * num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]
#
# theta_O3 = num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]


A_scal_O3 = 1e2 * LineIntScal  * Source * AscalConstKmToCm * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0] * num_mole / temp_values.reshape((SpecNumLayers,1))
#scalingConst = 1e11

theta_P = pressure_values.reshape((SpecNumLayers,1))

""" plot forward model values """


A = A_lin * A_scal_O3.T
np.savetxt('AMat.txt', A, fmt='%.15f', delimiter='\t')
ATA = np.matmul(A.T,A)
Au, As, Avh = np.linalg.svd(A)
cond_A =  np.max(As)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))


Ax = np.matmul(A, theta_P)

#convolve measurements and add noise
y, gamma  = add_noise(Ax, 20)#90 works fine
np.savetxt('dataY.txt', y, header = 'Data y including noise', fmt = '%.15f')
ATy = np.matmul(A.T,y)
# y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
# gamma = 7.6e-5
#SNR = np.mean(Ax**2)/np.var(y)
SNR = np.mean(np.abs(Ax) ** 2)*gamma
print(SNR)
#gamma = 1/(np.max(Ax) * 0.1)**2

''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''


##

# graph Laplacian
# direchlet boundary condition
NOfNeigh = 2#4
neigbours = np.zeros((len(height_values),NOfNeigh))

for i in range(0,len(height_values)):
    neigbours[i] = i-1, i+1


neigbours[neigbours >= len(height_values)] = np.nan
neigbours[neigbours < 0] = np.nan

L = generate_L(neigbours)

np.savetxt('GraphLaplacian.txt', L, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')

A, theta_scale_O3= composeAforO3(A_lin, temp_values, pressure_values, ind)
ATy = np.matmul(A.T, y)
ATA = np.matmul(A.T, A)
Ax =np.matmul(A, VMR_O3 * theta_scale_O3)
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


gamma0, lam0 = optimize.fmin(MinLogMargPostFirst, [gamma,(np.var(VMR_O3) * theta_scale_O3) /gamma ])
mu0 = 0
print(lam0)
print(lam0*gamma0)
##

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin)
ax1.scatter(y, tang_heights_lin)
ax1.plot(y, tang_heights_lin)
plt.show()


# def MinLogMargPostWithMean(params):#, coeff):
#     tol = 1e-8
#     # gamma = params[0]
#     # delta = params[1]
#     gam = params[0]
#     lamb = params[1]
#     mu = params[2]
#     if lamb < 0  or gam < 0 or mu < 0:
#         return np.nan
#
#     betaG = 1e-4
#     betaD = 1e-10
#     n = SpecNumLayers
#     m = SpecNumMeas
#     Bp = ATA + lamb * L
#     yMu = y - np.matmul(A, mu * np.ones((n,1)))
#     ATyMu = np.matmul(A.T, yMu)
#     B_inv_A_trans_yMu, exitCode = gmres(Bp, ATyMu[:,0], tol=tol, restart=25)
#     if exitCode != 0:
#         print(exitCode)
#
#     G = g(A, L,  lamb)
#     F = f(ATyMu, yMu,  B_inv_A_trans_yMu)
#
#     return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( betaD *  lamb * gam + betaG *gam)
#
#
# gamma0, lam0, mu0 = optimize.fmin(MinLogMargPostWithMean, [gamma, (np.var(VMR_O3) * theta_scale_O3) /gamma, np.mean(VMR_O3) * theta_scale_O3 ])
#
# print(lam0)

# ##
# def MinLogMargPostWithMean(params):#, coeff):
#     tol = 1e-8
#     # gamma = params[0]
#     # delta = params[1]
#     gam = params[0]
#     lamb = params[1]
#     n = SpecNumLayers
#     m = SpecNumMeas
#     paraMat = np.zeros((n, 3))
#     breakInd1 = 12
#     breakInd2 = 24
#     paraMat[0:breakInd1, 0] = np.ones(breakInd1)
#     paraMat[breakInd1:breakInd2, 1] = np.ones(breakInd2 - breakInd1)
#     paraMat[breakInd2:, 2] = np.ones(n - breakInd2)
#     mu = paraMat @ params[2:].reshape((3,1))
#     if lamb < 0  or gam < 0 or params[2] < 0 or params[4] < 0 or params[3] < 0:
#         return np.nan
#
#     betaG = 1e-4
#     betaD = 1e-10
#
#     Bp = ATA + lamb * L
#
#     yMu = y - np.matmul(A, mu )
#     ATyMu = np.matmul(A.T, yMu)
#     B_inv_A_trans_yMu, exitCode = gmres(Bp, ATyMu[:,0], tol=tol, restart=25)
#     if exitCode != 0:
#         print(exitCode)
#
#     G = g(A, L,  lamb)
#     F = f(ATyMu, yMu,  B_inv_A_trans_yMu)
#
#     return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( betaD *  lamb * gam + betaG *gam) + params[2] * 1e-5 + params[3] * 1e-5 + params[4] * 1e-5
#
#
#
# def MargPostSupp(Params):
# 	return all(0 < Params)
#
# MargPost = pytwalk.pytwalk( n=5, U=MinLogMargPostWithMean, Supp=MargPostSupp)
# startTime = time.time()
# tWalkSampNum= 100000
# burnIn = 1000
# MargPost.Run( T=tWalkSampNum+ burnIn, x0=np.array([gamma0, lam0, mu0, mu0, mu0]), xp0=1.02 * np.array([gamma0, lam0, mu0, mu0, mu0]) )
# SampParas = MargPost.Output
#
#
# fig, axs = plt.subplots( 5,1, tight_layout=True)
# axs[0].hist(SampParas[:,0], bins= 30)
# axs[0].set_ylabel('$\gamma$')
# axs[1].hist(SampParas[:,1], bins= 30)
# axs[1].set_ylabel('$\lambda$')
# axs[2].hist(SampParas[:,2], bins= 30)
# axs[2].set_ylabel('$\mu$')
# axs[3].hist(SampParas[:,3], bins= 30)
# axs[3].set_ylabel('$\mu$')
# axs[4].hist(SampParas[:,4], bins= 30)
# axs[4].set_ylabel('$\mu$')
# plt.show()


# ##

# n = SpecNumLayers
# m = SpecNumMeas
# #draw paramter samples
# paraSamp = 100#n_bins
# NewResults = np.zeros((paraSamp,n))
#
# paraMat = np.zeros((n, 3))
# breakInd1 = 12
# breakInd2 = 24
# paraMat[0:breakInd1, 0] = np.ones(breakInd1)
# paraMat[breakInd1:breakInd2, 1] = np.ones(breakInd2 - breakInd1)
# paraMat[breakInd2:, 2] = np.ones(n - breakInd2)
#
# #lam =3e5#lam0 #9e4
# delt = lam0 * gamma0
# SetGamma = gamma0
# SetDelta = 1e-7#lam * SetGamma
#
# # RandInd = np.random.randint(low=burnIn, high=tWalkSampNum, size=paraSamp)
# # SetGammas = SampParas[RandInd,0]
# # SetLambdas  = SampParas[RandInd,1]
# # Mus  = SampParas[RandInd,2:5]
#
# for p in range(paraSamp):
#     # SetGamma = SetGammas[p]
#     # SetDelta = SetGammas[p] * SetLambdas[p]
#     #Mu = 0#np.mean(VMR_O3) * theta_scale_O3#Mus[n]
#     Mu = np.zeros((n,1))
#     # Mu = paraMat @ Mus[p].reshape((3, 1))
#     #Mu = paraMat @ np.array([0,200,0]).reshape((3, 1))
#     SetB = SetGamma * ATA + SetDelta * L
#
#     W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)) )
#     v_1 = np.sqrt(SetGamma) * A.T @ W.reshape((m,1))
#     W2 = np.random.multivariate_normal(np.zeros(len(L)), L )
#     v_2 = np.sqrt(SetDelta) * W2.reshape((n,1))
#
#     RandX = (SetGamma * ATy + SetDelta * L @ Mu + v_1 + v_2)
#     NewResults[p,:], exitCode = gmres(SetB, RandX[0::, 0], tol=tol)
#
# ResCol = "#1E88E5"
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# #ax1.plot(Res/theta_scale_O3, height_values, linewidth = 2.5, label = 'my guess', marker = 'o')
#
# for n in range(0, paraSamp):
#     Sol = NewResults[n, :] / theta_scale_O3
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
# plt.savefig('DataStartTrueProfile.png')
# plt.show()
#


##

#
# def hypprior(x):
#     betah = 1e-5
#     betag = 1e1
#     betab = 1e-5
#     betam = 1e-5
#     return x**(0.45) * np.exp(-x * 4e2)
#
#
#
# xtry = np.linspace(0,1e-4,1000)
# #xtry = np.linspace(0,1e-5,100)
# #xtry = pressure_values
# #xtry = grad
# ytry = hypprior(xtry)
# #
# fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
# ax1.plot(xtry,ytry)
# plt.show()
##
"""update A so that O3 profile is constant"""
O3_Prof = np.mean(VMR_O3) * np.ones(SpecNumLayers)

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(O3_Prof, height_values, linewidth = 2.5, label = 'my guess', marker = 'o')
ax1.plot(VMR_O3, height_values, linewidth = 2.5, label = 'true profile', marker = 'o')
ax1.set_ylabel('Height in km')
ax1.set_xlabel('Volume Mixing Ratio of Ozone')
ax2 = ax1.twiny()
ax2.scatter(y, tang_heights_lin ,linewidth = 2, marker =  'x', label = 'data' , color = 'k')
ax2.set_xlabel(r'Spectral radiance in $\frac{W cm}{m^2  sr} $',labelpad=10)# color =dataCol,
ax1.legend()
plt.savefig('DataStartTrueProfile.png')
plt.show()





##
#
# grad = np.log(pressure_values[1:])- np.log(pressure_values[:-1])/(height_values[1:,0]- height_values[:-1,0])
# bfitup, afitup = np.polyfit(height_values[-20:,0], grad[-20:], 1)
# bfitlow, afitlow = np.polyfit(height_values[0:15,0], grad[0:15], 1)
#
# cross_heigth = (afitup - afitlow )/ (bfitlow - bfitup)

# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(bfitup *  height_values[:,0] + afitup, height_values, linewidth = 2)
# ax1.plot(bfitlow *  height_values[:,0] + afitlow, height_values, linewidth = 2)
# ax1.scatter(grad , height_values[1:])
# #ax1.axhline(y=height_values[breakInd][0])
# ax1.set_xlabel(r'Pressure in hPa ')
# ax1.set_ylabel('Height in km')
# #ax1.set_xscale('log')
# plt.savefig('samplesPressure.png')
# plt.show()
#




##
# breakInd = 15
# numPara = 2
# paraMat = np.zeros((len(height_values), numPara))
# #breakInd = 21
#
# paraMat[0:breakInd,0] = np.ones(breakInd)
# paraMat[breakInd:,1] = np.ones(int(len(height_values)) -breakInd)
# def pressFunc(x, b1, b2, h0, p0):
#     b = np.ones(len(x))
#     b[x>h0] = b2
#     b[x<=h0] = b1
#     return -b * (x - h0) + np.log(p0)
#
# popt, pcov = scy.optimize.curve_fit(pressFunc, height_values[:,0], np.log(pressure_values), p0=[-2e-2,-2e-2, 18, 15])
#
#
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

# ## prior prediction pressure
# breakInd1 = 10
# breakInd2 = 30
# paraMat = np.zeros((len(height_values), 3))
# paraMat[0:breakInd1,0] = np.ones(breakInd1)
# paraMat[breakInd1:breakInd2,1] = np.ones(breakInd2 -breakInd1)
# paraMat[breakInd2:,2] = np.ones(int(len(height_values)) -breakInd2)
#
# nSamp = 1000
# burnIn = 100
# k = 0
# deltas = np.zeros((burnIn+nSamp,3))
# xs = np.zeros((burnIn+nSamp,SpecNumLayers))
# def neg_log_O3_Prior(delta, x):
#     L_p = (paraMat @ delta.reshape((3,1))) * L
#
#     L_p[breakInd2-1,breakInd2] = L_p[breakInd2,breakInd2+1]
#     L_p[breakInd2-1, breakInd2-1] = -(L_p[breakInd2-1, breakInd2-2] + L_p[breakInd2-1, breakInd2])
#     L_p[breakInd1-1,breakInd1] = L_p[breakInd1,breakInd1+1]
#     L_p[breakInd1-1, breakInd1-1] = -(L_p[breakInd1-1, breakInd1-2] + L_p[breakInd1-1, breakInd1])
#     return 0.5 * np.log(np.sum(paraMat @ delta.reshape((3,1) ) ) ) -0.5 * (x.T @ (L_p @ x)) - np.sum(betaD * delta)
#
#
# xs[0,:] =VMR_O3# np.random.rand(SpecNumLayers) * np.max(VMR_O3)
# deltas[0] = 0.2e-5
# for i in range(1,burnIn+nSamp):
#     x_p = VMR_O3#np.random.rand(SpecNumLayers) * np.max(VMR_O3)
#     delta_p = normal(np.mean(deltas[i-1]), 1e-2,(1,3))
#     while (delta_p < 0).any():
#         delta_p =  normal(np.mean(deltas[i-1]), 1e-4,(1,3))
#         #print((delta_p < 0).any())
#     log_MH_ratio = neg_log_O3_Prior(delta_p, x_p) - neg_log_O3_Prior(deltas[i-1], xs[i-1,:])
#     #accept or rejeict new lam_p
#     u = uniform()
#     if np.log(u) <= log_MH_ratio:
#         #accept
#         k = k + 1
#         deltas[i] = delta_p
#         xs[i,:] = x_p
#     else:
#         #rejcet
#         deltas[i] = np.copy(deltas[i-1])
#         xs[i, :] = np.copy(xs[i-1,:])
#
# print('acceptance ratio: ' + str(k/(nSamp+burnIn)))
##
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(deltas[:,0], bins = 100)
# ax1.axvline(np.median(deltas[:,0]), color = 'r')
# plt.show()
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(deltas[:,1], bins = 100)
# ax1.axvline(np.mean(deltas[:,1]), color = 'r')
# plt.show()
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(deltas[:,2], bins = 100)
# ax1.axvline(np.mean(deltas[:,2]), color = 'r')
# plt.show()

# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(VMR_O3,height_values, linewidth = 2, marker = 'o', zorder=0)
#
# for n in range(burnIn, nSamp-1,50):
#     ax1.plot(xs[n,:], height_values, linewidth=0.1, color = "gray")
#
# ax1.set_xlabel(r'Pressure in hPa ')
# ax1.set_ylabel('Height in km')

#plt.savefig('samplesPressure.png')
# plt.show()
#
## prior modeling ozone
# fit function to 3 deltas

# d1 = 4e-3
# d2 = 1e-10
# d3 = 2e-3
# breakInd1 = 5
# breakInd2 = 20
#
# paraMat = np.zeros((len(height_values), 3))
# paraMat[0:breakInd1,0] = np.ones(breakInd1)
# paraMat[breakInd1:breakInd2,1] = np.ones(breakInd2 -breakInd1)
# paraMat[breakInd2:,2] = np.ones(int(len(height_values)) -breakInd2)
#
# x = paraMat @ [d1,d2,d3]

# popt = np.polyfit(height_values[:,0], x, 2)
# ds = height_values**2 * popt[0] + height_values *popt[1]+ popt[2]
# def parabel(x, a, h0, d0):
#     return a*(x - h0)**2 + d0
#
# popt, pcov = scy.optimize.curve_fit(parabel, height_values[:,0], x)
#

# fig3, ax1 = plt.subplots()
# ax1.plot(x,height_values)
# ax1.plot(ds,height_values)
# plt.show()

def twoParabel(x, a0, h0, h1, d0, a1):
    a = np.ones(x.shape)
    a[x <= h0] = 0
    a[x >= h0] = a0
    a[x > h1] =  -a1 * d0 /(np.max(x)- h1)
    p = np.ones(x.shape)
    p[x < h1] = 2
    p[x >= h1] = 1
    d = np.ones(x.shape)
    d[x < h0] = a0 * (h0 - h1)**2 + d0
    d[x >= h0] = d0
    return a * np.power((h1-x),p )+ d

#popt, pcov = scy.optimize.curve_fit(twoParabel, height_values[:,0], x, p0 = [ 5e-6, 17, 33, 3.5e-4] )

def twoParabel(x, a0, h0, h1, a1):
    d0 = a0
    a = np.ones(x.shape)
    a[x <= h0] = 0
    a[x >= h0] = a0
    a[x > h1] =  -a1 * d0 /(np.max(x)- h1)
    p = np.ones(x.shape)
    p[x < h1] = 2
    p[x >= h1] = 1
    d = np.ones(x.shape)
    d[x < h0] = a0 * (h0 - h1)**2 + d0
    d[x >= h0] = d0
    return a * np.power((h1-x),p )+ d

def MinLogMargPost(params):#, coeff):
    tol = 1e-8
    n = SpecNumLayers
    m = SpecNumMeas
    # gamma = params[0]
    # delta = params[1]
    gam = params[0]
    a0 = params[1]
    h0 = params[2]
    h1 = params[3]
    a1 = params[4]
    #delta = params[1]
    #delta = height_values ** 2 * params[1] + height_values * params[2] + params[3]
    #delta = parabel(height_values[:,0], a, h0, d0)

    delta = twoParabel(height_values, a0, h0, h1, a1)
    #delta = params[1] * np.ones((n,1))
    if a0 < 0 or gam < 0 or a1 < 1 or a0 > 1:
    #if (delta < 0).any() or gam < 0:
        #or params[1] > 0 or params[2] < 0 or params[3] < 0:
        return np.nan
    if h0 > 20 or h0 < height_values[2,0]  or h1 > 40 or h1 < 25 :# or d0 < 1e-4:
    #if (params<0).any :
        return np.nan


    TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * delta
    TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * delta.T
    Diag = np.eye(n) * np.sum(TriU + TriL, 0)

    L_d = -TriU + Diag - TriL
    L_d[0, 0] = 2 * L_d[0, 0]
    L_d[-1, -1] = 2 * L_d[-1, -1]

    Bp = ATA + 1/gam * L_d

    #y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
    #ATy = np.matmul(A.T, y)
    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[:,0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A, L_d,  1/gam)
    F = f(ATy, y,  B_inv_A_trans_y)
    alphaD = 1.1
    #np.sum(delta)
    #return - (0.5 + alphaD - 1 )* np.sum(np.log(delta/gam)) - (m/2+1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( 1e-1 *  d0 + betaG *gam)
    return - (0.5 + alphaD - 1 ) * np.sum(np.log(delta/gam))  - (m/2+1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( 1e-1 *  np.sum(delta) + betaG *gam)




gamma01,dp0,dp1,dp2, dp3= optimize.fmin(MinLogMargPost, [gamma,5e-5, 17, 33,1e1])
# print(delta/gamma01)\
# print(delta) ,(np.var(VMR_O3) * theta_scale_O3) /gamma

#ds = height_values**2 * dp0 + height_values *dp1+ dp2

#ds = parabel(height_values,dp0,dp1,dp2)

#fig3, ax1 = plt.subplots()
# ax1.plot(x,height_values)
# ax1.plot(twoParabel(height_values[:,0],5e-6, 17, 33, 3.5e-4), height_values, color = 'k')
# ax1.plot(twoParabel(height_values,*popt),height_values, color = "g")
# ax1.plot(ds,height_values, color = "r")
# plt.show()


#draw paramter samples

#ds = height_values**2 * popt[0] + height_values *popt[1]+ popt[2]
#SetGamma = gamma01
# d1 = 4e-3
# d2 = 1e-10
# d3 = 2e-3
# breakInd1 = 5
# breakInd2 = 20
# paraMat = np.zeros((len(height_values), 3))
# paraMat[0:breakInd1,0] = np.ones(breakInd1)
# paraMat[breakInd1:breakInd2,1] = np.ones(breakInd2 -breakInd1)
# paraMat[breakInd2:,2] = np.ones(int(len(height_values)) -breakInd2)
#

# x = paraMat @ [d1,d2,d3]
# def parabel(x, a, h0, d0):
#     return a*(x - h0)**2 + d0
# # def line(x, a, d0):
# #     return a * x + d0
#
# popt, pcov = scy.optimize.curve_fit(parabel, height_values[:,0], x)
# ind = 19
# #popt, pcov = scy.optimize.curve_fit(line, height_values[:,0], np.log(x))
# # ds[0:ind] = parabel(height_values, 5e-6,35,3e-4)[0:ind]
# # ds[ind:] = parabel(height_values,6e-7,35,ds[ind-1])[ind:]
#


#twoDs = twoParabel(height_values[:,0],5e-6, 17, 33, 3.5e-4)

# def dFunc(x, h0, h1, a0, a1, d0):
#     dh = 5
#     a = np.ones(x.shape)
#     a[x <= h0] = 0
#     a[x >= h0] = a0
#     a[x >= h1] = 0
#     a[x >= h1+dh] = a1
#     h = np.ones(x.shape)
#     h[x <= h0] = 0
#     h[x >= h0] = h0
#     h[x >= h1] = 0
#     h[x >= h1+dh] = h1+dh
#     d = np.ones(x.shape)
#     d[x <= h0] = d0
#     d[x >= h0] = d0 #/ (h0 - x[0])
#     d[x >= h1] = d0 + a0 * (h1-h0)
#     d[x >= h1+dh] = d0 + a0 * (h1-h0)
#     return a * (x-h) + d


# x =paraMat @ [5e-3,8e-4,2e-3]
#popt, pcov = scy.optimize.curve_fit(twoParabel, height_values[:,0], x, p0 = [ 5e-6, 6e-7, 17, 35, ds[ind-1,0]] )
#popt, pcov = scy.optimize.curve_fit(dFunc, height_values[:,0], x, p0 = [ 17, 35,-0.1e-4,0.5e-4, 8e-4] )



def twoParabel(x, a0, h0, h1, a1):
    d0 = a0
    a = np.ones(x.shape)
    a[x <= h0] = 0
    a[x >= h0] = a0
    a[x > h1] =  -a1 * d0 /(np.max(x)- h1)
    p = np.ones(x.shape)
    p[x < h1] = 2
    p[x >= h1] = 1
    d = np.ones(x.shape)
    d[x < h0] = a0 * (h0 - h1)**2 + d0
    d[x >= h0] = d0
    return a * np.power((h1-x),p )+ d

ds = twoParabel(height_values,dp0,dp1,dp2,dp3)
twoDs = twoParabel(height_values, 1e-5, 17, 30, 5e1)
#dstry = twoParabel(height_values[:,0],*popt)0.4*lam0*gamma0
#dsTry = dFunc(height_values[:,0],15, 30,-3e-4,0.1e-4, 5e-3 )

fig3, ax1 = plt.subplots()
#ax1.plot(x,height_values)
#ax1.scatter(dsTry,height_values)
ax1.scatter(ds,height_values, color = 'r')
#ax1.plot( dFunc(height_values[:,0],15, 28,-2e-4,0.5e-4, 5e-3 ),height_values, "k")

#ax1.plot(twoParabel( height_values[:,0], *popt),height_values)
#ax1.scatter(twoDs,height_values)
plt.show()

#SetDelta = twoParabel(height_values,*popt)

#SetDelta = twoParabel( height_values[:,0], *popt)
##
def twoParabel(x, a0, h0, h1, a1):
    d0 = a0
    a = np.ones(x.shape)
    a[x <= h0] = 0
    a[x >= h0] = a0
    a[x > h1] =  -a1 * d0 /(np.max(x)- h1)
    p = np.ones(x.shape)
    p[x < h1] = 2
    p[x >= h1] = 1
    d = np.ones(x.shape)
    d[x < h0] = a0 * (h0 - h1)**2 + d0
    d[x >= h0] = d0
    return a * np.power((h1-x),p )+ d
def oneParabeltoConst(x, h0, a0, d0):
    a = np.ones(x.shape)
    a[x <= h0] = a0
    a[x >= h0] = 0#-a1
    p = np.ones(x.shape)
    p[x <= h0] = 2
    p[x > h0] = 1
    return a * np.power((h0 -x),p )+ d0

twoDs = twoParabel(height_values, 1e-5, 17, 30, 5e1)

def skew_norm_pdf(x,mean=0,w=1,skewP=0, scale = 0.1, d0 = 0):
    # adapated from:
    # http://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy
    t = (x-mean) / w
    return -2.0 * w * scy.stats.norm.pdf(t) * scy.stats.norm.cdf(skewP*t) * scale + d0

popt, pcov = scy.optimize.curve_fit(skew_norm_pdf, height_values[:,0], twoDs[:,0], p0 = [33,20,4,0.0001,1e-3] )

def simpleDFunc(x, h0, a0, d0):
    a = np.ones(x.shape)
    a[x <=h0] = - a0
    a[x > h0] = 0
    return a * (x - h0) + d0
def log_post(Params):
    tol = 1e-8
    n = SpecNumLayers
    m = SpecNumMeas
    # gamma = params[0]
    # delta = params[1]
    gam = Params[0]
    h1 = Params[1]
    a0 = Params[2]
    # h0 = Params[2]


    # mean = Params[1]
    # w = Params[2]
    # skewP = Params[3]
    # scale = Params[4]
    d0 = Params[3]
    #a1 = Params[4]
    delta = oneParabeltoConst(height_values, h1,a0, d0)
    #delta = simpleDFunc(height_values, h1,a0, d0)
    #delta = twoParabel(height_values, a0, 0, h1, 0)
    #delta = skew_norm_pdf(height_values, 16, 50, 8, 9.5e-05, 3.7e-03)
    #delta = skew_norm_pdf(height_values[:,0],mean,w,skewP, scale, d0)
    TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * delta
    TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * delta.T
    Diag = np.eye(n) * np.sum(TriU + TriL, 0)

    L_d = -TriU + Diag - TriL
    L_d[0, 0] = 2 * L_d[0, 0]
    L_d[-1, -1] = 2 * L_d[-1, -1]

    Bp = ATA + 1/gam * L_d

    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[:,0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A, L_d,  1/gam)
    F = f(ATy, y,  B_inv_A_trans_y)
    alphaD = 1
    #sigmaP = 100
    #return - (0.5 + alphaD - 1 ) * np.sum(np.log(delta/gam))  - (m/2+1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( 1e1 *  np.sum(delta) + 1e2 *gam)+ ((8 - mean)/sigmaP) ** 2 + (( 1.7e-03 - d0)/1e-3) ** 2 + (( 5 - skewP)/10) ** 2 +(( 4.2e-05 - scale)/1e-4) ** 2 +(( 50 - w)/20) ** 2
    return - (0.5 + alphaD - 1 ) * np.sum(np.log(delta/gam))  - (m/2+1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( 1e4 *  np.sum(delta)/n + betaG *gam)+ 0.5 * ((20 -Params[1])/25) ** 2 + 0.5* (( 1e-4 - Params[2])/2e-4) ** 2




def MargPostSupp(Params):
    list = []
    list.append( Params[0] > 0)
    list.append(40> Params[1] >10)  # 5.5)
    list.append( 1e-1> Params[2] > 1e-6)
    # list.append(25 > Params[2] > 10)  # 6.5)

    list.append(1> Params[3] > 0)  # 5.5)
    #list.append(1e-4> Params[4] > 0)  # 5.5)
    # list.append(1e2 > Params[4] > 0)  # 5.5)

    # list.append(50 > Params[1] > 0)
    # list.append(100 > Params[2] > 0)
    # list.append(20 > Params[3] > 1)
    # list.append(1 > Params[4] > 0)
    # list.append(1e-2 > Params[5] > 0)
    # w = Params[2]
    # scale = Params[4]
    # d0 = Params[5]
    # list.append(-2.0 * w * scy.stats.norm.pdf(0) * scy.stats.norm.cdf(0) * scale + d0 > 0)
    return all(list)


MargPost = pytwalk.pytwalk(n=4, U=log_post, Supp=MargPostSupp)
# startTime = time.time()
#x0 = np.array([gamma, 8, 50, 5, 4.2e-05,1.7e-03])
x0 = np.array([gamma,20, 1e-4, lam0 * gamma0])
xp0 = 1.01 * x0
burnIn = 1000
tWalkSampNum = 50000
MargPost.Run(T=tWalkSampNum + burnIn, x0=x0, xp0=xp0)

Samps = MargPost.Output

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.hist(Samps[:,0], bins = 50)

plt.show()

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.hist(Samps[:,1], bins = 50)
#ax1.axvline(x=popt[1], color = 'r')
plt.show()


fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.hist(Samps[:,2], bins = 50)

plt.show()


fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.hist(Samps[:,3], bins = 50)
plt.show()

# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(Samps[:,4], bins = 50)
# plt.show()

#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(Samps[:,5], bins = 50)
# plt.show()
#

##
xm = 1e-4
def normalprior(x):
    sigma =2e-4


    return 1/sigma * np.exp(-0.5 * ((x - xm)/(sigma))**2)

def expDelta(x, a,b,d0):
    # a = 4
    # b = 1e-1
    # d0 = 50
    return -x**a * np.exp(-b * x) + d0
xTry = np.linspace(0,13*(xm),100)
fig3, ax1 = plt.subplots()
ax1.scatter(xTry, normalprior(xTry) , color = 'r')
#ax1.scatter(expDelta(height_values,4,1e-1,50), height_values, color = 'r')
plt.show()

##
def Parabel(x, a0, h0, d0):

    return a0 * np.power((h0-x),2 )+ d0

#ds = twoParabel(height_values,np.mean(Samps[:,1]), 0 , np.mean(Samps[:,2]), 0)
#ds = skew_norm_pdf(height_values,*np.mean(Samps[:,1:-1],0))
twoDs = twoParabel(height_values,10e-5, 0, 20, 0)
ds = simpleDFunc(height_values,np.mean(Samps[:,1]),np.mean(Samps[:,2]),np.mean(Samps[:,3]))
ds = oneParabeltoConst(height_values,np.mean(Samps[:,1]),np.mean(Samps[:,2]),np.mean(Samps[:,3]))


paraDs = Parabel(height_values,1e-6, 32, 1e-5)


#popt, pcov = scy.optimize.curve_fit(skew_norm_pdf, height_values[:,0], twoDs[:,0], p0 = [33,20,4,0.0001,1e-3] )
def skew_norm_pdf(x,mean=0,w=1,skewP=0, scale = 0.1, d0 = 0):
    # adapated from:
    # http://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy
    t = (x-mean) / w
    return -2.0 * scy.stats.norm.pdf(t) * scy.stats.norm.cdf(skewP*t) + d0


def simpleDFunc(x, h0, a0, d0):
    a = np.ones(x.shape)
    a[x <=h0] = - a0
    a[x > h0] = a0
    return a * (x - h0) + d0
#skewDs = skew_norm_pdf(height_values,*popt)
#skewDsTry = skew_norm_pdf(height_values,16, 50, 8, 9.5e-05,4.4e-03)
skewDsTry = skew_norm_pdf(height_values,0.1, 200, 18, 1,0.787)
simpDs = simpleDFunc(height_values,30,7e-6,1e-4)
oneParaDs = oneParabeltoConst(height_values,20,1e-6,1e-5)
#popt1, pcov1 = scy.optimize.curve_fit(expDelta, height_values[:,0], twoDs[:,0], p0 = [4,1e-1,50] )
#expDs = expDelta(height_values, *popt1)
#dstry = twoParabel(height_values[:,0],*popt)0.4*lam0*gamma0
#dsTry = dFunc(height_values[:,0],15, 30,-3e-4,0.1e-4, 5e-3 )
fig3, ax1 = plt.subplots()
#ax1.plot(x,height_values)
#ax1.scatter(skewDsTry,height_values,color = 'k')
ax1.scatter(ds,height_values, color = 'r')
ax1.scatter(paraDs,height_values, color = 'r')


ax1.scatter(simpDs,height_values, color = 'y')
ax1.scatter(oneParaDs,height_values, color = 'k')
#ax1.plot( dFunc(height_values[:,0],15, 28,-2e-4,0.5e-4, 5e-3 ),height_values, "k")
#ax1.scatter(skew_norm_pdf(height_values,33,20,skewP=4, scale = 0.0001, d0 = 1e-3),height_values, color = 'k')
#ax1.scatter(skewDs,height_values, color = 'g')

#ax1.plot(twoParabel( height_values[:,0], *popt),height_values)
ax1.scatter(twoDs,height_values, color = 'b')
plt.show()


n = SpecNumLayers
m = SpecNumMeas
paraSamp = 100#n_bins
NewResults = np.zeros((paraSamp,n))
#SetDelta = skewDsTry #ds
SetGamma = gamma
randInd = np.random.randint(low=burnIn, high=tWalkSampNum+burnIn, size = paraSamp)
for p in range(paraSamp):
    SetGamma = Samps[randInd[p],0]
    #SetDelta = twoParabel(height_values,Samps[randInd[p],1], 0, Samps[randInd[p],2],0)
    #SetDelta = simpleDFunc(height_values, Samps[randInd[p], 1], Samps[randInd[p], 2],  Samps[randInd[p], 2])
    SetDelta = oneParabeltoConst(height_values, Samps[randInd[p], 1], Samps[randInd[p], 2],  Samps[randInd[p], 3])

    #SetDelta = skew_norm_pdf(height_values,*Samps[randInd[p],1:-1])
    #SetDelta = skewDsTry
    #SetDelta = simpDs
    #SetDelta = twoDs
    SetDelta = paraDs
    #SetDelta = oneParaDs
    #SetDelta = ds
    #SetDelta = expDs
    #SetDelta = lam0 * gamma0 * 0.8
    #SetDelta = twoParabel(height_values, 1e-5, 17, 30, 5e1)
    Mu = np.zeros((n,1))
    TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
    TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
    Diag = np.eye(n) * np.sum(TriU + TriL, 0)

    L_d = -TriU + Diag - TriL
    L_d[0, 0] = 2 * L_d[0, 0]
    L_d[-1, -1] = 2 * L_d[-1, -1]
    SetB = SetGamma * ATA +  L_d

    W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)) )
    v_1 = np.sqrt(SetGamma) * A.T @ W.reshape((m,1))
    W2 = np.random.multivariate_normal(np.zeros(len(L)), L_d )
    v_2 = W2.reshape((n,1))

    RandX = (SetGamma * ATy + L_d @ Mu + v_1 + v_2)
    NewResults[p,:], exitCode = gmres(SetB, RandX[0::, 0], tol=tol)

ResCol = "#1E88E5"
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
#ax1.plot(Res/theta_scale_O3, height_values, linewidth = 2.5, label = 'my guess', marker = 'o')

for p in range(0, paraSamp):
    Sol = NewResults[p, :] / theta_scale_O3
    ax1.plot(Sol, height_values, marker='+', color=ResCol, zorder=1, linewidth=0.5, markersize=5)

ax1.plot(VMR_O3, height_values, linewidth = 2.5, label = 'true profile', marker = 'o', color = "k")
O3_Prof = np.mean(NewResults,0)/ theta_scale_O3

ax1.plot(O3_Prof, height_values, marker='>', color="k", zorder=2, linewidth=0.5,
             markersize=5)
ax1.set_ylabel('Height in km')
ax1.set_xlabel('Volume Mixing Ratio of Ozone')
ax2 = ax1.twiny()
ax2.scatter(y, tang_heights_lin ,linewidth = 2, marker =  'x', label = 'data' , color = 'k')
ax2.set_xlabel(r'Spectral radiance in $\frac{W cm}{m^2  sr} $',labelpad=10)# color =dataCol,
ax1.legend()
#plt.savefig('DataStartTrueProfile.png')
plt.show()


print('bla')

## prior prediction pressure
# def pressFunc(x, b1, b2, h0, p0):
#     b = np.ones(len(x))
#     b[x > h0] = b2
#     b[x <= h0] = b1
#     return np.exp(-b * (x - h0) + np.log(p0))
#
# SpecNumMeas, SpecNumLayers  = np.shape(A)
# def log_post(Params):
#     b1 = Params[0]
#     b2 = Params[1]
#     h0 = Params[2]
#     p0 = Params[3]
#     #return gamma * np.sum((y - A @ pressFunc(x[:, 0], b1, b2, h0, p0).reshape((SpecNumLayers, 1))) ** 2) + 1e-4 * p0 + 1e-5 * h0 + 1e-5 * (b1 + b2)
#     sigmaP = 5
#     sigmaH = 30
#     sigmaGrad = 0.3
#     #return ((popt[3] - p0)/sigmaP) ** 2 + ((popt[2] - h0)/sigmaH) ** 2 + 1/sigmaGrad**2 * ((popt[0] - b1) ** 2 + (popt[1] - b2) ** 2)
#     return ((popt[3] - p0)/sigmaP) ** 2 + ((popt[2] - h0)/sigmaH) ** 2
#
# def MargPostSupp(Params):
#     list = []
#     list.append(0.3 > Params[0] > 0.15)
#     list.append(0.3 > Params[1] > 0.15)
#     list.append(Params[2] > 0)  # 6.5)
#     list.append(Params[3] > 0)  # 5.5)
#     #list.append(Params[0] > Params[1])
#     return all(list)
#
# MargPost = pytwalk.pytwalk(n=4, U=log_post, Supp=MargPostSupp)
# #startTime = time.time()
# x0 = popt * 1.5
# xp0 = 1.01 * x0
# #print(" Support of Starting points:" + str(MargPostSupp(x0)) + str(MargPostSupp(xp0)))
# nSamples = 100000
# MargPost.Run(T=nSamples + burnIn, x0=x0, xp0=xp0)
# #elapsedtWalkTime = time.time() - startTime
# #print('Elapsed Time for t-walk: ' + str(elapsedtWalkTime))
# #MargPost.Ana()
# #MargPost.SavetwalkOutput("MargPostDat.txt")
# PriorSamp = MargPost.Output
# print('finished')
# ##
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(PriorSamp[:,0], bins = 100)
# ax1.axvline(x=popt[0], color = 'r')
# plt.show()
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(PriorSamp[:,1], bins = 100)
# ax1.axvline(x=popt[1], color = 'r')
# plt.show()
#
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(PriorSamp[:,2], bins = 100)
# ax1.axvline(x=popt[2], color = 'r')
# plt.show()
#
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(PriorSamp[:,3], bins = 100)
# ax1.axvline(x=popt[3], color = 'r')
# plt.show()
#
#
#
#
# ##
#
#
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(pressure_values,height_values, linewidth = 2, marker = 'o', zorder=0)
#
# for n in range(burnIn, nSamples-1,50):
#     ax1.plot(pressFunc(height_values[:,0],PriorSamp[n,0],PriorSamp[n,1],PriorSamp[n,2],PriorSamp[n,3]), height_values, linewidth=0.1, color = "gray")
#
# ax1.set_xlabel(r'Pressure in hPa ')
# ax1.set_ylabel('Height in km')
# ax1.set_xscale('log')
# #plt.savefig('samplesPressure.png')
# plt.show()




##
def normalprior(x):
    sigma = 0.4
    xm = popt[3]
    xm = np.mean(popt[0:2])
    return 1/sigma * np.exp(-0.5 * ((x - xm)/(sigma))**2)

# def normalprior(x):
#     sigma = 0.4
#     xm = popt[3]
#     xm = np.mean(popt[0:2])
#     return 1/sigma * np.exp(-0.5 * ((x - xm)/(sigma))**2)

grad = np.log(pressure_values[1:])- np.log(pressure_values[:-1])/(height_values[1:,0]- height_values[:-1,0])

xtry = np.linspace(0,100,100)
xtry = pressure_values
xtry = np.linspace(0,1,100)
#ytry = hypprior(xtry)
ytry = normalprior(xtry)
# fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
# ax1.plot(xtry,ytry)
# plt.show()
#



##

# graph Laplacian
# direchlet boundary condition
NOfNeigh = 2#4
neigbours = np.zeros((len(height_values),NOfNeigh))

for i in range(0,len(height_values)):
    neigbours[i] = i-1, i+1

neigbours[neigbours >= len(height_values)] = np.nan
neigbours[neigbours < 0] = np.nan

L = generate_L(neigbours)

np.savetxt('GraphLaplacian.txt', L, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')

##


def pressFunc(x, b1, b2, h0, p0):
    b = np.ones(len(x))
    b[x>h0] = b2
    b[x<=h0] = b1
    return np.exp(-b * (x - h0) + np.log(p0))


##
'''do the sampling'''
SampleRounds = 100
#O3_Prof = VMR_O3
print(np.mean(VMR_O3))
SetDelta = lam0 * gamma0
SetGamma = gamma0
B0 = (ATA + lam0 * L)
B_inv_A_trans_y0, exitCode = gmres(B0, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

number_samples =1500
recov_temp_fit = temp_values#np.mean(temp_values) * np.ones((SpecNumLayers,1))
recov_press = pressure_values#np.mean(pressure_values) * np.ones((SpecNumLayers,1))#1013 * np.exp(-np.mean(grad) * height_values[:,0])
Results = np.zeros((SampleRounds, len(VMR_O3)))
TempResults = np.zeros((SampleRounds, len(VMR_O3)))
PressResults = np.zeros((SampleRounds, len(VMR_O3)))
lamRes = np.zeros(SampleRounds)
gamRes = np.zeros(SampleRounds)
round = 0
tWalkSampNum = 2500
burnInT =100
burnInMH =100

while round < SampleRounds:




    A,  theta_scale_O3 = composeAforO3(A_lin, recov_temp_fit, recov_press, ind)
    ATy = np.matmul(A.T, y)
    ATA = np.matmul(A.T, A)
    B = (ATA + SetDelta/SetGamma * L)
    B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0= B_inv_A_trans_y0, tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    #gamma0, lam0 = optimize.fmin(MinLogMargPost, [gamma, gamma * np.var(VMR_O3)])
    #print(lam0)
    #print(SetDelta/SetGamma)
    #startTime = time.time()
    SamLambdas, SamGammas, k = MHwG(number_samples, A, burnInMH, SetDelta/SetGamma, SetGamma, y, ATA, L, B_inv_A_trans_y, ATy, tol, betaG, betaD, B)
    #elapsed = time.time() - startTime
    #print('MTC Done in ' + str(elapsed) + ' s')
    #print('acceptance ratio: ' + str(k/(number_samples+burnIn)))
    SamDeltas = SamLambdas * SamGammas

    # draw paramter samples
    #paraSamp = 1  # n_bin
    MWGRand = np.random.randint(low=0, high=number_samples)
    SetGamma = SamGammas[100 + MWGRand]#, size=paraSamp)]
    SetDelta = SamDeltas[100 + MWGRand]#, size=paraSamp)][0]
    SetB = SetGamma * ATA + SetDelta * L

    W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)))
    v_1 = np.sqrt(SetGamma) * A.T @ W
    W2 = np.random.multivariate_normal(np.zeros(len(L)), L)
    v_2 = np.sqrt(SetDelta) * W2
    RandX = (SetGamma * ATy[0::, 0] + v_1 + v_2)

    Results[round, :], exitCode = gmres(SetB, RandX, x0=B_inv_A_trans_y0, tol=tol)
    O3_Prof = Results[round, :]/ theta_scale_O3
    print(np.mean(O3_Prof))
    lamRes[round] = SetDelta/SetGamma
    gamRes[round] = SetGamma
    print(SetDelta/SetGamma)

    A, theta_scale = composeAforPress(A_lin, recov_temp_fit, O3_Prof, ind)
    SampParas = tWalkPress(height_values, A, y, grad, popt, tWalkSampNum, burnInT, SetGamma)
    randInd = np.random.randint(low=0, high=tWalkSampNum)

    sampB1 = SampParas[burnIn + randInd,0]
    sampB2 = SampParas[burnIn + randInd, 1]
    sampA1 = SampParas[burnIn + randInd, 2]
    sampA2 = SampParas[burnIn + randInd, 3]

    # sampB1 = np.mean(SampParas[burnIn:,0])
    # sampB2 = np.mean(SampParas[burnIn:,1])
    # sampA1 = np.mean(SampParas[burnIn:,2])
    # sampA2 = np.mean(SampParas[burnIn:,3])
    recov_press = pressFunc(height_values[:, 0], sampB1, sampB2, sampA1, sampA2)


    # try:
    #
    #     recov_temp_fit, recov_temp = updateTemp(height_values, temp_values, recov_press)
    #
    #
    # except TypeError:
    #     recov_temp_fit = np.mean(temp_values)* np.ones((SpecNumLayers,1))
    #     print("Type Errror")

    #TempResults[round,:] = recov_temp_fit[:,0]
    PressResults[round, :] = recov_press

    #recov_temp_fit = np.mean(temp_values) * np.ones((SpecNumLayers,1))

    # fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
    # ax1.plot(recov_temp_fit, height_values, linewidth=2.5, color='r',
    #          label='fitted T')
    # ax1.scatter(recov_temp, height_values[1:,0], color='r')
    # ax1.plot(temp_values, height_values, linewidth=5, label='true T', color='green', zorder=0)
    # ax1.legend()
    # plt.savefig('TemperatureSamp.png')
    # plt.show()
    #
    # fig3, ax1 = plt.subplots(tight_layout=True, figsize=set_size(245, fraction=fraction))
    # ax1.plot(press, heights, label='true press.')
    # ax1.plot(recov_press, height_values, linewidth=2.5, label='samp. press. fit')  #
    # ax1.set_xlabel(r'Pressure in hPa ')
    # ax1.set_ylabel('Height in km')
    # ax1.legend()
    # plt.savefig('samplesPressure.png')
    # plt.show()


    #print('temp calc')


    #print(SetGamma)
    # fig3, ax1 = plt.subplots(tight_layout=True, figsize=set_size(245, fraction=fraction))
    # ax1.plot(O3_Prof, height_values, linewidth=2.5, label='my guess', marker='o')
    # ax1.plot(VMR_O3, height_values, linewidth=2.5, label='true profile', marker='o')
    # #ax1.set_ylabel('Height in km')
    # #ax1.set_xlabel('Volume Mixing Ratio of Ozone')
    # #ax2 = ax1.twiny()
    # #ax2.scatter(y, tang_heights_lin, linewidth=2, marker='x', label='data', color='k')
    # #ax2.set_xlabel(r'Spectral radiance in $\frac{W cm}{m^2  sr} $', labelpad=10)  # color =dataCol,
    # #ax1.legend()
    # #plt.savefig('DataStartTrueProfile.png')
    # plt.show()



    round += 1
    print('Round ' + str(round))

np.savetxt('lamRes.txt', lamRes, fmt = '%.15f', delimiter= '\t')
np.savetxt('gamRes.txt', gamRes, fmt = '%.15f', delimiter= '\t')
np.savetxt('VMR_O3.txt', VMR_O3, fmt = '%.15f', delimiter= '\t')
np.savetxt('O3Res.txt', Results/theta_scale_O3, fmt = '%.15f', delimiter= '\t')
np.savetxt('PressRes.txt', PressResults, fmt = '%.15f', delimiter= '\t')

##

mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)

fig, axs = plt.subplots()#figsize = (7,  2))
# We can set the number of bins with the *bins* keyword argument.
axs.hist(lamRes,bins=200, color = 'k')#int(n_bins/math.ceil(IntAutoGam)))
axs.axvline(x=lam0, color = "r", linewidth = 5)
axs.set_title('$\lambda$ samples')
#axs.set_title(str(len(new_gam)) + r' $\gamma$ samples, the noise precision')
#axs.set_xlabel(str(len(new_gam)) + ' effective $\gamma$ samples')

#tikzplotlib.save("HistoResults1.tex",axis_height='3cm', axis_width='7cm')
#plt.close()
fig, axs = plt.subplots()#figsize = (7,  2))
# We can set the number of bins with the *bins* keyword argument.
axs.hist(gamRes,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoGam)))
axs.set_title('$\gamma$ samples')
#axs.set_title(str(len(new_gam)) + r' $\gamma$ samples, the noise precision')
#axs.set_xlabel(str(len(new_gam)) + ' effective $\gamma$ samples')

#tikzplotlib.save("HistoResults1.tex",axis_height='3cm', axis_width='7cm')
#plt.close()
##
def hypprior(x):
    return x**(0) * np.exp(-x * betaD)



xtry = np.linspace(0,1e-1,100)

ytry = hypprior(xtry)

fig, axs = plt.subplots()#figsize = (7,  2))
# We can set the number of bins with the *bins* keyword argument.
axs.hist(gamRes*lamRes, color = 'k',density = True)#int(n_bins/math.ceil(IntAutoGam)))
#axs.plot(xtry,1e3*ytry, linewidth = 5)
axs.set_title('$\delta$ samples')
#axs.set_title(str(len(new_gam)) + r' $\gamma$ samples, the noise precision')
#axs.set_xlabel(str(len(new_gam)) + ' effective $\gamma$ samples')

#tikzplotlib.save("HistoResults1.tex",axis_height='3cm', axis_width='7cm')
#plt.close()

plt.show()
##
plt.close('all')
DatCol =  'gray'
ResCol = "#1E88E5"
TrueCol = [50/255,220/255, 0/255]

mpl.use(defBack)

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 10})
plt.rcParams["font.serif"] = "cmr"

fig3, ax2 = plt.subplots(figsize=set_size(245, fraction=fraction))
line3 = ax2.scatter(y, tang_heights_lin, label = r'data', zorder = 0, marker = '*', color =DatCol )#,linewidth = 5

ax1 = ax2.twiny()

ax1.plot(VMR_O3,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=1 ,linewidth = 1.5, markersize =7)

for r in range(0,SampleRounds):
    Sol = Results[r, :] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)

    ax1.plot(Sol,height_values,marker= '+',color = ResCol,label = 'posterior samples ', zorder = 0, linewidth = 0.5, markersize = 5)
    # with open('Samp' + str(n) +'.txt', 'w') as f:
    #     for k in range(0, len(Sol)):
    #         f.write('(' + str(Sol[k]) + ' , ' + str(height_values[k]) + ')')
    #         f.write('\n')
O3_Prof = np.mean(Results[0:],0)/ (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)

ax1.plot(O3_Prof, height_values, marker='>', color="k", label='posterior samples ', zorder=0, linewidth=0.5,
             markersize=5)

ax1.set_xlabel(r'Ozone volume mixing ratio ')

ax2.set_ylabel('(Tangent) Height in km')
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.set_ylim([heights[minInd-1], heights[maxInd+1]])

#ax2.set_xlabel(r'Spectral radiance in $\frac{\text{W } \text{cm}}{\text{m}^2 \text{ sr}} $',labelpad=10)# color =dataCol,
ax2.tick_params(colors = DatCol, axis = 'x')
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.spines[:].set_visible(False)
#ax2.spines['top'].set_color(pyTCol)

plt.savefig('O3Results.png')
plt.show()
##
fig3, ax1 = plt.subplots(tight_layout=True, figsize=set_size(245, fraction=fraction))
ax1.plot(press, heights, label='true press.')
#ax1.plot(recov_press, height_values, linewidth=2.5, label='samp. press. fit')  #
for n in range(0, SampleRounds):
    Sol = PressResults[n, :]

    ax1.plot(Sol, height_values, marker='+', color=ResCol, label='posterior samples ', zorder=0, linewidth=0.5,
             markersize=5)
PressProf = np.mean(PressResults[0:],0)
ax1.plot(PressProf, height_values, marker='>', color="k", label='posterior samples ', zorder=0, linewidth=0.5,
         markersize=5)

#ax1.plot(2500 * np.exp(-np.mean(grad) * height_values[:,0]),height_values[:,0])
ax1.set_xlabel(r'Pressure in hPa ')
ax1.set_ylabel('Height in km')
#ax1.legend()
plt.savefig('samplesPressure.png')
plt.show()
##
fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
# ax1.plot(recov_temp_fit, height_values, linewidth=2.5, color='r',
#          label='fitted T')
# for n in range(0, SampleRounds):
#     Sol = TempResults[n, :]
#
#     ax1.plot(Sol, height_values, marker='+', color=ResCol, label='posterior samples ', zorder=0, linewidth=0.5,
#              markersize=5)
# TempProf = np.nanmean(TempResults,0)
# ax1.plot(TempProf, height_values, marker='>', color="k", label='posterior samples ', zorder=0, linewidth=0.5,
#          markersize=5)
ax1.plot(temp_values, height_values, linewidth=5, label='true T', color='green', zorder=0)
#ax1.legend()
plt.savefig('TemperatureSamp.png')
plt.show()

#tikzplotlib.save("FirstRecRes.pgf")
print('done')


#def hypprior():
# ##
# gammatries = np.linspace(gamma0 - gamma0 ,gamma0 +gamma0,100)
# ygamtrie = np.exp(-betaG* gammatries )
#
# def hypprior(x):
#     betaG = 1e-2
#     return np.exp(-x * betaG)
# ygamtrie= hypprior(gammatries)
# delt0 = lam0 * gamma0
# deltatries = np.linspace(delt0 - delt0 ,delt0 +delt0,100)
# ydeltrie = np.exp(-betaG* deltatries )
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot( deltatries,ydeltrie )
# ax1.plot( gammatries ,ygamtrie )
# plt.show()

