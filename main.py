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

betaG = 1e-4
betaD = 1e-10  # 1e-4

""" for B_inve"""
tol = 1e-8

df = pd.read_excel('ExampleOzoneProfiles.xlsx')
#print the column names
print(df.columns)

#get the values for a given column
press = df['Pressure (hPa)'].values #in hectpascal or millibars
O3 = df['Ozone (VMR)'].values
minInd = 7
maxInd = 44
pressure_values = press[minInd:maxInd]
VMR_O3 = O3[minInd:maxInd]
scalingConstkm = 1e-3
# https://en.wikipedia.org/wiki/Pressure_altitude
# https://www.weather.gov/epz/wxcalc_pressurealtitude
heights = 145366.45 * (1 - ( press /1013.25)**0.190284 ) * 0.3048 * scalingConstkm

height_values = heights[minInd:maxInd]

""" analayse forward map without any real data values"""

MinH = height_values[0]
MaxH = height_values[-1]
R_Earth = 6371 # earth radiusin km
ObsHeight = 500 # in km

''' do svd for one specific set up for linear case and then exp case'''

#find best configuration of layers and num_meas
#so that cond(A) is not inf
#exp case first
SpecNumMeas = 70
SpecNumLayers = len(height_values)

# find minimum and max angle in radians
# min and max angle are defined by the height values of the retrived profile

MaxAng = np.arcsin((height_values[-1]+ R_Earth) / (R_Earth + ObsHeight))
MinAng = np.arcsin((height_values[0] + R_Earth) / (R_Earth + ObsHeight))


#find best configuration of layers and num_meas
#so that cond(A) is not inf
# coeff = 1/np.log(SpecNumMeas)
# meas_ang = (MinAng) + (MaxAng - MinAng) * coeff * np.log( np.linspace(1, int(SpecNumMeas) , SpecNumMeas ))
#coeff = 1/(SpecNumMeas)
#meas_ang = (MinAng) + (MaxAng - MinAng) * np.exp(- coeff *5* np.linspace(0, int(SpecNumMeas) -1 , SpecNumMeas ))
#np.flip(meas_ang)
meas_ang = np.linspace(MinAng, MaxAng, SpecNumMeas)

A_lin, tang_heights_lin, extraHeight = gen_sing_map(meas_ang,height_values,ObsHeight,R_Earth)
# fig, axs = plt.subplots(tight_layout=True)
# plt.scatter(range(len(meas_ang )),tang_heights_lin )
# plt.show()

# fig, axs = plt.subplots(tight_layout=True)
# plt.scatter(range(len(tang_heights_lin)),tang_heights_lin)
# #plt.show()

ATA_lin = np.matmul(A_lin.T,A_lin)
#condition number for A
A_lin = A_lin
A_linu, A_lins, A_linvh = np.linalg.svd(A_lin)
cond_A_lin =  np.max(A_lins)/np.min(A_lins)
print("normal: " + str(orderOfMagnitude(cond_A_lin)))



#to test that we have the same dr distances
tot_r = np.zeros(SpecNumMeas)
#calculate total length
for j in range(0, SpecNumMeas):
    tot_r[j] = (np.sqrt( ( extraHeight + R_Earth)**2 - (tang_heights_lin[j] +R_Earth )**2) )
print('Distance through layers check: ' + str(np.allclose( sum(A_lin.T), tot_r)))








##
""" compose Precision matrices for priors"""


# graph Laplacian
# direchlet boundary condition
NOfNeigh = 2#4
neigbours = np.zeros((len(height_values),NOfNeigh))
# neigbours[0] = np.nan, np.nan, 1, 2
# neigbours[-1] = len(height_values)-2, len(height_values)-3, np.nan, np.nan
# neigbours[0] = 0, 1
# neigbours[-1] = len(height_values)-2, len(height_values)-1
for i in range(0,len(height_values)):

    neigbours[i] = i - 1, i + 1
    #neigbours[i] = i-2, i-1, i+1, i+2#, i+3 i-3,


neigbours[neigbours >= len(height_values)] = np.nan
neigbours[neigbours < 0] = np.nan

P = generate_L(neigbours)
P[0,0] = NOfNeigh
P[-1,-1] = NOfNeigh
# startInd = 24
# L[startInd::, startInd::] = L[startInd::, startInd::] * 10
# L[startInd, startInd] = -L[startInd, startInd-1] - L[startInd, startInd+1] #-L[startInd, startInd-2] - L[startInd, startInd+2]
#
# #L[startInd+1, startInd+1] = -L[startInd+1, startInd+1-1] - L[startInd+1,startInd+1+1] -L[startInd+1, startInd+1-2] - L[startInd+1, startInd+1+2]
# # L[16, 16] = 13
#
# np.savetxt('GraphLaplacian.txt', L, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')
#


#taylor exapnsion for f to do so we need y (data)

''' load data and pick wavenumber/frequency'''
#
##check absoprtion coeff in different heights and different freqencies
filename = 'tropical.O3.xml'

#VMR_O3, height_values, pressure_values = testReal.get_data(filename, ObsHeight * 1e3)
#[parts if VMR_O3 * 1e6 = ppm], [m], [Pa] = [kg / (m s^2) ]\
#height_values = np.around(height_values * 1e-3,2)#in km 1e2 # in cm
#d_height = (height_values[1::] - height_values[0:-1] )
#d_height = layers[1::] - layers[0:-1]
N_A = constants.Avogadro # in mol^-1
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
R_gas = N_A * k_b_cgs # in ..cm^3

# https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
# temperature = get_temp_values(heights)
# temp_values = temperature[minInd:maxInd]

R = constants.gas_constant
L_M_b = np.array([-6.5, 0, 1, 2.8, 0, -2.8, -2])
geoPotHeight = np.array([0, 11, 20, 32, 47, 51, 71, 84.8520])
R_star = R * 1e-3
grav_prime = 9.81
M_0 = 28.9644
T_M_b = np.zeros(len(geoPotHeight))
T_M_b[0] = 288.15


for i in range(1,len(T_M_b)):
    T_M_b[i] = T_M_b[i-1]+ L_M_b[i-1] * (geoPotHeight[i] - geoPotHeight[i-1])
#height_values = np.linspace(0,84,85)
delHeightB = np.zeros((len(height_values), len(geoPotHeight) ))
SepcT_M_b = np.zeros((len(height_values), len(geoPotHeight) ))
k = 0

for i in range(0,len(height_values)):
    if geoPotHeight[k+1] > height_values[i] >= geoPotHeight[k]:
        delHeightB[i,k] =  L_M_b[k] * (height_values[i] - geoPotHeight[k])
        SepcT_M_b[i,k]=  T_M_b[k]
    else:
        k += 1
        delHeightB[i,k] = L_M_b[k] * (height_values[i] - geoPotHeight[k])
        SepcT_M_b[i,k]=  T_M_b[k]


temp_values = np.sum(SepcT_M_b + delHeightB,1).reshape((SpecNumLayers,1))


# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(temp_values, height_values )
# ax1.plot(T_M_b, geoPotHeight )
# plt.show()

InvL_M_b = (1/T_M_b[1:] - 1/T_M_b[:-1])/(geoPotHeight[1:] - geoPotHeight[:-1])

InvT_M_b = np.zeros(len(geoPotHeight))
InvT_M_b[0] = 1/288.15
for i in range(1,len(T_M_b)):
    InvT_M_b[i] = InvT_M_b[i-1] + InvL_M_b[i-1] * (geoPotHeight[i] - geoPotHeight[i-1])



InvDelHeightB = np.zeros((len(height_values), len(geoPotHeight) ))
InvSepcT_M_b = np.zeros((len(height_values), len(geoPotHeight) ))
k = 0

for i in range(0,len(height_values)):
    if geoPotHeight[k+1] > height_values[i] >= geoPotHeight[k]:
        InvDelHeightB[i,k] =  InvL_M_b[k] * (height_values[i] - geoPotHeight[k])
        InvSepcT_M_b[i,k]=  InvT_M_b[k]
    else:
        k += 1
        InvDelHeightB[i,k] = InvL_M_b[k] * (height_values[i] - geoPotHeight[k])
        InvSepcT_M_b[i,k]=  InvT_M_b[k]

InvTemp = np.sum(InvSepcT_M_b + InvDelHeightB,1).reshape((SpecNumLayers,1))


# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(InvTemp, height_values ,linewidth = 15 )
# ax1.plot(1/T_M_b, geoPotHeight, linewidth = 5 )
# ax1.plot(InvT_M_b, geoPotHeight )
# plt.show()

temp_tilde = np.sum(delHeightB,1)

temp_Mat =  np.zeros(SepcT_M_b[:,:k+1].shape)
temp_Mat[SepcT_M_b[:,:k+1] != 0] = 1

# NOfNeigh = 9
# neigbours = np.zeros((SpecNumLayers,NOfNeigh))
#
# for i in range(0,SpecNumLayers):
#
#     if 1 < i <= 8:
#         neigbours[i] = i-1, i+1 , 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
#     else:
#         neigbours[i] = i - 1, i + 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
# neigbours[0] =  0, 1 , 2, 3, 4, 5, 6, 7 ,8
# neigbours[neigbours >= SpecNumLayers] = np.nan
# neigbours[neigbours < 0] = np.nan
#
# P = generate_L(neigbours)
# #T = np.eye(k+1)
# EndInd = 9
# P[:EndInd, :EndInd] = P[:EndInd, :EndInd] * 1
# P[EndInd-1, EndInd-1] = -np.sum(P[:EndInd-1, EndInd-1]) - np.sum(P[EndInd:, EndInd-1]) #-L[startInd, startInd-2] - L[startInd, startInd+2]
# P[0,0] = -2* np.sum(P[0,1:])
# P[-1,-1] = -2* np.sum(P[-1,:-1])
#P[EndInd, EndInd] = -np.sum(P[:EndInd, EndInd]) - np.sum(P[EndInd+2:, EndInd]) #-L[startInd, startInd-2] - L[startInd, startInd+2]

#np.savetxt('TempPrec.txt', T, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')



##






#x = VMR_O3 * N_A * pressure_values /(R_gas * temp_values)#* 1e-13
#https://hitran.org/docs/definitions-and-units/
#files = '/home/lennartgolks/Python/firstModelCheck/634f1dc4.par' #/home/lennartgolks/Python /Users/lennart/PycharmProjects
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
w_cross =   f_broad * 1e-4 * np.mean(VMR_O3) * np.ones((SpecNumLayers,1))
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

""" estimate O3"""
scalingConst = 1e5
A_scal_T = pressure_values.reshape((SpecNumLayers,1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm * num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]

theta_O3 = num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]

""" estimate temperature"""
# *
A_scal_O3 = 1e2 * LineIntScal  * Source * AscalConstKmToCm * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0] * num_mole
#scalingConst = 1e11

theta_P = pressure_values.reshape((SpecNumLayers,1)) / temp_values.reshape((SpecNumLayers,1))

""" plot forward model values """
#numDensO3 =  N_A * press * 1e2 * O3 / (R * temp_values[0,:]) * 1e-6


A = A_lin * A_scal_O3.T
ATA = np.matmul(A.T,A)
Au, As, Avh = np.linalg.svd(A)
cond_A =  np.max(As)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))

#theta[0] = 0
#theta[-1] = 0
Ax = np.matmul(A, theta_P)
# b = ABase @ np.sum(InvDelHeightB,1).reshape((SpecNumLayers,1))
# Ax = ABase @ temp_Mat @ (1/T_M_b[:k+1].reshape((k+1,1))) + b

# newA = ABase @ temp_Mat
# newATA = np.matmul(newA.T,newA)
# newAu, newAs, newAvh = np.linalg.svd(newA)
# cond_newA =  np.max(newAs)/np.min(newAs)
# print("new normal: " + str(orderOfMagnitude(cond_newA)))
#
# newtheta_T = (1/T_M_b[:k+1].reshape((k+1,1)))
#convolve measurements and add noise
y = add_noise(Ax, 0.01)
np.savetxt('dataY.txt', y, header = 'Data y including noise', fmt = '%.15f')
#y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))


''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''


w_cross =   f_broad * 1e-4 * np.mean(VMR_O3) * np.ones((SpecNumLayers,1))


""" estimate O3"""
scalingConst = 1e5
A_scal_O3 = 1e2 * LineIntScal  * Source * AscalConstKmToCm  * scalingConst * S[ind,0] * num_mole/ temp_values.reshape((SpecNumLayers,1))

""" plot forward model values """

A = A_lin * A_scal_O3.T
ATA = np.matmul(A.T,A)
Au, As, Avh = np.linalg.svd(A)
cond_A =  np.max(As)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))


theta_P = pressure_values.reshape((SpecNumLayers,1)) * w_cross.reshape((SpecNumLayers,1))

# newMeanInvTemp = np.zeros((k+1,1))
# newMeanInvTemp[0] =  1/T_M_b[0]
# newAMean = newA @ newMeanInvTemp

# Asec = A_lin * A_scal_O3.T
# Axsec = np.matmul(Asec, InvTemp)
# ysecond = add_noise(Axsec, 0.01)
#
# ythird = A @ temp_Mat @ (1/T_M_b[:k+1].reshape((k+1,1))) + A @ np.sum(InvDelHeightB,1).reshape((SpecNumLayers,1))
#
# Afourth = A_lin * A_scal_T.T
# Axfourth = np.matmul(Afourth, theta_O3)
#
#
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(Ax, tang_heights_lin ,linewidth = 15 )
# ax1.plot(Axsec, tang_heights_lin ,linewidth = 10)
# ax1.plot(ythird, tang_heights_lin ,linewidth = 5)
# ax1.plot(Axfourth, tang_heights_lin ,linewidth = 2)
# ax1.plot(Ax, tang_heights_lin ,linewidth = 15 )
# ax1.plot(newAx, tang_heights_lin ,linewidth = 5 )
# ax1.plot(A_scal_T * theta_T, height_values ,linewidth = 10)
# ax1.plot(A_scal_O3 * theta_P, height_values ,linewidth = 5)
# plt.show()


meanPress = np.zeros(theta_P.shape)
#meanPress[0] = theta_P[0]
#meanPress[1] = theta_P[1]
#meanPress[5] = theta_P[5]
#y[y<=0] = 0

"""update A so that O3 profile is constant"""
# w_cross =   f_broad * 1e-4 * np.mean(VMR_O3) * np.ones((SpecNumLayers,1))
# A_scal_O3 =  1e2 * LineIntScal * Source * AscalConstKmToCm * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0] * num_mole/ temp_values
# scalingConst = 1e11
# theta_T = pressure_values.reshape((SpecNumLayers,1))
# A = A_lin * A_scal_O3.T
# ATA = np.matmul(A.T,A)

#ATy = np.matmul(A.T, y - A @ meanPress)


#np.savetxt('Forw_A_O3.txt', A, header = 'Forward Matrix A', fmt = '%.15f', delimiter= '\t')

##
"""start the mtc algo with first guesses of noise and lumping const delta"""

vari = np.zeros((len(theta_P) - 2, 1))

for j in range(1, len(theta_P) - 1):
    vari[j-1] = np.var([theta_P[j - 1], theta_P[j], theta_P[j + 1]])
ATy = np.matmul(A.T, y - A @ meanPress)
#find minimum for first guesses
'''params[1] = tau
params[0] = gamma'''
def MinLogMargPost(params):#, coeff):

    # gamma = params[0]
    # delta = params[1]
    gamma = params[0]
    lamb_tau = params[1]
    if lamb_tau < 0  or gamma < 0:
        return np.nan

    n = SpecNumLayers
    m = SpecNumMeas

    Bp = ATA + lamb_tau * P


    B_inv_A_trans_y, exitCode = gmres(Bp, ATy, tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A, P,  lamb_tau)
    F = f(ATy, y - A @ meanPress,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb_tau) - (m/2 + 1) * np.log(gamma) + 0.5 * G + 0.5 * gamma * F +  ( betaD *  lamb_tau * gamma + betaG *gamma)

#minimum = optimize.fmin(MargPostU, [5e-5,0.5])
minimum = optimize.fmin(MinLogMargPost, [1/(np.max(Ax) * 0.01),1/(np.mean(vari))*(np.max(Ax) * 0.01)])

lam_tau_0 = minimum[1]
print('lambda_0: ' )
print('{:.1e}'.format(lam_tau_0))
B_0 = ATA + lam_tau_0 * P

B_inv_A_trans_y0, exitCode = gmres(B_0, ATy, tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

Bu, Bs, Bvh = np.linalg.svd(B_0)
cond_B =  np.max(Bs)/np.min(Bs)
print("Condition number B: " + str(orderOfMagnitude(cond_B)))



##
# """ finally calc f and g with a linear solver adn certain lambdas
#  using the gmres only do to check if taylor nth expansion is enough"""
#
# lam= np.logspace(-5,15,500)
# f_func = np.zeros(len(lam))
# g_func = np.zeros(len(lam))
#
#
#
# for j in range(len(lam)):
#
#     B = (ATA + lam[j] * T)
#
#     B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0=B_inv_A_trans_y0, tol=tol, restart=25)
#     #print(exitCode)
#
#     CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)
#     if np.linalg.norm(ATy[0::, 0]- CheckB_inv_ATy)/np.linalg.norm(ATy[0::, 0])<=tol:
#         f_func[j] = f(ATy, y, B_inv_A_trans_y)
#     else:
#         f_func[j] = np.nan
#
#     g_func[j] = g(A, T, lam[j])
#
#
# np.savetxt('f_func.txt', f_func, fmt = '%.15f')
# np.savetxt('g_func.txt', g_func, fmt = '%.15f')
# np.savetxt('lam.txt', lam, fmt = '%.15f')


##


B_inv_T = np.zeros(np.shape(B_0))

for i in range(len(B_0)):
    B_inv_T[:, i], exitCode = gmres(B_0, P[:, i], tol=tol, restart=25)
    if exitCode != 0:
        print('B_inv_L ' + str(exitCode))

#relative_tol_L = tol
#CheckB_inv_L = np.matmul(B, B_inv_L)
#print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)

B_inv_T_2 = np.matmul(B_inv_T, B_inv_T)
B_inv_T_3 = np.matmul(B_inv_T_2, B_inv_T)
B_inv_T_4 = np.matmul(B_inv_T_2, B_inv_T_2)
B_inv_T_5 = np.matmul(B_inv_T_4, B_inv_T)
B_inv_T_6 = np.matmul(B_inv_T_4, B_inv_T_2)


f_0_1 = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_T), B_inv_A_trans_y0)
f_0_2 = -1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_T_2), B_inv_A_trans_y0)
f_0_3 = 1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_T_3) ,B_inv_A_trans_y0)
f_0_4 = -1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_T_4) ,B_inv_A_trans_y0)
#f_0_5 = 120 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y)


g_0_1 = np.trace(B_inv_T)
g_0_2 = -1 / 2 * np.trace(B_inv_T_2)
g_0_3 = 1 /6 * np.trace(B_inv_T_3)
g_0_4 = -1 /24 * np.trace(B_inv_T_4)
g_0_5 = 0#1 /120 * np.trace(B_inv_L_5)
g_0_6 = 0#1 /720 * np.trace(B_inv_L_6)




##
'''do the sampling'''


number_samples = 10000
#inintialize sample

wLam = 0.8e-2
startTime = time.time()
lambdas, gammas, accepted = MHwG(number_samples, SpecNumMeas, SpecNumLayers, burnIn, lam_tau_0 , minimum[0], wLam, y - A @ meanPress, ATA, P, B_inv_A_trans_y0, ATy, tol, betaG, betaD, f_0_1, f_0_2, f_0_3, g_0_1, g_0_2, g_0_3)
elapsed = time.time() - startTime
print('MTC Done in ' + str(elapsed) + ' s')

print('acceptance ratio: ' + str(accepted /(number_samples+burnIn)))
deltas = lambdas * gammas
np.savetxt('samplesT.txt', np.vstack((gammas[burnIn::], deltas[burnIn::], lambdas[burnIn::])).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsed), fmt = '%.15f \t %.15f \t %.15f')


# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.Run_Autocorr_Ana_MTC_for_T(nargout=0)
# eng.quit()
#
#
# AutoCorrData = np.loadtxt("auto_corr_dat.txt", skiprows=3, dtype='float')
# #IntAutoLam, IntAutoGam , IntAutoDelt = np.loadtxt("auto_corr_dat.txt",userow = 1, skiprows=1, dtype='float'
#
# with open("auto_corr_dat.txt") as fID:
#     for n, line in enumerate(fID):
#        if n == 1:
#             IntAutoDelt, IntAutoGam, IntAutoLam = [float(IAuto) for IAuto in line.split()]
#             break
#


#refine according to autocorrelation time
new_lamb = lambdas#[burnIn::math.ceil(IntAutoLam)]

new_gam = gammas#[burnIn::math.ceil(IntAutoGam)]

new_delt = deltas#[burnIn::math.ceil(IntAutoDelt)]



fig, axs = plt.subplots(3, 1,tight_layout=True)
# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(new_gam,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoGam)))
#axs[0].set_title(str(len(new_gam)) + ' effective $\gamma$ samples')
axs[0].set_title(str(len(new_gam)) + r' $\gamma$ samples, the noise precision')
axs[1].hist(new_delt,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoDelt)))
axs[1].set_title(str(len(new_delt)) + ' $\delta$ samples, the prior precision')
axs[2].hist(new_lamb,bins=n_bins, color = 'k')#10)
#axs[2].xaxis.set_major_formatter(scientific_formatter)
#axs[2].set_title(str(len(new_lamb)) + ' effective $\lambda =\delta / \gamma$ samples')
axs[2].set_title(str(len(new_lamb)) + ' $\lambda$ samples, the regularization parameter')
plt.savefig('HistoResults.png')
plt.show()


##

#draw temperatur samples
paraSamp = 100#n_bins
Results = np.zeros((paraSamp,len(theta_P)))
NormRes = np.zeros(paraSamp)
xTLxRes = np.zeros(paraSamp)
SetGammas = new_gam[np.random.randint(low=0, high=len(new_gam), size=paraSamp)]
SetDeltas  = new_delt[np.random.randint(low=0, high=len(new_delt), size=paraSamp)]
ATy= np.matmul(A.T, y)
startTimeX = time.time()
for p in range(paraSamp):
    # SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
    SetGamma =SetGammas[p] #np.mean(new_gam)# minimum[0]#SetGammas[p] #
    SetDelta  =SetDeltas[p] #np.mean(new_delt)#minimum[1] * minimum[0]#SetDeltas[p] #
    W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)))
    v_1 = np.sqrt(SetGamma) *  (A.T @ W).reshape((SpecNumLayers,1))
    W2 = np.random.multivariate_normal(np.zeros(len(P)), P)
    v_2 = np.sqrt(SetDelta) * W2.reshape((SpecNumLayers,1))

    SetB = SetGamma * ATA + SetDelta * P
    RandX = (SetGamma * ATy + SetDelta * P @ meanPress + v_1 + v_2)

    # SetB_inv = np.zeros(np.shape(SetB))
    # for i in range(len(SetB)):
    #     e = np.zeros(len(SetB))
    #     e[i] = 1
    #     SetB_inv[:, i], exitCode = gmres(SetB, e, tol=tol, restart=25)
    #     if exitCode != 0:
    #         print(exitCode)

    B_inv_A_trans_y, exitCode = gmres(SetB, RandX, x0=B_inv_A_trans_y0, tol=tol)

    # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    #CheckB_inv = np.matmul(SetB, SetB_inv)
    #print(np.linalg.norm(np.eye(len(SetB)) - CheckB_inv) / np.linalg.norm(np.eye(len(SetB))) < tol)

    Results[p, :] = B_inv_A_trans_y #- meanPress.reshape((1,SpecNumLayers))

elapsedX = time.time() - startTimeX
print('Time to solve for x ' + str(elapsedX/paraSamp))

##
# BinHist = 200#n_bins
# lambHist, lambBinEdges = np.histogram(new_lamb, bins= BinHist, density =True)
# #gamHist, gamBinEdges = np.histogram(new_gam, bins= BinHist)
#
# MargResults = np.zeros((BinHist,len(theta_T)))
# MargVarResults = np.zeros((BinHist,len(theta_T)))
# #MargResults = np.zeros((BinHist,BinHist,len(theta)))
# #LamMean = 0
#
# for p in range(BinHist):
#     #DLambda = ( lambBinEdges[p+1] - lambBinEdges[p])/2
#     SetLambda =  lambBinEdges[p]
#     #LamMean = LamMean + SetLambda * lambHist[p]/sum(lambHist)
#     SetB = ATA + SetLambda * T
#
#     B_inv_A_trans_y, exitCode = gmres(SetB, ATy[0::, 0], x0=B_inv_A_trans_y0, tol=tol)
#
#     # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
#     if exitCode != 0:
#         print(exitCode)
#
#     MargResults[p, :] = B_inv_A_trans_y * lambHist[p]/ np.sum(lambHist)
#     MargVarResults[p, :] = (B_inv_A_trans_y)**2 * lambHist[p]/ np.sum(lambHist)
#
#
# trapezMat = 2 * np.ones(MargResults.shape)
# trapezMat[:,0] = 1
# trapezMat[:,-1] = 1
# MargInteg = 0.5 * np.sum(MargResults * trapezMat , 0) #* (lambBinEdges[1]- lambBinEdges[0] )
#
# MargIntegSq = 0.5 * np.sum(MargVarResults * trapezMat , 0)
#
# MargX =   (MargInteg )
# MargXErr =  np.sqrt( (MargIntegSq - MargInteg**2 ) )



##

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 12})
mpl.use(defBack)


ResCol = "#1E88E5"#"#0072B2"
MeanCol = "#FFC107"#"#d62728"
RegCol = "#D81B60"#"#D55E00"
TrueCol = "#004D40" #'k'
DatCol = 'k'#"#332288"#"#009E73"



fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
#line1 = ax1.plot(theta_P,height_values, color = TrueCol, linewidth = 7, label = 'True Temperatur', zorder=0)
#line1 = ax1.plot(  temp_values.reshape(pressure_values.shape) ,height_values, color = TrueCol, linewidth = 7, label = 'True Temperatur', zorder=0)
line1 = ax1.plot(  theta_P,height_values, color = TrueCol, linewidth = 7, label = 'True Temperatur', zorder=0)

#ax1.plot(Sol,height_values)
for n in range(0,paraSamp,4):
    Sol = Results[n, :]
    ax1.plot(Sol,height_values, linewidth = .5, color = ResCol )
#ax1.errorbar(MargX,height_values, color = MeanCol, capsize=4, yerr = np.zeros(len(height_values)), fmt = '-x', label = r'MTC E$_{\mathbf{x},\mathbf{\theta}| \mathbf{y}}[h(\mathbf{x})]$')
#ax1.plot(temp_Mat @ Results[99, :].reshape((k+1,1)) + np.sum(InvDelHeightB,1).reshape((SpecNumLayers,1)) ,height_values, linewidth = .5, color = ResCol )
#ax1.plot(1/ (temp_Mat @ np.mean(Results,0).reshape((k+1,1))) ,height_values, linewidth = 1, color = "k" , zorder = 3)

ax1.set_xlabel(r'Ozone volume mixing ratio ')

ax1.set_ylabel('Height in km')

fig3.savefig('TrueRecocovRTOData.png')#, dpi = dpi)

plt.show()

print("bla")



## now retrive temperature given the pressure


newA = A_lin * A_scal_T.T
newATA = np.matmul(newA.T,newA)
newAu, newAs, newAvh = np.linalg.svd(newA)
cond_newA =  np.max(newAs)/np.min(newAs)
print("new normal: " + str(orderOfMagnitude(cond_newA)))

theta_T = 1/temp_values

newAx = np.matmul(newA , theta_T)

meanTemp = np.zeros(theta_T.shape)
#meanTemp[-1] = theta_T[-1]

ATy = np.matmul(newA.T, y - newA @ meanTemp)

# graph Laplacian
# direchlet boundary condition
NOfNeigh = 2#4
neigbours = np.zeros((len(height_values),NOfNeigh))

for i in range(0,len(height_values)):

    neigbours[i] = i - 1, i + 1
    #neigbours[i] = i-2, i-1, i+1, i+2#, i+3 i-3,


neigbours[neigbours >= len(height_values)] = np.nan
neigbours[neigbours < 0] = np.nan

T = generate_L(neigbours)
startInd = 24
T[startInd::, startInd::] = T[startInd::, startInd::] * 10
T[startInd, startInd] = -T[startInd, startInd-1] - T[startInd, startInd+1] #-L[startInd, startInd-2] - L[startInd, startInd+2]

# #L[startInd+1, startInd+1] = -L[startInd+1, startInd+1-1] - L[startInd+1,startInd+1+1] -L[startInd+1, startInd+1-2] - L[startInd+1, startInd+1+2]
# # L[16, 16] = 13
#
# np.savetxt('GraphLaplacian.txt', L, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')




vari = np.zeros((len(theta_T) - 2, 1))

for j in range(1, len(theta_T) - 1):
    vari[j-1] = np.var([theta_T[j - 1], theta_T[j], theta_T[j + 1]])


#find minimum for first guesses
'''params[1] = tau
params[0] = gamma'''
def MinLogMargPostTemp(params):#, coeff):

    # gamma = params[0]
    # delta = params[1]
    gamma = params[0]
    lamb_tau = params[1]
    if lamb_tau < 0  or gamma < 0:
        return np.nan

    n = SpecNumLayers
    m = SpecNumMeas

    Bp = newATA + lamb_tau * T


    B_inv_A_trans_y, exitCode = gmres(Bp, ATy, tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(newA, T,  lamb_tau)
    F = f(ATy, y - newA @ meanTemp,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb_tau) - (m/2 + 1) * np.log(gamma) + 0.5 * G + 0.5 * gamma * F +  ( betaD *  lamb_tau * gamma + betaG *gamma)




minimum = optimize.fmin(MinLogMargPostTemp, [1/(np.max(newAx) * 0.01),1/(np.mean(vari))*(np.max(newAx) * 0.01)])




lam_tau_0 = minimum[1]
print('lambda_0: ' )
print('{:.1e}'.format(lam_tau_0))
B_0 = newATA + lam_tau_0 * T

B_inv_A_trans_y0, exitCode = gmres(B_0, ATy, tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

Bu, Bs, Bvh = np.linalg.svd(B_0)
cond_B =  np.max(Bs)/np.min(Bs)
print("Condition number B: " + str(orderOfMagnitude(cond_B)))






##


B_inv_T = np.zeros(np.shape(B_0))

for i in range(len(B_0)):
    B_inv_T[:, i], exitCode = gmres(B_0, T[:, i], tol=tol, restart=25)
    if exitCode != 0:
        print('B_inv_L ' + str(exitCode))

#relative_tol_L = tol
#CheckB_inv_L = np.matmul(B, B_inv_L)
#print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)

B_inv_T_2 = np.matmul(B_inv_T, B_inv_T)
B_inv_T_3 = np.matmul(B_inv_T_2, B_inv_T)
B_inv_T_4 = np.matmul(B_inv_T_2, B_inv_T_2)
B_inv_T_5 = np.matmul(B_inv_T_4, B_inv_T)
B_inv_T_6 = np.matmul(B_inv_T_4, B_inv_T_2)


f_0_1 = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_T), B_inv_A_trans_y0)
f_0_2 = -1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_T_2), B_inv_A_trans_y0)
f_0_3 = 1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_T_3) ,B_inv_A_trans_y0)
f_0_4 = -1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_T_4) ,B_inv_A_trans_y0)
#f_0_5 = 120 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y)


g_0_1 = np.trace(B_inv_T)
g_0_2 = -1 / 2 * np.trace(B_inv_T_2)
g_0_3 = 1 /6 * np.trace(B_inv_T_3)
g_0_4 = -1 /24 * np.trace(B_inv_T_4)
g_0_5 = 0#1 /120 * np.trace(B_inv_L_5)
g_0_6 = 0#1 /720 * np.trace(B_inv_L_6)




##
'''do the sampling for temperatur'''


number_samples = 10000
#inintialize sample

tempWLam = 1.5e5
startTime = time.time()
tempLambdas, tempGammas, tempAccepted = MHwG(number_samples, SpecNumMeas, SpecNumLayers, burnIn, lam_tau_0 , minimum[0], tempWLam, y - newA @ meanTemp, newATA, T, B_inv_A_trans_y0, ATy, tol, betaG, betaD, f_0_1, f_0_2, f_0_3, g_0_1, g_0_2, g_0_3)
elapsed = time.time() - startTime
print('MTC Done in ' + str(elapsed) + ' s')

print('acceptance ratio: ' + str(accepted /(number_samples+burnIn)))
tempDeltas = tempLambdas * tempGammas
np.savetxt('samplesT.txt', np.vstack((tempGammas[burnIn::], tempDeltas[burnIn::], tempLambdas[burnIn::])).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsed), fmt = '%.15f \t %.15f \t %.15f')


# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.Run_Autocorr_Ana_MTC_for_T(nargout=0)
# eng.quit()
#
#
# AutoCorrData = np.loadtxt("auto_corr_dat_temp.txt", skiprows=3, dtype='float')
# #IntAutoLam, IntAutoGam , IntAutoDelt = np.loadtxt("auto_corr_dat.txt",userow = 1, skiprows=1, dtype='float'
#
# with open("auto_corr_dat_temp.txt") as fID:
#     for n, line in enumerate(fID):
#        if n == 1:
#             tempIntAutoDelt, tempIntAutoGam, tempIntAutoLam = [float(IAuto) for IAuto in line.split()]
#             break



#refine according to autocorrelation time
new_tempLamb = tempLambdas#[burnIn::math.ceil(tempIntAutoLam)]

new_tempGam = tempGammas#[burnIn::math.ceil(tempIntAutoGam)]

new_tempDelt = tempDeltas#[burnIn::math.ceil(tempIntAutoDelt)]



fig, axs = plt.subplots(3, 1,tight_layout=True)
# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(new_tempGam,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoGam)))
#axs[0].set_title(str(len(new_gam)) + ' effective $\gamma$ samples')
axs[0].set_title(str(len(new_tempGam)) + r' $\gamma$ samples, the noise precision')
axs[1].hist(new_tempDelt,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoDelt)))
axs[1].set_title(str(len(new_tempDelt)) + ' $\delta$ samples, the prior precision')
axs[2].hist(new_tempLamb,bins=n_bins, color = 'k')#10)
#axs[2].xaxis.set_major_formatter(scientific_formatter)
#axs[2].set_title(str(len(new_lamb)) + ' effective $\lambda =\delta / \gamma$ samples')
axs[2].set_title(str(len(new_tempLamb)) + ' $\lambda$ samples, the regularization parameter')
plt.savefig('HistoResultsTemp.png')
plt.show()






##

#draw temperatur samples
paraSamp = 100#n_bins
TempResults = np.zeros((paraSamp,len(theta_T)))

SetGammas = new_tempGam[np.random.randint(low=0, high=len(new_tempGam), size=paraSamp)]
SetDeltas  = new_tempDelt[np.random.randint(low=0, high=len(new_tempDelt), size=paraSamp)]
ATy= np.matmul(newA.T, y)
startTimeX = time.time()
for p in range(paraSamp):
    # SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
    SetGamma =SetGammas[p] #np.mean(new_gam)# minimum[0]#SetGammas[p] #
    SetDelta  =SetDeltas[p] #np.mean(new_delt)#minimum[1] * minimum[0]#SetDeltas[p] #
    W = np.random.multivariate_normal(np.zeros(len(newA)), np.eye(len(newA)))
    v_1 = np.sqrt(SetGamma) *  (newA.T @ W).reshape((SpecNumLayers,1))
    W2 = np.random.multivariate_normal(np.zeros(len(T)), T)
    v_2 = np.sqrt(SetDelta) * W2.reshape((SpecNumLayers,1))

    SetB = SetGamma * newATA + SetDelta * T
    RandX = (SetGamma * ATy + SetDelta * T @ meanTemp + v_1 + v_2)

    # SetB_inv = np.zeros(np.shape(SetB))
    # for i in range(len(SetB)):
    #     e = np.zeros(len(SetB))
    #     e[i] = 1
    #     SetB_inv[:, i], exitCode = gmres(SetB, e, tol=tol, restart=25)
    #     if exitCode != 0:
    #         print(exitCode)

    B_inv_A_trans_y, exitCode = gmres(SetB, RandX, x0=B_inv_A_trans_y0, tol=tol)

    # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    #CheckB_inv = np.matmul(SetB, SetB_inv)
    #print(np.linalg.norm(np.eye(len(SetB)) - CheckB_inv) / np.linalg.norm(np.eye(len(SetB))) < tol)

    Results[p, :] = B_inv_A_trans_y #- meanTemp.reshape((1,SpecNumLayers))

elapsedX = time.time() - startTimeX
print('Time to solve for x ' + str(elapsedX/paraSamp))


##

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 12})
mpl.use(defBack)


ResCol = "#1E88E5"#"#0072B2"
MeanCol = "#FFC107"#"#d62728"
RegCol = "#D81B60"#"#D55E00"
TrueCol = "#004D40" #'k'
DatCol = 'k'#"#332288"#"#009E73"



fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
#line1 = ax1.plot( pressure_values.reshape((SpecNumLayers,1))/temp_values ,height_values, color = TrueCol, linewidth = 7, label = 'True Temperatur', zorder=0)
line1 = ax1.plot(  temp_values ,height_values, color = TrueCol, linewidth = 7, label = 'True Temperatur', zorder=0)
#line1 = ax1.plot(  VMR_O3 ,height_values, color = TrueCol, linewidth = 7, label = 'True Temperatur', zorder=0)

#ax1.plot(Sol,height_values)
for n in range(0,paraSamp,4):
    Sol = Results[n, :]
    ax1.plot(1/Sol ,height_values, linewidth = .5, color = ResCol )
#ax1.errorbar(MargX,height_values, color = MeanCol, capsize=4, yerr = np.zeros(len(height_values)), fmt = '-x', label = r'MTC E$_{\mathbf{x},\mathbf{\theta}| \mathbf{y}}[h(\mathbf{x})]$')
#ax1.plot(temp_Mat @ Results[99, :].reshape((k+1,1)) + np.sum(InvDelHeightB,1).reshape((SpecNumLayers,1)) ,height_values, linewidth = .5, color = ResCol )
#ax1.plot(1/ (temp_Mat @ np.mean(Results,0).reshape((k+1,1))) ,height_values, linewidth = 1, color = "k" , zorder = 3)

ax1.set_xlabel(r'Ozone volume mixing ratio ')

ax1.set_ylabel('Height in km')

fig3.savefig('TrueRecocovRTOData.png')#, dpi = dpi)

plt.show()

print('bla')

##
grav = 9.81 * ((R_Earth)/(R_Earth + height_values[1:]))**2

del_height = height_values[1:] - height_values[:-1]

del_log_pres = np.log(pressure_values[1:]) - np.log(pressure_values[:-1])

recov_temp = - 28.97 * grav * del_height  / ( R * del_log_pres )


L_M_b = np.array([-6.5, 0, 1, 2.8, 0, -2.8, -2])
geoPotHeight = np.array([0, 11, 20, 32, 47, 51, 71, 84.8520])
R_star = R * 1e-3
grav_prime = 9.81
M_0 = 28.9644
T_m_b = np.zeros(len(geoPotHeight))
T_m_b[0] = 288.15
P_m_b = np.zeros(len(geoPotHeight))
P_m_b[0] = np.log(101325)

for i in range(1,len(T_m_b)):

    T_m_b[i] = T_m_b[i-1]+ L_M_b[i-1] * (geoPotHeight[i] - geoPotHeight[i-1])
    #print(L_M_b[i-1])
    if L_M_b[i-1] != 0:

        Pexpo = grav_prime * M_0 / (R_star * L_M_b[i - 1])

        #print(T_m_b[i-1])
        P_m_b[i] = P_m_b[i-1] + np.log(T_m_b[i-1]/(T_m_b[i-1] + L_M_b[i-1] * (geoPotHeight[i] - geoPotHeight[i-1]))) * Pexpo
    else:
        #print(P_m_b[i-1])
        P_m_b[i] = P_m_b[i-1] - (grav_prime * M_0 *  (geoPotHeight[i] - geoPotHeight[i-1]) / R_star / T_m_b[i-1])

SecPres = pressure_values[0] * np.exp(-28.97 * grav.reshape((36,1)) / temp_values[:-1].reshape((36,1)) / R * height_values[:-1].reshape((36,1)) )

recov_temp2 = del_height[1:].reshape((35,1)) / np.log(SecPres[:-1] / SecPres[1:]) / R /grav[1:].reshape((35,1)) * 28.97
recov_temp3 =  height_values[:-1].reshape((36,1)) / np.log(pressure_values[1:].reshape((36,1)) / pressure_values[0]) / R /grav.reshape((36,1)) * 28.97

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))

#ax1.plot(recov_temp, height_values[1:])
#ax1.plot(recov_temp2, height_values[1:-1])
#ax1.plot(del_height, height_values[:-1])
ax1.plot(pressure_values, height_values)
#ax1.plot(recov_temp3, height_values[:-1])

#ax1.plot(temp_values, height_values)
ax1.plot(SecPres, height_values[:-1])
#ax1.plot(T_m_b, geoPotHeight )
#ax1.plot(P_m_b, geoPotHeight )
#ax1.plot(grav , height_values[1:])
plt.show()