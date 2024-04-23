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

minInd = 5
maxInd = 42
pressure_values = press[minInd:maxInd]
VMR_O3 = O3[minInd:maxInd]
scalingConstkm = 1e-3
# https://en.wikipedia.org/wiki/Pressure_altitude
# https://www.weather.gov/epz/wxcalc_pressurealtitude
#heights = 145366.45 * (1 - ( press /1013.25)**0.190284 ) * 0.3048 * scalingConstkm

def height_to_pressure(p0, x, dx):
    R = constants.gas_constant
    R_Earth = 6371  # earth radiusin km
    grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
    temp = get_temp(x)
    return p0 * np.exp(-28.97 * grav / temp / R * dx  )

calc_press = np.zeros((len(press)+1,1))
calc_press[0] = 1013.25
calc_press[1:] = press.reshape((len(press),1)) #hPa
try_heights = np.linspace(0,155,10000)
actual_heights = np.zeros((len(press)+1,1))

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
            else:
                actual_heights[i] = try_heights[j]
            #k = j
            break

print('got heights')

heights = actual_heights[1:]
height_values = heights[minInd:maxInd].reshape(maxInd-minInd)
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





#taylor exapnsion for f to do so we need y (data)

''' load data and pick wavenumber/frequency'''
#check absoprtion coeff in different heights and different freqencies
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


# L_M_b = np.array([-6.5, 0, 1, 2.8, 0, -2.8, -2])
# geoPotHeight = np.array([0, 11, 20, 32, 47, 51, 71, 84.8520])
# R_star = R * 1e-3
# grav_prime = 9.81
# M_0 = 28.9644
# T_M_b = np.zeros(len(geoPotHeight))
# T_M_b[0] = 288.15
#
#
# for i in range(1,len(T_M_b)):
#     T_M_b[i] = T_M_b[i-1]+ L_M_b[i-1] * (geoPotHeight[i] - geoPotHeight[i-1])
# #height_values = np.linspace(0,84,85)
# delHeightB = np.zeros((len(height_values), len(geoPotHeight) ))
# SepcT_M_b = np.zeros((len(height_values), len(geoPotHeight) ))
# k = 0
#
# for i in range(0,len(height_values)):
#     if geoPotHeight[k+1] > height_values[i] >= geoPotHeight[k]:
#         delHeightB[i,k] =  L_M_b[k] * (height_values[i] - geoPotHeight[k])
#         SepcT_M_b[i,k]=  T_M_b[k]
#     else:
#         k += 1
#         delHeightB[i,k] = L_M_b[k] * (height_values[i] - geoPotHeight[k])
#         SepcT_M_b[i,k]=  T_M_b[k]
#
#
# temp_values = np.sum(SepcT_M_b + delHeightB,1).reshape((SpecNumLayers,1))


# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(get_temp_values(height_values), height_values )
# #ax1.plot(T_M_b, geoPotHeight )
# plt.show()
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(np.log(calc_press), actual_heights )
# plt.show()
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(O3, actual_heights[1:] )
# plt.show()
#
# k =13
# z = 5
# del_height = actual_heights[z+1:-k] - actual_heights[z:-(k+1)]
# grav = 9.81 * ((R_Earth)/(R_Earth + actual_heights[z:-k]))**2
# calc_temp  = np.zeros((len(del_height),1))
#
# for i in range(0, len(del_height)):
#
#     calc_temp[i] =  -28.97 * grav[i] / np.log(calc_press[z+i] / calc_press[z+i-1]) / R * del_height[i]
#
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.scatter(calc_temp, actual_heights[z+1:-k] )
# ax1.plot(get_temp_values(actual_heights[z+1:-k]), actual_heights[z+1:-k] )
# plt.show()
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(O3[z:-k], actual_heights[z+1:-k] )
# ax2 = ax1.twiny()
# ax2.plot(press[z:-k], actual_heights[z+1:-k] )
# plt.show()
#
#
# print('plotted')

# ##
# ''' fitting to pressure'''
#
#
# #efit, dfit, cfit, bfit, afit = np.polyfit(actual_heights[1:].reshape(len(press)), np.log(press), 4)
#
# efit, dfit, cfit, bfit, afit = np.polyfit(actual_heights[z:-k].reshape(len(calc_press[z:-k])), np.log(calc_press[z:-k]), 4)
#
# def press(a,b,c,d,e,x):
#     #a[0] = pressure_values[0]*1.75e1
#     return np.exp( e * x**4 + d * x**3 + c * x**2 + b * x + a)
#
# fit_press = press(afit, bfit, cfit, dfit,efit, actual_heights[z:-k])
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(calc_press, actual_heights )
# ax1.plot(fit_press, actual_heights[z:-k])
# plt.show()
#
# 'calc temp with fitted press'
#
# calc_fit_temp  = np.zeros((len(del_height),1))
#
# for i in range(0, len(del_height)):
#
#     calc_fit_temp[i] =  -28.97 * grav[i] / np.log(fit_press[i] / fit_press[i-1]) / R * del_height[i]
#
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.scatter(calc_fit_temp, actual_heights[z+1:-k] )
# ax1.plot(get_temp_values(actual_heights[z+1:-k]), actual_heights[z+1:-k] )
# plt.show()


#print('fitted')
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
w_cross =   f_broad * 1e-4 * VMR_O3 #np.mean(VMR_O3) * np.ones((SpecNumLayers,1))
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
# A_scal_T = pressure_values.reshape((SpecNumLayers,1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm * num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]
#
# theta_O3 = num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]

""" estimate temperature"""
# *
A_scal_O3 = 1e2 * LineIntScal  * Source * AscalConstKmToCm * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0] * num_mole / temp_values.reshape((SpecNumLayers,1))
#scalingConst = 1e11

theta_P = pressure_values.reshape((SpecNumLayers,1))

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

gamma = 1/(np.max(Ax) * 0.01)**2

''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''


# w_cross =   f_broad * 1e-4 * np.mean(VMR_O3) * np.ones((SpecNumLayers,1))
#
#
# """ estimate O3"""
# scalingConst = 1e5
# A_scal_O3 = 1e2 * LineIntScal  * Source * AscalConstKmToCm  * scalingConst * S[ind,0] * num_mole/ temp_values.reshape((SpecNumLayers,1))

""" plot forward model values """

# A = A_lin * A_scal_O3.T
# ATA = np.matmul(A.T,A)
# Au, As, Avh = np.linalg.svd(A)
# cond_A =  np.max(As)/np.min(As)
# print("normal: " + str(orderOfMagnitude(cond_A)))
#
# ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
# cond_ATA = np.max(ATAs)/np.min(ATAs)
# print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))
#
#
# theta_P = pressure_values.reshape((SpecNumLayers,1)) * w_cross.reshape((SpecNumLayers,1))

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

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin ,linewidth = 15 )
ax1.plot(y, tang_heights_lin ,linewidth = 15 )
# ax1.plot(Axsec, tang_heights_lin ,linewidth = 10)
# ax1.plot(ythird, tang_heights_lin ,linewidth = 5)
# ax1.plot(Axfourth, tang_heights_lin ,linewidth = 2)
# ax1.plot(Ax, tang_heights_lin ,linewidth = 15 )
# ax1.plot(newAx, tang_heights_lin ,linewidth = 5 )
# ax1.plot(A_scal_T * theta_T, height_values ,linewidth = 10)
# ax1.plot(A_scal_O3 * theta_P, height_values ,linewidth = 5)
plt.show()


#meanPress = np.zeros(theta_P.shape)
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

# ##
# """start the mtc algo with first guesses of noise and lumping const delta"""
#
# vari = np.zeros((len(theta_P) - 2, 1))
#
# for j in range(1, len(theta_P) - 1):
#     vari[j-1] = np.var([theta_P[j - 1], theta_P[j], theta_P[j + 1]])
# ATy = np.matmul(A.T, y )  #- A @ meanPress)
# #find minimum for first guesses
# '''params[1] = tau
# params[0] = gamma'''
# def MinLogMargPost(params):#, coeff):
#
#     # gamma = params[0]
#     # delta = params[1]
#     gamma = params[0]
#     lamb_tau = params[1]
#     if lamb_tau < 0  or gamma < 0:
#         return np.nan
#
#     n = SpecNumLayers
#     m = SpecNumMeas
#
#     Bp = ATA + lamb_tau * P
#
#
#     B_inv_A_trans_y, exitCode = gmres(Bp, ATy, tol=tol, restart=25)
#     if exitCode != 0:
#         print(exitCode)
#
#     G = g(A, P,  lamb_tau)
#     F = f(ATy, y ,  B_inv_A_trans_y)
#
#     return -n/2 * np.log(lamb_tau) - (m/2 + 1) * np.log(gamma) + 0.5 * G + 0.5 * gamma * F +  ( betaD *  lamb_tau * gamma + betaG *gamma)
#
# #minimum = optimize.fmin(MargPostU, [5e-5,0.5])
# minimum = optimize.fmin(MinLogMargPost, [1/(np.max(Ax) * 0.01),1/(np.mean(vari))*(np.max(Ax) * 0.01)])
# print(minimum)
# # lam_tau_0 = minimum[1]
# # print('lambda_0: ' )
# # print('{:.1e}'.format(lam_tau_0))
# # B_0 = ATA + lam_tau_0 * P
# #
# # B_inv_A_trans_y0, exitCode = gmres(B_0, ATy, tol=tol, restart=25)
# # if exitCode != 0:
# #     print(exitCode)
# #
# # Bu, Bs, Bvh = np.linalg.svd(B_0)
# # cond_B =  np.max(Bs)/np.min(Bs)
# # print("Condition number B: " + str(orderOfMagnitude(cond_B)))
# #
#


##
'''do t-walk '''
import pytwalk

numPara = 3
tWalkSampNum = 100000
burnIn = 1000
# samplesOfa = np.zeros((numPara, numberOfSamp + burnIn))
# samplesOfb = np.zeros((numPara, numberOfSamp + burnIn))
# samplesOfPress = np.zeros((SpecNumLayers, numberOfSamp + burnIn))


#efit, dfit, cfit,
cfit, bfit, afit = np.polyfit(height_values, np.log(pressure_values), 2)

# a_curr = np.random.uniform(low=pressure_values[0], high=np.exp(afit)+np.exp(afit)/2, size=numPara)
# b_curr = np.random.uniform(low=-bfit+bfit/2, high=-bfit-bfit/2, size=numPara)

# def press(a,b,x):
#     a[0] = pressure_values[0]*1.75e1
#     return ((temp_Mat @ a) * np.exp(- (temp_Mat @ b) * x))
def press(a,b,c,d,e,x):
    #a[0] = pressure_values[0]*1.75e1
    return np.exp( e * x**4 + d * x**3 + c * x**2 + b * x + a)

gamma = 1/(np.max(Ax) * 0.01)
def log_post(Params):
    a = Params[0]
    b = Params[1]
    c = Params[2]
    d = 0#Params[3]
    e = 0#Params[4]
    #print( gamma/2 * np.sum( ( y - A @ press(a,b,c,d,height_values).reshape((SpecNumLayers,1)) )**2 ))
    return gamma/2 * np.sum( ( y - A @ press(a,b,c,d,e,height_values).reshape((SpecNumLayers,1)) )**2 )

# def MargPostSupp(Params):
#     list = [0 < Params.all(), (pressure_values[0] <= Params[1:numPara]).all(), (Params[1:numPara] <= (np.exp(afit)+np.exp(afit)/2)).all(), ((-bfit+bfit/2) <= Params[numPara::]).all(), (Params[numPara::] <= (-bfit-bfit/2)).all() ]
#     return all(list)

def MargPostSupp(Params):
    list = []
    list.append(Params[0] > 0)
    #list.append(Params[0] < 17)
    list.append(Params[1] < 0)
    list.append(Params[1] > -2e-1)
    # list.append(Params[2] < 0)
    # list.append(Params[2] > -4e-3)
    # list.append(Params[3] > 2e-5)
    # list.append(Params[3] < 10e-5)
    # list.append(Params[4] < 0)
    # list.append(Params[4] > -6e-7)
    return all(list)

MargPost = pytwalk.pytwalk( n=numPara, U=log_post, Supp=MargPostSupp)
startTime = time.time()
x0 =  np.ones(numPara)
x0[0] = afit
x0[1] = bfit
x0[2] = cfit
# x0[3] = dfit
# x0[4] = efit

# x0[numPara::]= -bfit * x0[numPara::]
# xp0 =  np.ones(2*numPara)
# xp0[:numPara] = a_curr
# xp0[numPara::]= b_curr
xp0 = 1.02 * x0
xp0[0] = 7
print(MargPostSupp(x0))
print(MargPostSupp(xp0))

# while (MargPostSupp(xp0) != True) and (MargPostSupp(x0) != True) :
#     xp0[:numPara] = np.random.uniform(low=np.exp(afit)-49*np.exp(afit)/50, high=np.exp(afit)+np.exp(afit)/2, size=numPara)
#     xp0[numPara::] = np.random.uniform(low=-bfit+bfit/2, high=-bfit-bfit/2, size=numPara)
#     x0[numPara::] = np.random.uniform(low=-bfit+bfit/2, high=-bfit-bfit/2, size=numPara)
#     x0[:numPara] = np.random.uniform(low=np.exp(afit)-49*np.exp(afit)/50, high=np.exp(afit)+np.exp(afit)/2, size=numPara)


MargPost.Run( T=tWalkSampNum + burnIn, x0=x0, xp0=xp0 )

elapsedtWalkTime = time.time() - startTime
print('Elapsed Time for t-walk: ' + str(elapsedtWalkTime))
#MargPost.Ana()
#MargPost.TS()

#MargPost.Hist( par=0 )
#MargPost.Hist( par=1 )

MargPost.SavetwalkOutput("MargPostDat.txt")

SampParas = np.loadtxt("MargPostDat.txt")

MeanParas = np.mean(SampParas[3*burnIn:,:],0)


# fig, axs = plt.subplots(numPara, 1, tight_layout=True)
# #burnIn = 50
# # We can set the number of bins with the *bins* keyword argument.
# axs[0].hist(SampParas[burnIn::math.ceil(IntAutoGamPyT),0],bins=n_bins)
# axs[0].set_title( str(len(SampParas[burnIn::math.ceil(IntAutoGamPyT),0]))+ ' effective $\gamma$ sample' )
# #axs[1].hist(SampParas[burnIn::math.ceil(IntAutoDeltaPyT),1],bins=n_bins)
# axs[1].hist(deltasPyT[burnIn::math.ceil(IntAutoDeltaPyT)],bins=n_bins)
# axs[1].set_title(str(len(deltasPyT[burnIn::math.ceil(IntAutoDeltaPyT)])) + ' effective $\delta$ samples')
# #axs[1].set_title(str(len(SampParas[burnIn::math.ceil(IntAutoDeltaPyT),1])) + ' effective $\delta$ samples')
# #axs[2].hist(lambasPyT[burnIn::math.ceil(IntAutoLamPyT)],bins=n_bins)
# axs[2].hist(SampParas[burnIn::math.ceil(IntAutoLamPyT),1],bins=n_bins)
# axs[2].set_title(str(len(SampParas[burnIn::math.ceil(IntAutoLamPyT),1])) + ' effective $\delta$ samples')
# #axs[2].xaxis.set_major_formatter(scientific_formatter)
# #axs[2].set_title(str(len(SampParas[burnIn::math.ceil(IntAutoDeltaPyT),1])) + ' effective $\lambda =\delta / \gamma samples $')
# #plt.savefig('PyTWalkHistoResults.png')
# #plt.show()

##
cfit, bfit, afit = np.polyfit(height_values, np.log(pressure_values), 2)
calc_fit_press = press(afit, bfit, cfit, 0, 0, height_values)
t_walk_press = press(MeanParas[0] ,MeanParas[1], 0, 0,0, height_values)
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(t_walk_press, height_values)
ax1.plot(calc_fit_press, height_values)
ax1.plot(pressure_values, height_values )
#ax1.plot(f(*popt,height_values), height_values )
ax1.set_xlabel(r'Pressure in hPa ')
ax1.set_ylabel('Height in km')
plt.show()

##
fig3, axs = plt.subplots(2,1)
axs[0].plot(range(tWalkSampNum),SampParas[1*burnIn:,0] )
axs[1].plot(range(tWalkSampNum),SampParas[1*burnIn:,1] )
plt.show()

print('twalk done')
## get temp values



##
grav = 9.81 * ((R_Earth)/(R_Earth + height_values))**2
R = constants.gas_constant
del_height = height_values[1:] - height_values[:-1]

calc_fit_temp  = np.zeros((len(del_height),1))
recov_temp  = np.zeros((len(del_height),1))


recov_press =t_walk_press#  press(MeanParas[0] ,MeanParas[1], MeanParas[2], MeanParas[3], MeanParas[4], height_values)

for i in range(1, len(del_height)):

    #calc_press[i] = calc_press[i-1] * np.exp(-28.97 * grav[i] / temp_values[i] / R * del_height[i] )

    calc_fit_temp[i] = -28.97 * grav[i] / np.log(calc_fit_press[i] / calc_fit_press[i - 1]) / R * del_height[i]

    recov_temp[i] =  -28.97 * grav[i] / np.log(recov_press[i] / recov_press[i-1]) / R * del_height[i]


fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.scatter(recov_temp[1:], height_values[1:-1])
ax1.plot(temp_values, height_values, linewidth = 0.5)
ax1.plot(calc_fit_temp[1:], height_values[1:-1], linewidth = 0.5)
plt.show()

print('temp calc')
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


# B_inv_T = np.zeros(np.shape(B_0))
#
# for i in range(len(B_0)):
#     B_inv_T[:, i], exitCode = gmres(B_0, P[:, i], tol=tol, restart=25)
#     if exitCode != 0:
#         print('B_inv_L ' + str(exitCode))
#
# #relative_tol_L = tol
# #CheckB_inv_L = np.matmul(B, B_inv_L)
# #print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)
#
# B_inv_T_2 = np.matmul(B_inv_T, B_inv_T)
# B_inv_T_3 = np.matmul(B_inv_T_2, B_inv_T)
# B_inv_T_4 = np.matmul(B_inv_T_2, B_inv_T_2)
# B_inv_T_5 = np.matmul(B_inv_T_4, B_inv_T)
# B_inv_T_6 = np.matmul(B_inv_T_4, B_inv_T_2)
#
#
# f_0_1 = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_T), B_inv_A_trans_y0)
# f_0_2 = -1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_T_2), B_inv_A_trans_y0)
# f_0_3 = 1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_T_3) ,B_inv_A_trans_y0)
# f_0_4 = -1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_T_4) ,B_inv_A_trans_y0)
# #f_0_5 = 120 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y)
#
#
# g_0_1 = np.trace(B_inv_T)
# g_0_2 = -1 / 2 * np.trace(B_inv_T_2)
# g_0_3 = 1 /6 * np.trace(B_inv_T_3)
# g_0_4 = -1 /24 * np.trace(B_inv_T_4)
# g_0_5 = 0#1 /120 * np.trace(B_inv_L_5)
# g_0_6 = 0#1 /720 * np.trace(B_inv_L_6)
#



##
'''do the sampling'''


# number_samples = 10000
#inintialize sample

# wLam = 0.8e7
# startTime = time.time()
# lambdas, gammas, accepted = MHwG(number_samples, SpecNumMeas, SpecNumLayers, burnIn, lam_tau_0 , minimum[0], wLam, y - A @ meanPress, ATA, P, B_inv_A_trans_y0, ATy, tol, betaG, betaD, f_0_1, f_0_2, f_0_3, g_0_1, g_0_2, g_0_3)
# elapsed = time.time() - startTime
# print('MTC Done in ' + str(elapsed) + ' s')
#
# print('acceptance ratio: ' + str(accepted /(number_samples+burnIn)))
# deltas = lambdas * gammas
# np.savetxt('samplesT.txt', np.vstack((gammas[burnIn::], deltas[burnIn::], lambdas[burnIn::])).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsed), fmt = '%.15f \t %.15f \t %.15f')


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
# new_lamb = lambdas#[burnIn::math.ceil(IntAutoLam)]
#
# new_gam = gammas#[burnIn::math.ceil(IntAutoGam)]

# new_delt = deltas#[burnIn::math.ceil(IntAutoDelt)]
#
#
#
# fig, axs = plt.subplots(3, 1,tight_layout=True)
# # We can set the number of bins with the *bins* keyword argument.
# axs[0].hist(new_gam,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoGam)))
# #axs[0].set_title(str(len(new_gam)) + ' effective $\gamma$ samples')
# axs[0].set_title(str(len(new_gam)) + r' $\gamma$ samples, the noise precision')
# axs[1].hist(new_delt,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoDelt)))
# axs[1].set_title(str(len(new_delt)) + ' $\delta$ samples, the prior precision')
# axs[2].hist(new_lamb,bins=n_bins, color = 'k')#10)
# #axs[2].xaxis.set_major_formatter(scientific_formatter)
# #axs[2].set_title(str(len(new_lamb)) + ' effective $\lambda =\delta / \gamma$ samples')
# axs[2].set_title(str(len(new_lamb)) + ' $\lambda$ samples, the regularization parameter')
# plt.savefig('HistoResults.png')
# plt.show()

##
'''do metropolis hastings on A and B to find pressure'''
numPara = 6
numberOfSamp = 500000
burnIn = 1000
samplesOfa = np.zeros((numPara, numberOfSamp + burnIn))
samplesOfb = np.zeros((numPara, numberOfSamp + burnIn))
samplesOfPress = np.zeros((SpecNumLayers, numberOfSamp + burnIn))

temp_Mat =  np.zeros((SpecNumLayers,numPara))
temp_Mat[0:4,0] = 1
temp_Mat[4:8,1] = 1
temp_Mat[8:13,2] = 1
temp_Mat[13:18,3] = 1
temp_Mat[18:24,4] = 1
temp_Mat[24:,5] = 1
#temp_Mat[30:,6] = 1


def press(a,b,x):
    return ((temp_Mat @ a) * np.exp(- (temp_Mat @ b) * x))

def log_post(a,b):
    return -gamma/2 * np.sum( ( y - A @ press(a, b,height_values).reshape((SpecNumLayers,1)) )**2 )


bfit, afit = np.polyfit(height_values, np.log(pressure_values), 1)
# x * bfit +  afit

#28.97 * 9.81 / temp_values[0] / R * (height_values[0,]
#B = np.random.normal(28.97 * 9.81 / temp_values[0] / R , 0.01, SpecNumLayers)
#B = np.random.normal(-bfit , 0.01, SpecNumLayers)
#A = np.random.normal(pressure_values[0], 0.01,SpecNumLayers)
#A = np.random.normal(np.exp(afit), 1000,SpecNumLayers)



b_curr = np.random.uniform(low=-bfit+bfit/2, high=-bfit-bfit/2, size=numPara)
a_curr = np.random.uniform(low=np.exp(afit)-np.exp(afit)/2, high=np.exp(afit)+np.exp(afit)/2, size=numPara)

b_curr = -bfit * np.ones(numPara)
a_curr = np.exp(afit) * np.ones(numPara)

# b_curr = np.array([0.16770334, 0.19523378, 0.1217303 , 0.18741372, 0.2245279 ])
# a_curr = np.array([16.11508425, 27.83926419,  6.29815698, 20.33031615, 37.60576857])
samplesOfb[:,0] = b_curr
samplesOfa[:,0] = a_curr
samplesOfPress[:,0] = press(a_curr, b_curr,height_values)
k = 0
for t in range(numberOfSamp + burnIn-1):

    #b_prime = np.random.uniform(low=-bfit+bfit, high=-bfit-bfit, size=6)
    #a_prime = np.random.uniform(low=np.exp(afit)-np.exp(afit), high=np.exp(afit)+np.exp(afit), size=SpecNumLayers)
    b_prime = np.random.normal(b_curr, b_curr / 20, size=numPara)
    while np.any( b_prime < 0 ):
        b_prime = np.random.normal(b_curr, b_curr/20, size=numPara)

    a_prime = np.random.normal(a_curr, a_curr / 20, size=numPara)
    while np.any( a_prime < 0 ):
        a_prime =  np.random.normal(a_curr, a_curr/20, size=numPara)

    press_prime = press(a_prime,b_prime,height_values)



    #accept or reject new sample
    accept_prob = log_post(a_prime,b_prime) - log_post(a_curr,b_curr)

    u = uniform()
    if np.log(u) <= accept_prob:
    #accept
        k = k + 1
        samplesOfa[:,t+1] = a_prime
        samplesOfb[:,t+1] = b_prime
        samplesOfPress[:,t+1] = press_prime
        a_curr = np.copy(a_prime)
        b_curr = np.copy(b_prime)
    else:
        samplesOfa[:,t+1] = np.copy(b_curr)
        samplesOfb[:,t+1] = np.copy(b_curr)
        samplesOfPress[:,t+1] = press(a_curr,b_curr,height_values)

print('acceptance ratio: ' + str(k/(numberOfSamp+burnIn)))
print('k: ' + str(k) )

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(np.mean(samplesOfPress[:,burnIn::],1), height_values)
ax1.plot(pressure_values, height_values )
#ax1.plot(f(*popt,height_values), height_values )
ax1.set_xlabel(r'Pressure in hPa ')
ax1.set_ylabel('Height in km')
plt.show()






