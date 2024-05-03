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
# heights = 145366.45 * (1 - ( press /1013.25)**0.190284 ) * 0.3048 * scalingConstkm


height_values = heights[minInd:maxInd].reshape((maxInd-minInd,1))
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


scalingConst = 1e5
# A_scal_T = pressure_values.reshape((SpecNumLayers,1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm * num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]
#
# theta_O3 = num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]


A_scal_O3 = 1e2 * LineIntScal  * Source * AscalConstKmToCm * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0] * num_mole / temp_values.reshape((SpecNumLayers,1))
#scalingConst = 1e11

theta_P = pressure_values.reshape((SpecNumLayers,1))

""" plot forward model values """


A = A_lin * A_scal_O3.T
ATA = np.matmul(A.T,A)
Au, As, Avh = np.linalg.svd(A)
cond_A =  np.max(As)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))


Ax = np.matmul(A, theta_P)

#convolve measurements and add noise
y, gamma  = add_noise(Ax, 1)
np.savetxt('dataY.txt', y, header = 'Data y including noise', fmt = '%.15f')
# y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
# gamma = 7.6e-5

#gamma = 1/(np.max(Ax) * 0.1)**2

''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''




# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(Ax, tang_heights_lin ,linewidth = 5 )
# ax1.plot(y, tang_heights_lin ,linewidth = 5 )
# plt.show()

##
"""update A so that O3 profile is constant"""
w_cross =   f_broad * 1e-4 * np.mean(VMR_O3) * gaussian(height_values,35,10).reshape((SpecNumLayers,1))

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(gaussian(height_values, 35, 10) * np.mean(VMR_O3), height_values, linewidth = 2.5, label = 'my guess', marker = 'o')
ax1.plot(VMR_O3, height_values, linewidth = 2.5, label = 'true profile', marker = 'o')
ax1.set_ylabel('Height in km')
ax1.set_xlabel('Volume Mixing Ratio of Ozone')
ax2 = ax1.twiny()
ax2.scatter(y, tang_heights_lin ,linewidth = 2, marker =  'x', label = 'data' , color = 'k')
ax2.set_xlabel(r'Spectral radiance in $\frac{W cm}{m^2  sr} $',labelpad=10)# color =dataCol,
ax1.legend()
plt.savefig('DataStartTrueProfile.png')
plt.show()

# internal partition sum
Q = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ np.mean(temp_values))
Q_ref = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ 296)
LineIntScal =  Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ np.mean(temp_values))/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ np.mean(temp_values)))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))


C1 =2 * scy.constants.h * scy.constants.c**2 * v_0**3 * 1e8
C2 = scy.constants.h * scy.constants.c * 1e2 * v_0  / (scy.constants.Boltzmann * np.mean(temp_values) )
#plancks function
Source = np.array(C1 /(np.exp(C2) - 1) )





A_scal_O3 = 1e2 * LineIntScal  * Source * AscalConstKmToCm * w_cross * scalingConst * S[ind,0] * num_mole / np.mean(temp_values)
#scalingConst = 1e11




""" plot forward model values """


A = A_lin * A_scal_O3.T
ATA = np.matmul(A.T,A)
Au, As, Avh = np.linalg.svd(A)
cond_A =  np.max(As)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))

##

grad = np.log(pressure_values[1:])- np.log(pressure_values[:-1])/(height_values[1:,0]- height_values[:-1,0])
bfitup, afitup = np.polyfit(height_values[-5:,0], grad[-5:], 1)
bfitlow, afitlow = np.polyfit(height_values[0:25,0], grad[0:25], 1)

numPara = 2
paraMat = np.zeros((len(height_values), numPara))
breakInd = 28

paraMat[0:breakInd,0] = np.ones(breakInd)
paraMat[breakInd:,1] = np.ones(int(len(height_values)) -breakInd)

def pressFunc(x, b1, b2, a1, a2):
    b = paraMat @ [b1,b2]
    a = paraMat @ [a1, a2]
    #a = np.log(1013)
    return b * x + a

popt, pcov = scy.optimize.curve_fit(pressFunc, height_values[:,0], np.log(pressure_values))#, p0=[2e-2,2e-2, np.log(1013)])



calc_fit_press = pressFunc(height_values[:,0], *popt)

cross_heigth = (afitup - afitlow )/ (bfitlow - bfitup)

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# for i in range(burnIn, len(SampParas),100):
#     t_walk_press = press(afit, SampParas[i,0] ,SampParas[i,1], 0,0, height_values)
#     ax1.plot(t_walk_press, height_values, linewidth = 0.5)
ax1.plot(bfitup *  height_values[:,0] + afitup, height_values, linewidth = 2)
ax1.plot(bfitlow *  height_values[:,0] + afitlow, height_values, linewidth = 2)
ax1.plot(calc_fit_press, height_values, linewidth = 2)
ax1.plot(np.log(press), heights , label = 'true press.')
ax1.scatter(grad , height_values[1:])
#ax1.plot(fit_press, height_values, linewidth = 2.5, label = 'samp. press. fit')#
#ax1.plot(f(*popt,height_values), height_values )
ax1.set_xlabel(r'Pressure in hPa ')
ax1.set_ylabel('Height in km')
#ax1.set_xscale('log')
ax1.legend()
plt.show()
plt.savefig('samplesPressure.png')

##
'''do t-walk '''
import pytwalk


tWalkSampNum = 90000
burnIn =3000

# #efit, dfit, cfit,
# bfit, afit = np.polyfit(height_values, np.log(pressure_values), 1)
#
#
# def pressFunc(a,b,c,d,e,x):
#     #a[0] = pressure_values[0]*1.75e1
#     return np.exp( e * x**4 + d * x**3 + c * x**2 + b * x + a)

numPara = 2
paraMat = np.zeros((len(height_values), numPara))
#breakInd = 21

paraMat[0:breakInd,0] = np.ones(breakInd)
paraMat[breakInd:,1] = np.ones(int(len(height_values)) -breakInd)

def pressFunc(x, b1, b2, a1, a2):
    b = paraMat @ [b1,b2]
    a = paraMat @ [a1, a2]
    #a = np.log(1013)
    return np.exp(b * x + a)

#gamma = 1/(np.max(Ax) * 0.01)
def log_post(Params):
    a1 = Params[2]
    a2 = Params[3]##
    b1 = Params[0]
    b2 = Params[1]
    c = 0
    d = 0
    e = 0#Params[4]
    #print( gamma/2 * np.sum( ( y - A @ press(a,b,c,d,height_values).reshape((SpecNumLayers,1)) )**2 ))
    #return  gamma/2 * np.sum( ( y - A @ pressFunc(a,b,c,d,e,height_values).reshape((SpecNumLayers,1)) )**2 )
    #return np.sum( ( y - A @ pressFunc(a,b,c,d,e,height_values).reshape((SpecNumLayers,1)) )**2 )
    return np.sum( ( y - A @  pressFunc(height_values[:,0], b1, b2,a1, a2).reshape((SpecNumLayers,1)) )**2 )

# def MargPostSupp(Params):
#     list = [0 < Params.all(), (pressure_values[0] <= Params[1:numPara]).all(), (Params[1:numPara] <= (np.exp(afit)+np.exp(afit)/2)).all(), ((-bfit+bfit/2) <= Params[numPara::]).all(), (Params[numPara::] <= (-bfit-bfit/2)).all() ]
#     return all(list)

def MargPostSupp(Params):
    list = []
    list.append(Params[2] > 6.5)
    list.append(Params[2] < 7.5)#np.log(1200))
    list.append(Params[3] > 5.5)
    list.append(Params[3] < 6.5)
    list.append(Params[0] < 0)
    list.append(Params[0] > -2e-1)
    list.append(Params[1] < 0)
    list.append(Params[1] > -2e-1)
    #list.append(abs(Params[1] - Params[0]) < 5e-1)
    # list.append(Params[2] > 0)
    # list.append(Params[2] < 1e-3)
    #list.append(Params[3] > 0)
    # list.append(Params[2] > -4e-3)
    # list.append(Params[3] > 2e-5)
    # list.append(Params[3] < 10e-5)
    # list.append(Params[4] < 0)
    # list.append(Params[4] > -6e-7)
    return all(list)

MargPost = pytwalk.pytwalk( n=4, U=log_post, Supp=MargPostSupp)
startTime = time.time()
x0 = popt# np.ones(numPara)
#x0[0] = np.log(1013)
#x0[0] = bfit
#x0[0] = bfit
#x0[2] = cfit
# x0[3] = dfit

# x0[numPara::]= -bfit * x0[numPara::]
# xp0 =  np.ones(2*numPara)
# xp0[:numPara] = a_curr
# xp0[numPara::]= b_curr
xp0 = 1.001 * x0
#xp0[0] = 7
print(MargPostSupp(x0))
print(MargPostSupp(xp0))


MargPost.Run( T=tWalkSampNum + burnIn, x0=x0, xp0=xp0 )

elapsedtWalkTime = time.time() - startTime
print('Elapsed Time for t-walk: ' + str(elapsedtWalkTime))
MargPost.Ana()
#MargPost.TS()

#MargPost.Hist( par=0 )
#MargPost.Hist( par=1 )

MargPost.SavetwalkOutput("MargPostDat.txt")

SampParas = np.loadtxt("MargPostDat.txt")

MeanParas = np.mean(SampParas[3*burnIn:,:],0)


##
#bfit, afit = np.polyfit(height_values, np.log(pressure_values), 1)
#recov_press = pressFunc(height_values, MeanParas[0],-0.1299,6)
recov_press = pressFunc(height_values[:,0], MeanParas[0],MeanParas[1],MeanParas[2],MeanParas[3])
fit_press = pressFunc(height_values[:,0], *popt)
#t_walk_press = press(afit, SampParas[2500,0] ,SampParas[2500,1], 0,0, height_values)
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# for i in range(burnIn, len(SampParas),100):
#     t_walk_press = press(afit, SampParas[i,0] ,SampParas[i,1], 0,0, height_values)
#     ax1.plot(t_walk_press, height_values, linewidth = 0.5)
ax1.plot(fit_press, height_values[:,0], linewidth = 5)
ax1.plot(press, heights , label = 'true press.')
#ax1.plot(np.log(pressure_values[:-1])- np.log(pressure_values[1:]), height_values[1:])
ax1.plot(recov_press, height_values, linewidth = 2.5, label = 'samp. press. fit')#
#ax1.plot(f(*popt,height_values), height_values )
ax1.set_xlabel(r'Pressure in hPa ')
ax1.set_ylabel('Height in km')
ax1.legend()
plt.show()
plt.savefig('samplesPressure.png')

##
fig3, axs = plt.subplots(2,1)
axs[0].plot(range(tWalkSampNum),SampParas[1*burnIn:,0] )
axs[1].plot(range(tWalkSampNum),SampParas[1*burnIn:,1] )
plt.show()


print('twalk done')
## get temp values
R_Earth = 6371
grav = 9.81 * ((R_Earth)/(R_Earth + height_values))**2
R = constants.gas_constant

del_height = height_values[1:] - height_values[:-1]

calc_fit_temp  = np.zeros((len(del_height),1))
recov_temp  = np.zeros((len(del_height),1))


for i in range(1, len(del_height)):

    #calc_press[i] = calc_press[i-1] * np.exp(-28.97 * grav[i] / temp_values[i] / R * del_height[i] )

    calc_fit_temp[i] = -28.97 * grav[i] / np.log(fit_press[i] / fit_press[i - 1]) / R * del_height[i]

    recov_temp[i] =  -28.97 * grav[i] / np.log(recov_press[i] / recov_press[i-1]) / R * del_height[i]



#recov_temp[recov_temp < 0.2 *np.mean(temp_values)] = np.nan
#recov_temp[recov_temp > 1.2 *np.mean(temp_values)] = np.nan
idx = np.isfinite(recov_temp)
eTempfit, dTempfit, cTempfit, bTempfit, aTempfit = np.polyfit(height_values[:,0], temp_values, 4)

fit_heights = height_values[1:]
eTempSamp, dTempSamp, cTempSamp, bTempSamp, aTempSamp = np.polyfit(fit_heights[idx], recov_temp[idx], 4)


def temp(a,b,c,d,e,x):
    #a[0] = pressure_values[0]*1.75e1
    return  e * x**4 + d * x**3 + c * x**2 + b * x + a

fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
ax1.scatter(recov_temp[1:], height_values[1:-1],label = 'sampled T',color = 'r')
ax1.plot(temp(aTempSamp,bTempSamp,cTempSamp, dTempSamp, eTempSamp,height_values), height_values, linewidth = 2.5, color = 'r',label = 'fitted T')
ax1.plot(temp_values, height_values, linewidth = 5, label = 'true T', color = 'green', zorder = 0)
#ax1.plot(temp(aTempfit,bTempfit,cTempfit, dTempfit, eTempfit,height_values), height_values, linewidth = 1.5, color = 'r')
ax1.plot(calc_fit_temp[1:], height_values[1:-1], linewidth = 0.5)
ax1.legend()
plt.show()
plt.savefig('TemperatureSamp.png')
print('temp calc')

## set new forward model and

TempSamp = temp(aTempSamp,bTempSamp,cTempSamp, dTempSamp, eTempSamp,height_values)
#TempSamp = temp_values
#recov_press = pressure_values

"""update A so with new temp and pressure"""
# w_cross =   f_broad * 1e-4 * gaussian(height_values, 35,10) * np.mean(VMR_O3)
# w_cross =   f_broad * 1e-4 * gaussian(height_values, 25,5) * np.mean(VMR_O3)
#w_cross =  VMR_O3 * f_broad * 1e-4


# internal partition sum
Q = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ TempSamp)
Q_ref = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ 296)
LineIntScal =  Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ TempSamp)/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ TempSamp))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))


C1 =2 * scy.constants.h * scy.constants.c**2 * v_0**3 * 1e8
C2 = scy.constants.h * scy.constants.c * 1e2 * v_0  / (scy.constants.Boltzmann * TempSamp )
#plancks function
Source = np.array(C1 /(np.exp(C2) - 1) )



#take linear
num_mole = 1 / ( scy.constants.Boltzmann )#* temp_values)

AscalConstKmToCm = 1e3
#1e2 for pressure values from hPa to Pa
A_scal = recov_press.reshape((SpecNumLayers,1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm/ ( TempSamp)

#theta =(num_mole * w_cross.reshape((SpecNumLayers,1)) * Source * scalingConst )
theta = num_mole * w_cross * np.ones((SpecNumLayers,1)) * scalingConst * S[ind,0]


""" plot forward model values """

A = A_lin * A_scal.T
ATA = np.matmul(A.T,A)
Au, As, Avh = np.linalg.svd(A)
cond_A =  np.max(As)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))

Ax = np.matmul(A, theta)


ATy = np.matmul(A.T, y)


##
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
#ax1.plot(theta, height_values, linewidth = 2.5)
ax1.plot(y, tang_heights_lin, linewidth = 1.5)
ax1.plot(Ax, tang_heights_lin, linewidth = 2.5)

plt.show()
##



# graph Laplacian
# direchlet boundary condition
NOfNeigh = 2#4
neigbours = np.zeros((len(height_values),NOfNeigh))
# neigbours[0] = np.nan, np.nan, 1, 2
# neigbours[-1] = len(height_values)-2, len(height_values)-3, np.nan, np.nan
# neigbours[0] = np.nan, 1
# neigbours[-1] = len(height_values)-2, np.nan
for i in range(0,len(height_values)):
    neigbours[i] = i-1, i+1
    #neigbours[i] = i-2, i-1, i+1, i+2#, i+3 i-3,


neigbours[neigbours >= len(height_values)] = np.nan
neigbours[neigbours < 0] = np.nan

L = generate_L(neigbours)
# startInd = 23
# L[startInd::, startInd::] = L[startInd::, startInd::] * 5
# L[startInd, startInd] = -L[startInd, startInd-1] - L[startInd, startInd+1] #-L[startInd, startInd-2] - L[startInd, startInd+2]

#L[startInd+1, startInd+1] = -L[startInd+1, startInd+1-1] - L[startInd+1,startInd+1+1] -L[startInd+1, startInd+1-2] - L[startInd+1, startInd+1+2]
# L[16, 16] = 13

np.savetxt('GraphLaplacian.txt', L, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')


##

vari = np.zeros((len(theta)-2,1))

for j in range(1,len(theta)-1):
    vari[j-1] = np.var([theta[j-1],theta[j],theta[j+1]])

#find minimum for first guesses
'''params[1] = delta
params[0] = gamma'''
def MinLogMargPost(params):#, coeff):

    # gamma = params[0]
    # delta = params[1]
    gam = params[0]
    lamb = params[1]
    if lamb < 0  or gamma < 0:
        return np.nan

    n = SpecNumLayers
    m = SpecNumMeas

    Bp = ATA + lamb * L


    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( betaD *  lamb * gam + betaG *gam)

#minimum = optimize.fmin(MargPostU, [5e-5,0.5])
minimum = optimize.fmin(MinLogMargPost, [1/gamma,gamma/np.var(theta)])

lam0 = minimum[1]
print(minimum)


##
# """ finally calc f and g with a linear solver adn certain lambdas
#  using the gmres"""
#
# lam= np.logspace(-5,15,500)
# f_func = np.zeros(len(lam))
# g_func = np.zeros(len(lam))
#
#
#
# for j in range(len(lam)):
#
#     B = (ATA + lam[j] * L)
#
#     B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
#     #print(exitCode)
#
#     CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)
#     if np.linalg.norm(ATy[0::, 0]- CheckB_inv_ATy)/np.linalg.norm(ATy[0::, 0])<=tol:
#         f_func[j] = f(ATy, y, B_inv_A_trans_y)
#     else:
#         f_func[j] = np.nan
#
#     g_func[j] = g(A, L, lam[j])
#
#
# np.savetxt('f_func.txt', f_func, fmt = '%.15f')
# np.savetxt('g_func.txt', g_func, fmt = '%.15f')
# np.savetxt('lam.txt', lam, fmt = '%.15f')


##
#taylor series arounf lam_0
B = (ATA + lam0* L)

B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
#print(exitCode)

CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)



B_inv_L = np.zeros(np.shape(B))

for i in range(len(B)):
    B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
    if exitCode != 0:
        print('B_inv_L ' + str(exitCode))

#relative_tol_L = tol
#CheckB_inv_L = np.matmul(B, B_inv_L)
#print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)

B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)
B_inv_L_6 = np.matmul(B_inv_L_4, B_inv_L_2)


f_0_1 = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L), B_inv_A_trans_y)
f_0_2 = -1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L_2), B_inv_A_trans_y)
f_0_3 = 1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_3) ,B_inv_A_trans_y)
f_0_4 = -1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y)
#f_0_5 = 120 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y)


g_0_1 = np.trace(B_inv_L)
g_0_2 = -1 / 2 * np.trace(B_inv_L_2)
g_0_3 = 1 /6 * np.trace(B_inv_L_3)
g_0_4 = -1 /24 * np.trace(B_inv_L_4)
g_0_5 = 0#1 /120 * np.trace(B_inv_L_5)
g_0_6 = 0#1 /720 * np.trace(B_inv_L_6)

##

'''do the sampling'''
number_samples = 10000


#inintialize sample
gamma0 = minimum[0]
lambda0 = minimum[1]
B = (ATA + lambda0 * L)
B_inv_A_trans_y0, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

Bu, Bs, Bvh = np.linalg.svd(B)
cond_B =  np.max(Bs)/np.min(Bs)
print("Condition number B: " + str(orderOfMagnitude(cond_B)))


#wLam = 2e2#5.5e2
#wgam = 1e-5
#wdelt = 1e-1

alphaG = 1
alphaD = 1
rate = f(ATy, y, B_inv_A_trans_y0) / 2 + betaG + betaD * lambda0
# draw gamma with a gibs step
shape = SpecNumMeas/2 + alphaD + alphaG

f_new = f(ATy, y,  B_inv_A_trans_y0)
#g_old = g(A, L,  lambdas[0])

def MHwG(number_samples, burnIn, lambda0, gamma0):
    wLam = 1.5e4#1.5e4#7e1

    alphaG = 1
    alphaD = 1
    k = 0

    gammas = np.zeros(number_samples + burnIn)
    #deltas = np.zeros(number_samples + burnIn)
    lambdas = np.zeros(number_samples + burnIn)

    gammas[0] = gamma0
    lambdas[0] = lambda0

    B = (ATA + lambda0 * L)
    B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0=B_inv_A_trans_y0, tol=tol)

    #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    shape = SpecNumMeas / 2 + alphaD + alphaG
    rate = f(ATy, y, B_inv_A_trans_y) / 2 + betaG + betaD * lambda0


    for t in range(number_samples + burnIn-1):
        #print(t)

        # # draw new lambda
        lam_p = normal(lambdas[t], wLam)

        while lam_p < 0:
                lam_p = normal(lambdas[t], wLam)

        delta_lam = lam_p - lambdas[t]
        # B = (ATA + lam_p * L)
        # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
        # if exitCode != 0:
        #     print(exitCode)


        # f_new = f(ATy, y,  B_inv_A_trans_y)
        # g_new = g(A, L,  lam_p)
        #
        # delta_f = f_new - f_old
        # delta_g = g_new - g_old

        delta_f = f_0_1 * delta_lam + f_0_2 * delta_lam**2 + f_0_3 * delta_lam**3
        delta_g = g_0_1 * delta_lam + g_0_2 * delta_lam**2 + g_0_3 * delta_lam**3

        log_MH_ratio = ((SpecNumLayers)/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

        #accept or rejeict new lam_p
        u = uniform()
        if np.log(u) <= log_MH_ratio:
        #accept
            k = k + 1
            lambdas[t + 1] = lam_p
            #only calc when lambda is updated

            B = (ATA + lam_p * L)
            B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0= B_inv_A_trans_y0,tol=tol, restart=25)
            #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)

            # if exitCode != 0:
            #         print(exitCode)

            f_new = f(ATy, y,  B_inv_A_trans_y)
            #g_old = np.copy(g_new)
            rate = f_new/2 + betaG + betaD * lam_p#lambdas[t+1]

        else:
            #rejcet
            lambdas[t + 1] = np.copy(lambdas[t])




        gammas[t+1] = np.random.gamma(shape = shape, scale = 1/rate)

        #deltas[t+1] = lambdas[t+1] * gammas[t+1]

    return lambdas, gammas,k



startTime = time.time()
lambdas ,gammas, k = MHwG(number_samples, burnIn, lambda0, gamma0)
elapsed = time.time() - startTime
print('MTC Done in ' + str(elapsed) + ' s')



print('acceptance ratio: ' + str(k/(number_samples+burnIn)))
deltas = lambdas * gammas
np.savetxt('samples.txt', np.vstack((gammas[burnIn::], deltas[burnIn::], lambdas[burnIn::])).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsed), fmt = '%.15f \t %.15f \t %.15f')

#delt_aav, delt_diff, delt_ddiff, delt_itau, delt_itau_diff, delt_itau_aav, delt_acorrn = uWerr(deltas, acorr=None, s_tau=1.5, fast_threshold=5000)

import matlab.engine
eng = matlab.engine.start_matlab()
eng.Run_Autocorr_Ana_MTC(nargout=0)
eng.quit()


AutoCorrData = np.loadtxt("auto_corr_dat.txt", skiprows=3, dtype='float')
#IntAutoLam, IntAutoGam , IntAutoDelt = np.loadtxt("auto_corr_dat.txt",userow = 1, skiprows=1, dtype='float'

with open("auto_corr_dat.txt") as fID:
    for n, line in enumerate(fID):
       if n == 1:
            IntAutoDelt, IntAutoGam, IntAutoLam = [float(IAuto) for IAuto in line.split()]
            break



#refine according to autocorrelation time
new_lamb = lambdas[burnIn::math.ceil(IntAutoLam)]
#SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
new_gam = gammas[burnIn::math.ceil(IntAutoGam)]
#SetGamma = new_gam[np.random.randint(low = 0,high =len(new_gam),size =1)]
new_delt = deltas[burnIn::math.ceil(IntAutoDelt)]
#SetDelta = new_delt[np.random.randint(low = 0,high =len(new_delt),size =1)]

##
import tikzplotlib

mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)

fig, axs = plt.subplots(figsize = (7,  2))
# We can set the number of bins with the *bins* keyword argument.
axs.hist(new_gam,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoGam)))
#axs[0].set_title(str(len(new_gam)) + ' effective $\gamma$ samples')
axs.set_title(str(len(new_gam)) + r' $\gamma$ samples, the noise precision')
#axs.set_xlabel(str(len(new_gam)) + ' effective $\gamma$ samples')

#tikzplotlib.save("HistoResults1.tex",axis_height='3cm', axis_width='7cm')
#plt.close()


fig, axs = plt.subplots( )
axs.hist(new_delt,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoDelt)))
axs.set_title(str(len(new_delt)) + ' $\delta$ samples, the prior precision')
#axs.set_xlabel(str(len(new_delt)) + ' $\delta$ samples, the prior precision')

#tikzplotlib.save("HistoResults2.tex",axis_height='3cm', axis_width='7cm')
#plt.close()

fig, axs = plt.subplots( )
axs.hist(new_lamb,bins=n_bins, color = 'k')#10)
#axs.set_xlabel(str(len(new_lamb)) + ' effective $\lambda =\delta / \gamma$ samples')
#axs[2].xaxis.set_major_formatter(scientific_formatter)
#axs[2].set_title(str(len(new_lamb)) + ' effective $\lambda =\delta / \gamma$ samples')
axs.set_title(str(len(new_lamb)) + ' $\lambda$ samples, the regularization parameter')
#plt.savefig('HistoResults.png')
plt.show()

#tikzplotlib.save("HistoResults3.tex",axis_height='3cm', axis_width='7cm')

##
#draw paramter samples
paraSamp = 200#n_bins
Results = np.zeros((paraSamp,len(theta)))
NormRes = np.zeros(paraSamp)
xTLxRes = np.zeros(paraSamp)
SetGammas = new_gam[np.random.randint(low=0, high=len(new_gam), size=paraSamp)]
SetDeltas  = new_delt[np.random.randint(low=0, high=len(new_delt), size=paraSamp)]

startTimeX = time.time()
for p in range(paraSamp):
    # SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
    SetGamma = SetGammas[p] #minimum[0]
    SetDelta  = SetDeltas[p] #minimum[1]
    W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)))
    v_1 = np.sqrt(SetGamma) *  A.T @ W
    W2 = np.random.multivariate_normal(np.zeros(len(L)), L)
    v_2 = np.sqrt(SetDelta) * W2

    SetB = SetGamma * ATA + SetDelta * L
    RandX = (SetGamma * ATy[0::, 0] + v_1 + v_2)

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

    Results[p, :] = B_inv_A_trans_y

    NormRes[p] = np.linalg.norm( np.matmul(A,B_inv_A_trans_y) - y[0::,0])
    xTLxRes[p] = np.sqrt(np.matmul(np.matmul(B_inv_A_trans_y.T, L), B_inv_A_trans_y))

elapsedX = time.time() - startTimeX
print('Time to solve for x ' + str(elapsedX/paraSamp))


###
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

for n in range(0,paraSamp,35):
    Sol = Results[n, :] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)

    ax1.plot(Sol,height_values,marker= '+',color = ResCol,label = 'posterior samples ', zorder = 0, linewidth = 0.5, markersize = 5)
    # with open('Samp' + str(n) +'.txt', 'w') as f:
    #     for k in range(0, len(Sol)):
    #         f.write('(' + str(Sol[k]) + ' , ' + str(height_values[k]) + ')')
    #         f.write('\n')

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


plt.show()


#tikzplotlib.save("FirstRecRes.pgf")
print('done')
