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
actual_heights = np.zeros((len(press)+1,1))
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

fig, axs = plt.subplots(tight_layout=True)
plt.plot(calc_press,actual_heights)
plt.show()


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
y, gamma  = add_noise(Ax, 10)
np.savetxt('dataY.txt', y, header = 'Data y including noise', fmt = '%.15f')
#y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))

#gamma = 1/(np.max(Ax) * 0.1)**2

''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''




fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin ,linewidth = 15 )
ax1.plot(y, tang_heights_lin ,linewidth = 15 )
plt.show()

##
"""update A so that O3 profile is constant"""
w_cross =   f_broad * 1e-4 * gaussian(height_values, 35,10) * np.max(VMR_O3)

#w_cross =   f_broad * 1e-4 * VMR_O3


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


##
'''do t-walk '''
import pytwalk

numPara = 4
tWalkSampNum = 50000
burnIn = 1000

#efit, dfit, cfit,
dfit, cfit, bfit, afit = np.polyfit(height_values, np.log(pressure_values), numPara-1)


def press(a,b,c,d,e,x):
    #a[0] = pressure_values[0]*1.75e1
    return np.exp( e * x**4 + d * x**3 + c * x**2 + b * x + a)

gamma = 1/(np.max(Ax) * 0.01)
def log_post(Params):
    a = Params[0]
    b = Params[1]
    c = Params[2]
    d = Params[3]
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
    list.append(Params[2] > 0)
    #list.append(Params[1] < 2e-1)
    list.append(Params[3] > 0)
    # list.append(Params[2] > -4e-3)
    # list.append(Params[3] > 2e-5)
    # list.append(Params[3] < 10e-5)
    # list.append(Params[4] < 0)
    # list.append(Params[4] > -6e-7)
    return all(list)

MargPost = pytwalk.pytwalk( n=numPara, U=log_post, Supp=MargPostSupp)
startTime = time.time()
x0 =  np.ones(numPara)
#x0[0] = afit
x0[0] = afit
x0[1] = bfit
x0[2] = cfit
x0[3] = dfit

# x0[numPara::]= -bfit * x0[numPara::]
# xp0 =  np.ones(2*numPara)
# xp0[:numPara] = a_curr
# xp0[numPara::]= b_curr
xp0 = 1.02 * x0
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
dfit, cfit, bfit, afit = np.polyfit(height_values, np.log(pressure_values), 3)
calc_fit_press = press(afit, bfit, cfit, dfit, 0, height_values)
t_walk_press = press(MeanParas[0], MeanParas[1] ,MeanParas[2],MeanParas[3],0, height_values)
#t_walk_press = press(afit, SampParas[2500,0] ,SampParas[2500,1], 0,0, height_values)
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# for i in range(burnIn, len(SampParas),100):
#     t_walk_press = press(afit, SampParas[i,0] ,SampParas[i,1], 0,0, height_values)
#     ax1.plot(t_walk_press, height_values, linewidth = 0.5)
ax1.plot(calc_fit_press, height_values)
ax1.plot(pressure_values, height_values )
ax1.plot(t_walk_press, height_values, linewidth = 2.5)
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




eTempfit, dTempfit, cTempfit, bTempfit, aTempfit = np.polyfit(height_values, temp_values, 4)

eTempSamp, dTempSamp, cTempSamp, bTempSamp, aTempSamp = np.polyfit(height_values[1:-1], recov_temp[1:], 4)


def temp(a,b,c,d,e,x):
    #a[0] = pressure_values[0]*1.75e1
    return  e * x**4 + d * x**3 + c * x**2 + b * x + a

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.scatter(recov_temp[1:], height_values[1:-1])
ax1.plot(temp(aTempSamp,bTempSamp,cTempSamp, dTempSamp, eTempSamp,height_values), height_values, linewidth = 2.5)
ax1.plot(temp_values, height_values, linewidth = 1.5)
ax1.plot(temp(aTempfit,bTempfit,cTempfit, dTempfit, eTempfit,height_values), height_values, linewidth = 1.5)
ax1.plot(calc_fit_temp[1:], height_values[1:-1], linewidth = 0.5)
plt.show()

print('temp calc')

##
def gaussian(x, mu, sigma):
    """
    Compute the value of the Gaussian (normal) distribution at point x.

    Parameters:
        x: float or numpy array
            The point(s) at which to evaluate the Gaussian function.
        mu: float
            The mean (average) of the Gaussian distribution.
        sigma: float
            The standard deviation (spread) of the Gaussian distribution.

    Returns:
        float or numpy array
            The value(s) of the Gaussian function at point x.
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)



fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(gaussian(height_values, 35, 10) * np.max(VMR_O3), height_values, linewidth = 2.5)
ax1.plot(VMR_O3, height_values, linewidth = 2.5)

plt.show()
