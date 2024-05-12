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


scalingConst = 1e5
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
y, gamma  = add_noise(Ax, 1)
np.savetxt('dataY.txt', y, header = 'Data y including noise', fmt = '%.15f')
ATy = np.matmul(A.T,y)
# y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
# gamma = 7.6e-5

#gamma = 1/(np.max(Ax) * 0.1)**2

''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''
##

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

A, theta_scale = composeAforO3(A_lin, temp_values, pressure_values, ind)
ATy = np.matmul(A.T, y)
ATA = np.matmul(A.T, A)

def MinLogMargPost(params):#, coeff):
    tol = 1e-8
    # gamma = params[0]
    # delta = params[1]
    gam = params[0]
    lamb = params[1]
    if lamb < 0  or gam < 0:
        return np.nan

    betaG = 1e-4
    betaD = 1e-10




    n = SpecNumLayers
    m = SpecNumMeas
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


gamma0, lam0 = optimize.fmin(MinLogMargPost, [gamma, 1/ gamma / np.var(VMR_O3) ])

#print(lam0)


fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin)
ax1.plot(y, tang_heights_lin)
plt.show()
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

grad = np.log(pressure_values[1:])- np.log(pressure_values[:-1])/(height_values[1:,0]- height_values[:-1,0])
bfitup, afitup = np.polyfit(height_values[-5:,0], grad[-5:], 1)
bfitmid, afitmid = np.polyfit(height_values[20:25,0], grad[20:25], 1)
bfitlow, afitlow = np.polyfit(height_values[0:15,0], grad[0:15], 1)


cross_heigth = (afitup - afitmid )/ (bfitmid - bfitup)

cross_heigth1 = (afitmid - afitlow )/ (bfitlow - bfitmid)

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(bfitup *  height_values[:,0] + afitup, height_values, linewidth = 2)
ax1.plot(bfitmid *  height_values[:,0] + afitmid, height_values, linewidth = 2)
ax1.plot(bfitlow *  height_values[:,0] + afitlow, height_values, linewidth = 2)
ax1.scatter(grad , height_values[1:])
ax1.set_xlabel(r'Pressure in hPa ')
ax1.set_ylabel('Height in km')
#ax1.set_xscale('log')
plt.savefig('samplesPressure.png')
plt.show()


##
breakInd = 28
numPara = 2
paraMat = np.zeros((len(height_values), numPara))
#breakInd = 21

paraMat[0:breakInd,0] = np.ones(breakInd)
paraMat[breakInd:,1] = np.ones(int(len(height_values)) -breakInd)
def pressFunc(x, b1, b2, a1, a2):
    b = paraMat @ [b1,b2]
    a = paraMat @ [a1, a2]
    #a = np.log(1013)
    return b * x + a

popt, pcov = scy.optimize.curve_fit(pressFunc, height_values[:,0], np.log(pressure_values))#, p0=[2e-2,2e-2, np.log(1013)])


tWalkSampNum = 4000
burnIn =1000



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

def pressFunc(x, b1, b2, a1, a2):
    numPara = 2
    paraMat = np.zeros((len(x), numPara))
    breakInd = 28
    paraMat[0:breakInd, 0] = np.ones(breakInd)
    paraMat[breakInd:, 1] = np.ones(int(len(x)) - breakInd)
    b = paraMat @ [b1, b2]
    a = paraMat @ [a1, a2]
    return np.exp(b * x + a)

##

'''do the sampling'''
SetDeltas = lam0 * gamma0
SetGammas = gamma0
number_samples = 2000
recov_temp_fit = np.mean(temp_values) * np.ones((SpecNumLayers,1))

round = 0
while round < 200:
    A, theta_scale = composeAforPress(A_lin, recov_temp_fit, O3_Prof, ind)
    tWalkPress(height_values, A, y, grad, popt, tWalkSampNum, burnIn, gamma)
    SampParas = np.loadtxt("MargPostDat.txt")
    MeanParas = np.mean(SampParas[burnIn:, :], 0)
    recov_press = pressFunc(height_values[:, 0], MeanParas[0], MeanParas[1], MeanParas[2], MeanParas[3])
    recov_temp_fit, recov_temp = updateTemp(height_values, temp_values, recov_press)
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


    print('temp calc')

    A,  theta_scale = composeAforO3(A_lin, recov_temp_fit, recov_press, ind)
    ATy = np.matmul(A.T, y)
    ATA = np.matmul(A.T, A)
    #gamma0, lam0 = optimize.fmin(MinLogMargPost, [gamma, gamma * np.var(VMR_O3)])
    B0 = (ATA + lam0 * L)
    B_inv_A_trans_y0, exitCode = gmres(B0, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)
    print(lam0)
    print(SetDeltas/SetGammas)
    startTime = time.time()
    lambdas ,gammas, k = MHwG(number_samples, A, burnIn, SetDeltas/SetGammas, SetGammas, y, ATA, L, B_inv_A_trans_y0, ATy, tol, betaG, betaD, B0)
    elapsed = time.time() - startTime
    print('MTC Done in ' + str(elapsed) + ' s')
    print('acceptance ratio: ' + str(k/(number_samples+burnIn)))
    deltas = lambdas * gammas

    # draw paramter samples
    paraSamp = 1  # n_bins
    Results = np.zeros((paraSamp, len(VMR_O3)))
    NormRes = np.zeros(paraSamp)
    xTLxRes = np.zeros(paraSamp)
    SetGammas = gammas[np.random.randint(low=0, high=len(gammas), size=paraSamp)]
    SetDeltas = deltas[np.random.randint(low=0, high=len(deltas), size=paraSamp)]

    startTimeX = time.time()
    for p in range(paraSamp):
        # SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
        SetGamma = SetGammas[p]  # minimum[0]
        SetDelta = SetDeltas[p]  # minimum[1]
        W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)))
        v_1 = np.sqrt(SetGamma) * A.T @ W
        W2 = np.random.multivariate_normal(np.zeros(len(L)), L)
        v_2 = np.sqrt(SetDelta) * W2

        SetB = SetGamma * ATA + SetDelta * L
        RandX = (SetGamma * ATy[0::, 0] + v_1 + v_2)

        Results[p, :], exitCode = gmres(SetB, RandX, x0=B_inv_A_trans_y0, tol=tol)

    O3_Prof = np.mean(Results,0)/ theta_scale

    # fig3, ax1 = plt.subplots(tight_layout=True, figsize=set_size(245, fraction=fraction))
    # ax1.plot(O3_Prof, height_values, linewidth=2.5, label='my guess', marker='o')
    # ax1.plot(VMR_O3, height_values, linewidth=2.5, label='true profile', marker='o')
    # ax1.set_ylabel('Height in km')
    # ax1.set_xlabel('Volume Mixing Ratio of Ozone')
    # ax2 = ax1.twiny()
    # ax2.scatter(y, tang_heights_lin, linewidth=2, marker='x', label='data', color='k')
    # ax2.set_xlabel(r'Spectral radiance in $\frac{W cm}{m^2  sr} $', labelpad=10)  # color =dataCol,
    # ax1.legend()
    # plt.savefig('DataStartTrueProfile.png')
    # plt.show()



    round += 1
    print('Round' + str(round))

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

for n in range(0,paraSamp,5):
    Sol = Results[n, :] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)

    ax1.plot(Sol,height_values,marker= '+',color = ResCol,label = 'posterior samples ', zorder = 0, linewidth = 0.5, markersize = 5)
    # with open('Samp' + str(n) +'.txt', 'w') as f:
    #     for k in range(0, len(Sol)):
    #         f.write('(' + str(Sol[k]) + ' , ' + str(height_values[k]) + ')')
    #         f.write('\n')
O3_Prof = np.mean(Results,0)/ (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)

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

fig3, ax1 = plt.subplots(tight_layout=True, figsize=set_size(245, fraction=fraction))
ax1.plot(press, heights, label='true press.')
ax1.plot(recov_press, height_values, linewidth=2.5, label='samp. press. fit')  #
ax1.set_xlabel(r'Pressure in hPa ')
ax1.set_ylabel('Height in km')
ax1.legend()
plt.savefig('samplesPressure.png')
plt.show()
##
fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
ax1.plot(recov_temp_fit, height_values, linewidth=2.5, color='r',
         label='fitted T')
fitTemp, scatTemp = updateTemp(height_values, temp_values, recov_press)
ax1.scatter(scatTemp, height_values[1:,0], color='r')
ax1.plot(fitTemp, height_values, color='r')
ax1.plot(temp_values, height_values, linewidth=5, label='true T', color='green', zorder=0)
ax1.legend()
plt.savefig('TemperatureSamp.png')
plt.show()

#tikzplotlib.save("FirstRecRes.pgf")
print('done')


