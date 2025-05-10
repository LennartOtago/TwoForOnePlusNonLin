import numpy as np
import matplotlib as mpl
from scipy import constants, optimize
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
from numpy.random import uniform, normal, gamma
import scipy as scy
from matplotlib.ticker import FuncFormatter
import time, pytwalk
def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = 1#(5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim



# def pressFunc(x, b1, b2, h0, p0):
#     b = np.ones(len(x))
#     b[x<=h0] = b1
#     b[x>h0] = b2
#     return np.exp(-b * (x -h0)  + np.log(p0))

def pressFunc(x, b, h0, p0):
    return np.exp(-b * (x -h0)  + np.log(p0))

def temp_func(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,b0):
    a = np.ones(x.shape)
    b = np.ones(x.shape)
    a[x < h0] = a0
    a[h0 <= x] = 0
    a[h1 <= x] = a1
    a[h2 <= x] = a2
    a[h3 <= x] = 0
    a[h4 <= x ] = a3
    a[h5 <= x ] = a4
    b[x < h0] = b0
    b[h0 <= x] = b0 + h0 * a0
    b[h1 <= x] = b0 + h0 * a0
    b[h2 <= x] = a1 * (h2-h1) + b0 + h0 * a0
    b[h3 <= x ] = a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0
    b[h4 <= x ] = a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0
    b[h5 <= x ] = a3 * (h5-h4) + a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0
    h = np.ones(x.shape)
    h[x < h0] = 0
    h[h0 <= x] = h0
    h[h1 <= x] = h1
    h[h2 <= x] = h2
    h[h3 <= x] = h3
    h[h4 <= x] = h4
    h[h5 <= x] = h5
    return a * (x - h) + b


def ErrorPressFuncTemp(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,b0, b1, b2, hp0, p0, Var):
    b = np.ones(len(x))
    varB = np.ones(len(x))
    b[x<=hp0] = b1
    b[x>hp0] = b2

    varB[x<=hp0] = Var[0]
    varB[x>hp0] = Var[1]
    F = pressFunc(x, b1, b2, hp0, p0)/temp_func(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,b0)
    return ((x-hp0)* F)**2 * varB + ( b * F)**2 * Var[2] +  (np.exp(-b * (x - hp0) )//temp_func(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,b0))**2 * Var[3]

def ErrorPressFunc(x, b, hp0, p0, Var):
    # b = np.ones(len(x))
    # varB = np.ones(len(x))
    # b[x<=hp0] = b1
    # b[x>hp0] = b2
    #
    # varB[x<=hp0] = Var[0]
    # varB[x>hp0] = Var[1]
    varB = Var[0]
    #F = pressFunc(x, b1, b2, hp0, p0)
    F = pressFunc(x, b, hp0, p0)
    return ((x-hp0)* F)**2 * varB+ ( b * F)**2 * Var[1] + np.exp(-b * (x - hp0) ) **2 * Var[2]


def ErrorOneOverTempFunc(x, h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, b0, b1, b2, hp0, p0, Var):

    ErrorMat = np.zeros((x.shape[0],12))
    pTsq =  pressFunc(x, b1, b2, hp0, p0) / temp_func(x, h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, b0) ** 2


    # h0
    ErrorMat[x < h0, 0] = 0
    ErrorMat[h0 <= x, 0] = a0 * (pTsq[h0 <= x])
    # h1
    ErrorMat[x < h1, 1] = 0
    ErrorMat[h1 <= x, 1] = a1 * (pTsq[h1 <= x])
    # h2
    ErrorMat[x < h2, 2] = 0
    ErrorMat[h2 <= x, 2] = (a1 - a2) * (pTsq[h2 <= x])
    # h3
    ErrorMat[x < h3, 3] = 0
    ErrorMat[h3 <= x, 3] = a2 * (pTsq[h3 <= x])
    # h4
    ErrorMat[x < h4, 4] = 0
    ErrorMat[h4 <= x, 4] = a3 * (pTsq[h4 <= x])
    # h5
    ErrorMat[x < h5, 5] = 0
    ErrorMat[h5 <= x, 5] = (a3-a4) * (pTsq[h5 <= x])

    # a0
    ErrorMat[x < h1, 6] = x[x < h1]
    ErrorMat[h1 <= x, 6] = h0 * (pTsq[h1 <= x])
    # a1
    ErrorMat[x < h1, 7] = 0
    ErrorMat[h1 <= x, 7] = (x[h1 <= x] - h1) * (pTsq[h1 <= x])
    ErrorMat[h2 <= x, 7] = (h2 - h1) * (pTsq[h2 <= x])
    # a2
    ErrorMat[x < h2, 8] = 0
    ErrorMat[h2 <= x, 8] = (x[h2 <= x] - h2) * (pTsq[h2 <= x])
    ErrorMat[h3 <= x, 8] = (h3 - h2) * (pTsq[h3 <= x])
    # a3
    ErrorMat[x < h4, 9] = 0
    ErrorMat[h4 <= x, 9] = (x[h4 <= x] - h4) * (pTsq[h4 <= x])
    ErrorMat[h5 <= x, 9] = (h5 - h4) * (pTsq[h5 <= x])
    # a4
    ErrorMat[x < h5, 10] = 0
    ErrorMat[h5 <= x, 10] = (x[h5 <= x] - h5) * (pTsq[h5 <= x])

    # b0* (pTsq[h5 <= x])
    ErrorMat[:, 11] = 1* (pTsq)


    return ErrorMat**2 @ Var

def g(A, L, l):
    """ calculate g"""
    B = np.matmul(A.T,A) + l * L
    #Bu, Bs, Bvh = np.linalg.svd(B)
    upL = scy.linalg.cholesky(B)
    #return np.sum(np.log(Bs))
    return 2* np.sum(np.log(np.diag(upL)))

def f(ATy, y, B_inv_A_trans_y):
    return np.matmul(y[0::,0].T, y[0::,0]) - np.matmul(ATy[0::,0].T,B_inv_A_trans_y)

""" for plotting figures,
PgWidth in points, either collumn width page with of Latex"""
def scientific(x, pos):
    # x:  tick value
    # pos: tick position
    return '%.e' % x
scientific_formatter = FuncFormatter(scientific)

fraction = 1.5
dpi = 300
PgWidthPt =  421/2
#PgWidthPt = 1/0.3 *fraction * 421/4 #phd
defBack = mpl.get_backend()
mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 10,#1/0.3 *fraction *
                     'text.usetex': True,
                     'font.family' : 'serif',
                     'font.serif'  : 'cm',
                     'text.latex.preamble': r'\usepackage{bm, amsmath}'})

""" for plotting histogram and averaging over lambda """
n_bins = 40

""" for MwG"""
burnIn = 50

betaG = 1e-10# 1e-18#
betaD = 1e-10#3#9e3#1e-3#1e-10#1e-22#  # 1e-4

import numpy as np


dir = '/home/lennartgolks/PycharmProjects/firstModelCheckPhD/'
dir = '/Users/lennart/PycharmProjects/firstModelCheckPhD/'
dir = '/Users/lennart/PycharmProjects/TTDecomposition/'
#dir = '/home/lennartgolks/PycharmProjects/TTDecomposition/'

B_inv_A_trans_y0 = np.loadtxt(dir + 'B_inv_A_trans_y0.txt')
VMR_O3 = np.loadtxt(dir + 'VMR_O3.txt')
newCondMean = np.loadtxt(dir + 'seccondMean.txt').reshape((len(VMR_O3), 1))
pressure_values = np.loadtxt(dir + 'pressure_values.txt')
temp_values = np.loadtxt(dir + 'temp_values.txt')
height_values = np.loadtxt(dir + 'height_values.txt')
A = np.loadtxt(dir + 'AMat.txt')

APressTemp = np.loadtxt(dir + 'APT.txt')
newAPT = np.loadtxt(dir + 'newAPT.txt')
gamma0 = np.loadtxt(dir + 'gamma0.txt')
y = np.loadtxt(dir + 'nonLinDataY.txt')

RealMap = np.loadtxt(dir + 'RealMap.txt')
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
newA = RealMap @ A
newATy = np.matmul(newA.T, y)
newATA = np.matmul(newA.T,newA)


Ax =np.matmul(newA, VMR_O3 * theta_scale_O3)
AxPT =np.matmul(RealMap@ APressTemp, pressure_values.reshape((n, 1))/temp_values.reshape((n, 1)) )
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin)

ax1.plot(AxPT, tang_heights_lin)
ax1.scatter(y, tang_heights_lin)
ax1.plot(y, tang_heights_lin)

plt.show()


fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction),dpi = 300)
ax1.plot(pressure_values.reshape((SpecNumLayers,1))/temp_values.reshape((SpecNumLayers,1)), height_values)
ax2 = ax1.twiny()
ax2.plot(1/temp_values.reshape((SpecNumLayers,1)), height_values)
#ax2.plot(pressure_values.reshape((SpecNumLayers,1)), height_values)
#ax1.set_xscale('log')

fig3.savefig('TruePressTemp.png')
plt.show()


##


popt, pcov = scy.optimize.curve_fit(pressFunc, height_values[:,0], pressure_values, p0=[1.5e-1, 8, pressure_values[0]])


TrueCol = [50/255,220/255, 0/255]

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
axs.plot( pressure_values,height_values,marker = 'o' ,markerfacecolor = TrueCol, color = TrueCol, label = 'true profile', zorder=0 ,linewidth = 1.5, markersize =7)
#
# axs.plot( np.exp(pressFunc(height_values[:,0], *popt)) ,height_values, markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', markersize =2, linewidth =0.5)
axs.plot(pressFunc(height_values[:, 0], *popt), height_values, markeredgecolor='k', color='k', zorder=3,
         marker='.', markersize=5, linewidth=0.5, label = 'fitted profile')
#axs.plot(np.log(pressure_values), height_values, markeredgecolor='k', color='k', zorder=3,='.', markersize=2, linewidth=0.5)
axs.set_xlabel(r'pressure in hPa')
axs.set_ylabel(r'height in km')
axs.legend()
fig.savefig('TruePress.png')
plt.show()

print(popt)

def log_postTP(params, means, sigmas, A, y, height_values, gamma0):
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

    #sigmaGrad1 = sigmas[12]
    sigmaGrad2 = sigmas[12]
    sigmaH = sigmas[13]
    sigmaP = sigmas[14]


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

    #b1 = params[12]
    b2 = params[12]
    h0P =params[13]
    p0 = params[14]
    gam = gamma0#params[15]
    paramT = [h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, b0]
    paramP = [b2, h0P, p0]
    # postDatT = - gamma0 * np.sum((y - A @ (1 / temp_func(height_values, *paramT).reshape((n, 1)))) ** 2)
    # postDatP = gamma0 * 1e-3 * np.sum((y - A @ pressFunc(height_values[:, 0], *paramP).reshape((n, 1))) ** 2)
    PT = pressFunc(height_values[:, 0], *paramP).reshape((n, 1)) /temp_func(height_values, *paramT).reshape((n, 1))
    #postDat = + SpecNumMeas / 2  * np.log(gam) - 0.5 * gam * np.sum((y - A @ PT ) ** 2)- betaG * gam
    postDat = - 0.5 * gam * np.sum((y - A @ PT) ** 2)

    #postDat = 0
    #- ((popt[0] - b1) / sigmaGrad1) ** 2
    Values =     - ((h0 - h0Mean) / h0Sigm) ** 2 - ((h1 - h1Mean) / h1Sigm) ** 2 - (
                (h2 - h2Mean) / h2Sigm) ** 2 - (
                        (h3 - h3Mean) / h3Sigm) ** 2 - ((h4 - h4Mean) / h4Sigm) ** 2 - (
                        (h5 - h5Mean) / h5Sigm) ** 2 - ((a0 - a0Mean) / a0Sigm) ** 2 - (
                        (a1 - a1Mean) / a1Sigm) ** 2 - ((a2 - a2Mean) / a2Sigm) ** 2 \
                - ((a3 - a3Mean) / a3Sigm) ** 2 - ((a4 - a4Mean) / a4Sigm) ** 2 - ((b0 - b0Mean) / b0Sigm) ** 2 \
                 - ((means[12] - b2) / sigmaGrad2) ** 2 - (
                            (means[14] - p0) / sigmaP) ** 2 \
                - ((means[13] - h0P) / sigmaH) ** 2

    return postDat + 0.5 * Values
##


means = np.loadtxt(dir + 'PTMeans.txt')
sigmas = np.loadtxt(dir + 'PTSigmas.txt')
#means = np.copy(means[:-1])
#sigmas = np.copy(sigmas[:-1])
#sigmas =  3* np.copy(sigmas)
#means = 2 * np.copy(means)
#means[11] = 2 * means[11]
#means[15] = 2 * means[15]
#sigmas[0] = 5#*3#0.5 #* 0.1
#sigmas[11] = 10# b0
#sigmas[14] =  5 # h0p
#sigmas[12] = 0.001 #sigmaGrad1
# sigmas[13] = 0.0001 #sigmaGrad2

#means[12] = popt[0]
# means[12] = popt[0]
# means[13] = popt[1]
# means[14] = popt[2]
#
# sigmaP = 2 # * 2
# sigmaH = 0.2
# #sigmaGrad1 = 0.001#0.005
# sigmaGrad2 = 0.0001#0.01 #* 5
#
# #sigmas[12] = sigmaGrad1
# sigmas[12] = sigmaGrad2
# sigmas[13] = sigmaH
# sigmas[14] = sigmaP

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
Errors = np.sqrt( ErrorPressFunc(height_values[:,0],  *means[12:], sigmas[12:]**2))
ax1.errorbar(pressure_values.reshape((SpecNumLayers)), height_values, xerr = Errors )
ax1.plot(pressure_values, height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0 ,linewidth = 1.5, markersize =7)


plt.show()

##
fig3, ax1 = plt.subplots( figsize=(PgWidthPt/ 72.27, 2*PgWidthPt / 72.27), tight_layout = True,dpi = 300)#,figsize=(4,8))
x = np.linspace(5,90,1000)
ax1.plot(np.exp(-0.5 * (x - means[0])**2 / (sigmas[0]*1) **2),x, label = "$h_{1}$")
ax1.plot(np.exp(-0.5 * (x - means[1])**2 / sigmas[1] **2),x, label = "$h_{2}$")
ax1.plot(np.exp(-0.5 * (x - means[2])**2 / sigmas[2] **2),x, label = "$h_{3}$")
ax1.plot(np.exp(-0.5 * (x - means[3])**2 / (sigmas[3]*1) **2),x, label = "$h_{4}$")
ax1.plot(np.exp(-0.5 * (x - means[4])**2 / (sigmas[4]*1) **2),x, label = "$h_{5}$")
ax1.plot(np.exp(-0.5 * (x - means[5])**2 / (sigmas[5]*1) **2),x, label = "$h_{6}$")
ax1.set_ylabel(r'height in km')
ax1.legend()
fig3.savefig('HeightPriors.png')
plt.show()

##

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(temp_values, height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0 ,linewidth = 1.5, markersize =7)
ax1.set_xlabel(r'temperature in K')
ax1.set_ylabel(r'height in km')
ax1.legend()
fig3.savefig('TrueTemp.png')

plt.show()

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(VMR_O3, height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0, linewidth = 1.5,  markersize =7)
ax1.set_xlabel(r'ozone volume mixing ratio')
ax1.set_ylabel(r'height in km')
ax1.legend()
fig3.savefig('TrueO3.png')

plt.show()



## prior analyis

TrueCol =  [50/255,220/255, 0/255]
tests = 100
alpha = 0.4

binCol = 'C0'
PriorSamp = np.random.multivariate_normal(means, np.eye(len(sigmas))*sigmas, tests)

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,dpi = 300)
axs.plot( pressure_values,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0,linewidth = 3, markersize =15)
Sol = pressFunc(height_values[:, 0], *PriorSamp[0, 12:])
axs.plot(Sol, height_values, markeredgecolor=binCol, color=binCol, zorder=1, marker='.', markersize=2, linewidth=0.25 , label = 'prior sample', alpha = alpha)
#axs.scatter(popt[2], popt[1], color='r')

for r in range(1, tests):

    Sol = pressFunc(height_values[:,0], *PriorSamp[r, 12:])
    axs.plot( Sol ,height_values, markeredgecolor = binCol, color = binCol ,zorder=1, marker = '.', markersize =2, linewidth =0.25, alpha = alpha)


axs.set_xlabel(r'pressure in hPa')

axs.set_ylabel(r'height in km')
axs.legend()
plt.savefig('PriorPressPostMeanSigm.png')
plt.show()

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,dpi = 300)
axs.plot( temp_values,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0 ,linewidth = 3, markersize =15)
Sol = temp_func(height_values[:, 0], *PriorSamp[0, :12])
axs.plot(Sol, height_values, markeredgecolor=binCol, color=binCol, zorder=1, marker='.', markersize=2, linewidth=0.25, label = 'prior sample', alpha = alpha)

for r in range(0, tests):
    Sol = temp_func(height_values[:,0], *PriorSamp[r, :12])
    axs.plot( Sol ,height_values , markeredgecolor =binCol, color = binCol ,zorder=1, marker = '.', markersize =2, linewidth =0.25, alpha = alpha)

axs.set_xlabel(r'temperature in K ')

axs.set_ylabel(r'height in km')
axs.legend()
plt.savefig('PriorTempPostMeanSigm.png')
plt.show()

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,dpi = 300)
axs.plot( 1/temp_values,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0 ,linewidth = 3, markersize =15)
Sol = temp_func(height_values[:, 0], *PriorSamp[0, :12])
axs.plot(1 / Sol, height_values, markeredgecolor=binCol, color=binCol, zorder=1, marker='.', markersize=2, linewidth=0.25, label = 'prior sample', alpha = alpha)

for r in range(1, tests):

    Sol = temp_func(height_values[:,0], *PriorSamp[r, :12])
    axs.plot(1/ Sol ,height_values , markeredgecolor =binCol, color = binCol ,zorder=1, marker = '.', markersize =2, linewidth =0.25, alpha = alpha)

axs.set_xlabel(r'temperature in 1/K ')

axs.set_ylabel(r'height in km')
axs.legend()
plt.savefig('PriorOverTempPost.png')
plt.show()

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,dpi = 300)
axs.plot( pressure_values.reshape((SpecNumLayers))/temp_values.reshape((SpecNumLayers)),height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile',linewidth = 3, markersize =15, zorder=0 )
SolP = pressFunc(height_values[:, 0], *PriorSamp[0, 12:])
SolT = temp_func(height_values[:, 0], *PriorSamp[0, :12])
axs.plot(SolP / SolT, height_values, markeredgecolor=binCol, color=binCol, zorder=1, marker='.', markersize=2, linewidth=0.25, label = 'prior sample', alpha = alpha)

for r in range(1, tests):
    SolP = pressFunc(height_values[:, 0], *PriorSamp[r, 12:])
    SolT = temp_func(height_values[:,0], *PriorSamp[r, :12])
    axs.plot( SolP/SolT ,height_values , markeredgecolor =binCol, color = binCol ,zorder=1, marker = '.', markersize =2, linewidth =0.25, alpha = alpha)

axs.set_xlabel(r'pressure/temperature in hPa/K ')

axs.set_ylabel(r'height in km')
axs.legend()
plt.savefig('PriorTempOverPostMeanSigm.png')
plt.show()


##
GamSamp = np.loadtxt(dir + 'GamSamp.txt')
log_post = lambda params: -log_postTP(params, means, sigmas, newAPT, y, height_values, GamSamp)

import glob

dim = len(glob.glob(dir + 'uniVarGridPT*.txt'))

univarGrid = [None] * dim
for i in range(0, dim):
    univarGrid[i] = np.loadtxt(dir+'uniVarGridPT' +str(i)+ '.txt')

# def MargPostSupp(Params):
#     list = []
#     return all(list)

def MargPostSupp(Params):
    list = []
    #list.append( Params[5] >Params[4] >Params[3] >Params[2] >Params[1] >Params[0] )
    list.append(univarGrid[0][-1] > Params[0] > univarGrid[0][0])
    list.append(univarGrid[1][-1] > Params[1] > univarGrid[1][0])
    list.append(univarGrid[2][-1] > Params[2] > univarGrid[2][0])
    list.append(univarGrid[3][-1] > Params[3] > univarGrid[3][0])
    list.append(univarGrid[4][-1] > Params[4] > univarGrid[4][0])
    list.append(univarGrid[5][-1] > Params[5] > univarGrid[5][0])
    list.append(univarGrid[6][0] < Params[6] < univarGrid[6][-1])
    list.append(univarGrid[7][-1] > Params[7] > univarGrid[7][0])
    list.append(univarGrid[8][-1] > Params[8] > univarGrid[8][0])
    list.append(univarGrid[9][0] < Params[9] < univarGrid[9][-1])
    list.append(univarGrid[10][0] < Params[10] < univarGrid[10][-1])
    list.append(univarGrid[11][-1] > Params[11] > univarGrid[11][0])
    list.append(univarGrid[12][-1] > Params[12] > univarGrid[12][0])
    list.append(univarGrid[13][-1] > Params[13] > univarGrid[13][0])
    list.append(univarGrid[14][-1] > Params[14] > univarGrid[14][0])
    #list.append(univarGrid[15][-1] >Params[15] > univarGrid[15][0])
    #list.append((320>temp_func(height_values, *Params[:12])).all())
    #list.append((temp_func(height_values, *Params[:12]) > 120).all())
    return all(list)

# def MargPostSupp(Params):
#     list = []
#     list.append(Params[0] > 0)
#     list.append(Params[1] > 0)
#     list.append(Params[2] > 0)
#     list.append(Params[3] > 0)
#     list.append(Params[4] > 0)
#     list.append(Params[5] > 0)
#     list.append(Params[6] < 0)
#     list.append(Params[7] > 0)
#     list.append(Params[8] > 0)
#     list.append(Params[9] < 0)
#     list.append(Params[10] < 0)
#     list.append(Params[11] > 0)
#     list.append(1 >Params[12] > 0)
#     list.append(1 >Params[13] > 0)
#     list.append(Params[14] > 0)
#     list.append(Params[15] > 0)
#     #list.append(1 > Params[16] > 0)
#
#     return all(list)
x0 = np.append(means, gamma0)
x0 = means
xp0 = 0.9999999 * x0
dim = len(x0)
burnIn = 10000
tWalkSampNum = 1000000

MargPost = pytwalk.pytwalk(n=dim, U=log_post, Supp=MargPostSupp)


#print(" Support of Starting points:" + str(MargPostSupp(x0)) + str(MargPostSupp(xp0)))
startTime = time.time()
MargPost.Run(T=tWalkSampNum + burnIn, x0=x0, xp0=xp0)
elapsTime = time.time() - startTime
print(f'elapsed time : {elapsTime/60}')

SampParas = MargPost.Output
np.savetxt('SampParas.txt',SampParas,  fmt = '%.30f')

fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)#tight_layout = True,
#axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
for i in range(0,3):

    axs[i].hist(SampParas[burnIn:,i],bins=n_bins)
    axs[i].axvline(means[i], color = 'red')
    axs[i].set_yticklabels([])


axs[0].set_xlabel('$h_0$')
axs[1].set_xlabel('$h_1$')
axs[2].set_xlabel('$h_2$')

fig.savefig('TempPostHistSamp0.png')



fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)#tight_layout = True,

for i in range(3,6):
    axs[i-3].hist(SampParas[burnIn:,i],bins=n_bins)
    axs[i-3].axvline(means[i], color='red')
    axs[i-3].set_yticklabels([])

axs[0].set_xlabel('$h_3$')
axs[1].set_xlabel('$h_4$')
axs[2].set_xlabel('$h_5$')

fig.savefig('TempPostHistSamp1.png')



fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)

for i in range(6,9):
    axs[i-6].hist(SampParas[burnIn:,i],bins=n_bins)
    axs[i-6].axvline(means[i], color='red')
    axs[i-6].set_yticklabels([])
axs[0].set_xlabel('$a_0$')
axs[1].set_xlabel('$a_1$')
axs[2].set_xlabel('$a_2$')

fig.savefig('TempPostHistSamp2.png')


fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
for i in range(9,12):

    axs[i-9].hist(SampParas[burnIn:,i],bins=n_bins)
    axs[i-9].axvline(means[i], color='red')
    axs[i-9].set_yticklabels([])

axs[0].set_xlabel('$a_3$')
axs[1].set_xlabel('$a_4$')
axs[2].set_xlabel('$b_0$')
#axs[3].set_xlabel('$\gamma$')

fig.savefig('TempPostHistSamp3.png')

fig, axs = plt.subplots(4,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
for i in range(12,dim):
    axs[i-12].hist(SampParas[burnIn:, i], bins=n_bins)
    axs[i-12].axvline(means[i], color='red')
    axs[i-12].set_yticklabels([])
y_val = SampParas[burnIn:, -1][SampParas[burnIn:, -1] < 2.5* np.mean(SampParas[burnIn:, -1]) ]
axs[3].plot(range(len(y_val)), y_val)
axs[0].set_xlabel('$b$')
#axs[1].set_xlabel('$b_2$')
axs[1].set_xlabel('$h_0$')
axs[2].set_xlabel('$p_0$')
#axs[4].set_xlabel('$\gamma$')
fig.savefig('PressPostHistSamp4.png')
plt.show()
print(np.mean(SampParas[burnIn:, -1]))
print('done')
##
TrueCol = [50/255,220/255, 0/255]
tests = 250

indcies = np.random.randint(low=burnIn, high=burnIn+tWalkSampNum, size=tests)

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
axs.plot( pressure_values,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0,linewidth = 1.5, markersize =7)
Sol = pressFunc(height_values[:,0], *SampParas[indcies[0], 12:- 1])
axs.plot( Sol ,height_values, markeredgecolor ='k', color = 'k' ,zorder=1, marker = '.', markersize =2, linewidth =0.5, label = 'posterior sample')

for r in range(1, tests):

    Sol = pressFunc(height_values[:,0], *SampParas[indcies[r], 12:- 1])
    axs.plot( Sol ,height_values, markeredgecolor ='k', color = 'k' ,zorder=1, marker = '.', markersize =2, linewidth =0.5, label = '')


axs.set_xlabel(r'pressure in hPa')

axs.set_ylabel(r'height in km')
axs.legend()
plt.savefig('PressPostMeanSigm.png')
plt.show()

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
axs.plot( temp_values,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=1 ,linewidth = 1.5, markersize =7)
Sol = temp_func(height_values[:, 0], *SampParas[indcies[0], :12])
axs.plot(Sol, height_values, markeredgecolor='k', color='k', zorder=0, marker='.', markersize=2, linewidth=0.5, label = 'posterior sample')

for r in range(1, tests):

    Sol = temp_func(height_values[:,0], *SampParas[indcies[r], :12])
    axs.plot( Sol ,height_values , markeredgecolor ='k', color = 'k' ,zorder=0, marker = '.', markersize =2, linewidth =0.5)

axs.set_xlabel(r'temperature in K ')

axs.set_ylabel(r'height in km')
axs.legend()
plt.savefig('TempPostMeanSigm.png')
plt.show()

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
axs.plot( 1/temp_values,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=1 ,linewidth = 1.5, markersize =7)
Sol = temp_func(height_values[:, 0], *SampParas[indcies[0], :12])
axs.plot(1 / Sol, height_values, markeredgecolor='k', color='k', zorder=0, marker='.', markersize=2, linewidth=0.5, label = 'posterior sample')

for r in range(1, tests):

    Sol = temp_func(height_values[:,0], *SampParas[indcies[r], :12])
    axs.plot(1/ Sol ,height_values , markeredgecolor ='k', color = 'k' ,zorder=0, marker = '.', markersize =2, linewidth =0.5)

axs.set_xlabel(r'temperature in 1/K ')

axs.set_ylabel(r'height in km')
axs.legend()
plt.savefig('OverTempPost.png')
plt.show()

##
fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
axs.plot( pressure_values.reshape((SpecNumLayers))/temp_values.reshape((SpecNumLayers)),height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile',linewidth = 1.5, markersize =7, zorder=0 )
SolP = pressFunc(height_values[:, 0], *SampParas[indcies[0], 12:- 1])
SolT = temp_func(height_values[:, 0], *SampParas[indcies[0], :12])
axs.plot(SolP / SolT, height_values, markeredgecolor='k', color='k', zorder=1, marker='.', markersize=2, linewidth=0.5, label = 'posterior sample')

for r in range(1, tests):
    SolP = pressFunc(height_values[:, 0], *SampParas[indcies[r], 12:- 1])
    SolT = temp_func(height_values[:,0], *SampParas[indcies[r], :12])
    axs.plot( SolP/SolT ,height_values , markeredgecolor ='k', color = 'k' ,zorder=1, marker = '.', markersize =2, linewidth =0.5)

#axs.errorbar(pressure_values.reshape((SpecNumLayers))/temp_values.reshape((SpecNumLayers)), height_values, xerr = ErrorsPT )

axs.set_xlabel(r'pressure/temperature in hPa/K ')

axs.set_ylabel(r'height in km')
axs.legend()
plt.savefig('TempOverPostMeanSigm.png')
plt.show()




print(min(SampParas[:, 12]))
