import numpy as np
import matplotlib as mpl
from scipy import constants, optimize
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
from numpy.random import uniform, normal, gamma
import scipy as scy
from matplotlib.ticker import FuncFormatter
import time, pytwalk
from puwr import tauint

import os
from pathlib import Path
cwd = os.getcwd()
path = Path(cwd)
parentDir = str( path.parent.absolute())

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

# def pressFunc(x, b, h0, p0):
#     return np.exp(-b * (x -h0)  + np.log(p0))
def pressFunc(x, b, p0):
    return np.exp(-b * x  + np.log(p0))

def pressFuncFullFit(x, b1, b2, h0, p0):
    b = np.ones(len(x))
    b[x<=h0] = b1
    b[x>h0] = b2
    return np.exp(-b * (x -h0)  + np.log(p0))

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

#betaG = 1e-10# 1e-18#
#betaD = 1e-10#3#9e3#1e-3#1e-10#1e-22#  # 1e-4

import numpy as np


dir = '/home/lennartgolks/PycharmProjects/firstModelCheckPhD/'
dir = '/Users/lennart/PycharmProjects/firstModelCheckPhD/'
dir = '/Users/lennart/PycharmProjects/TTDecomposition/'
dir = parentDir + '/TTDecomposition/'
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
AxPT = APressTemp @ (pressure_values.reshape((n))/temp_values.reshape((n)))
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin)

ax1.plot(newAPT @ (pressure_values.reshape((n, 1))/temp_values.reshape((n, 1))), tang_heights_lin)
ax1.plot(AxPT, tang_heights_lin)
ax1.scatter(y, tang_heights_lin)
#ax1.plot(y, tang_heights_lin)

plt.show()


# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(pressure_values.reshape((SpecNumLayers,1))/temp_values.reshape((SpecNumLayers,1)), height_values)
# ax2 = ax1.twiny()
# ax2.plot(1/temp_values.reshape((SpecNumLayers,1)), height_values)
# #ax2.plot(pressure_values.reshape((SpecNumLayers,1)), height_values)
# #ax1.set_xscale('log')
#
# fig3.savefig('TruePressTemp.png',dpi = dpi)
# plt.show()


##


#popt, pcov = scy.optimize.curve_fit(pressFuncFullFit, height_values[:,0], pressure_values, p0=[1.5e-1, 8, pressure_values[0]])
popt = np.loadtxt(dir + 'popt.txt')

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
fig.savefig('TruePress.png',dpi = dpi)
plt.show()

print(popt)


def log_postTP(params, means, sigmas, A, y, height_values, gamma0):
    n = len(height_values)
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

    #h6Mean = means[6]
    #h6Sigm = sigmas[6]

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
    #sigmaGrad2 = sigmas[13]
    #sigmaH = sigmas[13]
    sigmaP = sigmas[0]
    pmean = means[0]



    h1 = params[6]
    h2 = params[4]
    h3 = params[10]
    h4 = params[12]
    h5 = params[14]
    #h6 = params[6]
    a0 = params[7]
    a1 = params[5]
    a2 = params[8]
    a3 = params[9]
    a4 = params[11]
    a5 = params[13]
    a6 = params[15]
    b0 = params[1]
    h0 = params[2]
    #b1 = params[12]
    b2 = params[3]
    #h0P =params[13]
    p0 = params[0]
    gam = gamma0#params[15]
    paramT = [h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, a5, a6, b0]
    paramP = [b2, p0]
    #- ((h6 - h6Mean) / h6Sigm) ** 2
    # postDatT = - gamma0 * np.sum((y - A @ (1 / temp_func(height_values, *paramT).reshape((n, 1)))) ** 2)
    # postDatP = gamma0 * 1e-3 * np.sum((y - A @ pressFunc(height_values[:, 0], *paramP).reshape((n, 1))) ** 2)
    PT = pressFunc(height_values[:, 0], *paramP).reshape((n, 1)) / temp_func(height_values, *paramT).reshape((n, 1))
    #postDat = + SpecNumMeas / 2  * np.log(gam) - 0.5 * gam * np.sum((y - A @ PT ) ** 2)- betaG * gam
    postDat = - 0.5 * gam * np.sum((y - A @ PT) ** 2)

    #postDat = 0
    #- ((popt[0] - b1) / sigmaGrad1) ** 2
    Values =     (- ((h0 - h0Mean) / h0Sigm) ** 2 - ((h1 - h1Mean) / h1Sigm) ** 2 - (
                (h2 - h2Mean) / h2Sigm) ** 2 - (
                        (h3 - h3Mean) / h3Sigm) ** 2 - ((h4 - h4Mean) / h4Sigm) ** 2 - (
                        (h5 - h5Mean) / h5Sigm) ** 2  - ((a0 - a0Mean) / a0Sigm) ** 2 - (
                        (a1 - a1Mean) / a1Sigm) ** 2 - ((a2 - a2Mean) / a2Sigm) ** 2
                - ((a3 - a3Mean) / a3Sigm) ** 2 - ((a4 - a4Mean) / a4Sigm) ** 2- ((a5 - a5Mean) / a5Sigm) ** 2
                  - ((a6 - a6Mean) / a6Sigm) ** 2 - ((b0 - b0Mean) / b0Sigm) ** 2
                 - ((bmean - b2) / sigmaGrad1) ** 2 - (
                            (pmean - p0) / sigmaP) ** 2)
                #- ((means[13] - h0P) / sigmaH) ** 2

    return postDat + 0.5 * Values
##


means = np.loadtxt(dir + 'PTMeans.txt')
sigmas = np.loadtxt(dir + 'PTSigmas.txt')
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

# h6Mean = means[6]
# h6Sigm = sigmas[6]

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
# sigmaGrad2 = sigmas[13]
# sigmaH = sigmas[13]
sigmaP = sigmas[0]
pmean = means[0]



##

fig3, ax1 = plt.subplots( figsize=(PgWidthPt/ 72.27, 2*PgWidthPt / 72.27), tight_layout = True)#,figsize=(4,8))
x = np.linspace(5,90,1000)
ax1.plot(np.exp(-0.5 * (x - h0Mean)**2 / h0Sigm **2),x, label = "$h_{1}$")
ax1.plot(np.exp(-0.5 * (x - h1Mean)**2 / h1Sigm **2),x, label = "$h_{2}$")
ax1.plot(np.exp(-0.5 * (x - h2Mean)**2 / h2Sigm **2),x, label = "$h_{3}$")
ax1.plot(np.exp(-0.5 * (x - h3Mean)**2 / h3Sigm **2),x, label = "$h_{4}$")
ax1.plot(np.exp(-0.5 * (x - h4Mean)**2 / h4Sigm **2),x, label = "$h_{5}$")
ax1.plot(np.exp(-0.5 * (x - h5Mean)**2 / h5Sigm **2),x, label = "$h_{6}$")
#ax1.plot(np.exp(-0.5 * (x - means[6])**2 / (sigmas[6]) **2),x, label = "$h_{7}$")
ax1.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
ax1.set_ylabel(r'height in km')
ax1.set_xlim(0)
ax1.legend()
fig3.savefig('HeightPriors.png',dpi = dpi)
plt.show(block = True)



##

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(temp_values, height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0 ,linewidth = 1.5, markersize =7)
ax1.set_xlabel(r'temperature in K')
ax1.set_ylabel(r'height in km')
ax1.legend()
fig3.savefig('TrueTemp.png',dpi = dpi)

plt.show()

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(VMR_O3, height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0, linewidth = 1.5,  markersize =7)
ax1.set_xlabel(r'ozone volume mixing ratio')
ax1.set_ylabel(r'height in km')
ax1.legend()
fig3.savefig('TrueO3.png',dpi = dpi)

plt.show()



## prior analyis

TrueCol =  [50/255,220/255, 0/255]
tests = 100
alpha = 0.4

binCol = 'C0'
PriorSamp = np.random.multivariate_normal(means, np.eye(len(sigmas))*sigmas**2, tests)

h1 = PriorSamp[:,6]
h2 = PriorSamp[:,4]
h3 = PriorSamp[:,10]
h4 = PriorSamp[:,12]
h5 = PriorSamp[:,14]
# h6 = params[6]
a0 = PriorSamp[:,7]
a1 = PriorSamp[:,5]
a2 = PriorSamp[:,8]
a3 = PriorSamp[:,9]
a4 = PriorSamp[:,11]
a5 = PriorSamp[:,13]
a6 = PriorSamp[:,15]
b0 = PriorSamp[:,1]
h0 = PriorSamp[:,2]
# b1 = params[12]
b2 = PriorSamp[:,3]
# h0P =params[13]
p0 = PriorSamp[:,0]

paramT = np.array([h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, a5, a6, b0])
paramP = np.array([b2, p0])

##
fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)
axs.plot( pressure_values,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0,linewidth = 3, markersize =15)
Sol = pressFunc(height_values[:, 0], *paramP[:,0])
axs.plot(Sol, height_values, markeredgecolor=binCol, color=binCol, zorder=1, marker='.', markersize=2, linewidth=0.25 , label = 'prior sample', alpha = alpha)
#axs.scatter(popt[2], popt[1], color='r')

for r in range(1, tests):

    Sol = pressFunc(height_values[:,0], *paramP[:,r])
    axs.plot( Sol ,height_values, markeredgecolor = binCol, color = binCol ,zorder=1, marker = '.', markersize =2, linewidth =0.25, alpha = alpha)


axs.set_xlabel(r'pressure in hPa')

axs.set_ylabel(r'height in km')
axs.legend()
plt.savefig('PriorPressPostMeanSigm.png',dpi = dpi)
plt.show()

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)
axs.plot( temp_values,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0 ,linewidth = 3, markersize =15)
Sol = temp_func(height_values[:, 0], *paramT[:,0])
axs.plot(Sol, height_values, markeredgecolor=binCol, color=binCol, zorder=1, marker='.', markersize=2, linewidth=0.25, label = 'prior sample', alpha = alpha)

for r in range(0, tests):
    Sol = temp_func(height_values[:,0],*paramT[:,r])
    axs.plot( Sol ,height_values , markeredgecolor =binCol, color = binCol ,zorder=1, marker = '.', markersize =2, linewidth =0.25, alpha = alpha)

axs.set_xlabel(r'temperature in K ')

axs.set_ylabel(r'height in km')
axs.legend()
plt.savefig('PriorTempPostMeanSigm.png',dpi = dpi)
plt.show()

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)
axs.plot( 1/temp_values,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0 ,linewidth = 3, markersize =15)
Sol = temp_func(height_values[:, 0],*paramT[:,0])
axs.plot(1 / Sol, height_values, markeredgecolor=binCol, color=binCol, zorder=1, marker='.', markersize=2, linewidth=0.25, label = 'prior sample', alpha = alpha)

for r in range(1, tests):

    Sol = temp_func(height_values[:,0], *paramT[:,r])
    axs.plot(1/ Sol ,height_values , markeredgecolor =binCol, color = binCol ,zorder=1, marker = '.', markersize =2, linewidth =0.25, alpha = alpha)

axs.set_xlabel(r'temperature in 1/K ')

axs.set_ylabel(r'height in km')
axs.legend()
plt.savefig('PriorOverTempPost.png',dpi = dpi)
plt.show()
##

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)
axs.plot( pressure_values.reshape((SpecNumLayers))/temp_values.reshape((SpecNumLayers)),height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile',linewidth = 3, markersize =15, zorder=0 )
SolP = pressFunc(height_values[:, 0], *paramP[:,0])
SolT = temp_func(height_values[:, 0],*paramT[:,0])
axs.plot(SolP / SolT, height_values, markeredgecolor=binCol, color=binCol, zorder=1, marker='.', markersize=2, linewidth=0.25, label = 'prior sample', alpha = alpha)

for r in range(1, tests):
    SolP = pressFunc(height_values[:, 0], *paramP[:,r])
    SolT = temp_func(height_values[:,0],*paramT[:,r])
    axs.plot( SolP/SolT ,height_values , markeredgecolor =binCol, color = binCol ,zorder=1, marker = '.', markersize =2, linewidth =0.25, alpha = alpha)

axs.set_xlabel(r'pressure/temperature in hPa/K ')

axs.set_ylabel(r'height in km')
axs.legend()
plt.savefig('PriorTempOverPostMeanSigm.png',dpi = dpi)
plt.show(block = True)

##
means = np.loadtxt(dir + 'PTMeans.txt')
sigmas = np.loadtxt(dir + 'PTSigmas.txt')
GamSamp = np.loadtxt(dir + 'GamSamp.txt')
log_post = lambda params: -log_postTP(params, means, sigmas, newAPT, y, height_values, GamSamp)

import glob

dim = len(glob.glob(dir + 'uniVarGridPT*.txt'))

univarGrid = [None] * dim
for i in range(0, dim):
    univarGrid[i] = np.loadtxt(dir+'uniVarGridPT' +str(i)+ '.txt')

def MargPostSupp(Params):
    list = []
    return all(list)

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
    return all(list)
##
#sigmas[0] = np.copy(sigmas[0])*10
x0 = means+ sigmas * 3e-1
xp0 =  means + sigmas * 2e-1
dim = len(x0)
burnIn = 100*2000
tWalkSampNum = 2000000

MargPost = pytwalk.pytwalk(n=dim, U=log_post, Supp=MargPostSupp)
print(" Support of Starting points:" + str(MargPostSupp(x0)) + str(MargPostSupp(xp0)))
##
#print(" Support of Starting points:" + str(MargPostSupp(x0)) + str(MargPostSupp(xp0)))
startTime = time.time()
MargPost.Run(T=tWalkSampNum + burnIn, x0=x0, xp0=xp0)
elapsTime = time.time() - startTime
print(f'elapsed time : {elapsTime/60}')

SampParas = MargPost.Output
np.savetxt('SampParas.txt',SampParas,  fmt = '%.30f')
np.savetxt('Twalktime.txt',[elapsTime/60], fmt = '%.30f', delimiter= '\t')

## int AutoCorrelation time
Uwerrmean = np.zeros(len(univarGrid))
Uwerrdelta = np.zeros(len(univarGrid))
Uwerrtint = np.zeros(len(univarGrid))
Uwerrd_tint = np.zeros(len(univarGrid))

# plt.rcParams.update({'font.size': 10,#1/0.3 *fraction *
#                      'text.usetex': True,
#                      'font.family' : 'serif',
#                      'font.serif'  : 'cm',
#                      'text.latex.preamble': r'\usepackage{bm, amsmath}',
#                      'figure.figsize' : set_size(PgWidthPt, fraction=fraction),
#                      'figure.autolayout': True})


for i in range(0, len(univarGrid)):
    Uwerrmean[i], Uwerrdelta[i], Uwerrtint[i], Uwerrd_tint[i] = tauint([[SampParas[burnIn:, i]]], 0)
    #print(np.correlate(SampParas[burnIn:, i],SampParas[burnIn:, i]))

np.savetxt('TwalkUwerrmean.txt', Uwerrmean,  fmt = '%.30f')
np.savetxt('TwalkUwerrdelta.txt', Uwerrdelta,  fmt = '%.30f')
np.savetxt('TwalkUwerrtint.txt', Uwerrtint,  fmt = '%.30f')
np.savetxt('TwalkUwerrd_tint.txt', Uwerrd_tint,  fmt = '%.30f')




##


fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)#tight_layout = True,
#axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
for i in range(0,3):

    axs[i].hist(SampParas[burnIn:,i],bins=n_bins)
    axs[i].axvline(means[i], color = 'red')
    axs[i].set_yticklabels([])


axs[0].set_xlabel('$h_0$')
axs[1].set_xlabel('$h_1$')
axs[2].set_xlabel('$h_2$')

fig.savefig('TempPostHistSamp0.png',dpi = dpi)



fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)#tight_layout = True,

for i in range(3,6):
    axs[i-3].hist(SampParas[burnIn:,i],bins=n_bins)
    axs[i-3].axvline(means[i], color='red')
    axs[i-3].set_yticklabels([])

axs[0].set_xlabel('$h_3$')
axs[1].set_xlabel('$h_4$')
axs[2].set_xlabel('$h_5$')
#axs[3].set_xlabel('$h_6$')
fig.savefig('TempPostHistSamp1.png',dpi = dpi)



fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)

for i in range(6,9):
    axs[i-6].hist(SampParas[burnIn:,i],bins=n_bins)
    axs[i-6].axvline(means[i], color='red')
    axs[i-6].set_yticklabels([])
axs[0].set_xlabel('$a_0$')
axs[1].set_xlabel('$a_1$')
axs[2].set_xlabel('$a_2$')

fig.savefig('TempPostHistSamp2.png',dpi = dpi)


fig, axs = plt.subplots(3,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
for i in range(9,12):

    axs[i-9].hist(SampParas[burnIn:,i],bins=n_bins)
    axs[i-9].axvline(means[i], color='red')
    axs[i-9].set_yticklabels([])

axs[0].set_xlabel('$a_3$')
axs[1].set_xlabel('$a_4$')
axs[2].set_xlabel('$a_5$')
#axs[3].set_xlabel('$\gamma$')

fig.savefig('TempPostHistSamp3.png',dpi = dpi)

fig, axs = plt.subplots(4,1, figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True,)
for i in range(12,dim):
    axs[i-12].hist(SampParas[burnIn:, i], bins=n_bins)
    axs[i-12].axvline(means[i], color='red')
    axs[i-12].set_yticklabels([])
#y_val = SampParas[burnIn:, -1][SampParas[burnIn:, -1] < 2.5* np.mean(SampParas[burnIn:, -1]) ]
#axs[2].plot(range(len(y_val)), y_val)
axs[0].set_xlabel('$a_6$')
axs[1].set_xlabel('$T_0$')
axs[2].set_xlabel('$b$')
#axs[1].set_xlabel('$b_2$')
#axs[1].set_xlabel('$h_0$')
axs[3].set_xlabel('$p_0$')
#axs[4].set_xlabel('$\gamma$')
fig.savefig('TempPostHistSamp4.png',dpi = dpi)
plt.show()
print(np.mean(SampParas[burnIn:, -1]))

fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction))
y_val  = SampParas[burnIn:, -1]
axs.plot(range(len(y_val)), y_val, color = 'k', linewidth = 0.1)
axs.set_xlabel('number of samples')
axs.set_ylabel(r'$\ln {\pi(\cdot|\gamma,\bm{y})}$')
fig.savefig('TraceTWalk.png',dpi = dpi)
plt.show()


print('done')