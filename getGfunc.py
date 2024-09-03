import numpy as np
import scipy as scy



def Parabel(x, h0, a0, d0):

    return a0 * np.power((h0-x),2 )+ d0

def detL(indices):
    Values = np.zeros(len(indices))
    n = len(height_values)
    for j in range(len(indices)):
        h1 = univarGrid[0][indices[j,0].astype(np.int32)]
        a0 = univarGrid[1][indices[j,1].astype(np.int32)]
        d0 = univarGrid[2][indices[j,2].astype(np.int32)]
        delta = Parabel(height_values, h1, a0, d0)
        TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * delta
        TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * delta.T
        Diag = np.eye(n) * np.sum(TriU + TriL, 0)

        L_d = -TriU + Diag - TriL
        L_d[0, 0] = 2 * L_d[0, 0]
        L_d[-1, -1] = 2 * L_d[-1, -1]

        upTriL = scy.linalg.cholesky(L_d)
        Values[j] = 2 * np.sum(np.log(np.diag(upTriL)))
    return Values

height_values = np.loadtxt('height_values.txt')
height_values = height_values.reshape((len(height_values),1))
d0Mean = 0.8e-4
d0Sigm = 75e-7
a0Mean = 1e-6
a0Sigm = 1.5e-6
hMean = 31
hSigm = 1

factor = 8
gridSize = 25

univarGrid = [np.linspace(hMean - hSigm* factor, hMean + hSigm * factor, gridSize),
              np.linspace(0, a0Mean + a0Sigm * factor, gridSize),
             np.linspace(d0Mean - d0Sigm * factor,d0Mean + d0Sigm * factor, gridSize)]


initRank = 3

dim = len(univarGrid)

import tt
import tt.cross
from tt.cross.rectcross import rect_cross
# Run cross
#random inital cores
f0 = tt.rand(gridSize, dim, r=initRank)
ttTrain = rect_cross.cross(detL, f0, nswp=50, kickrank=1, rf=2, eps=1e-5)
print(ttTrain)

TTCore = [None] * dim
# Cores of f must be extracted carefully, since we might have discontinuous ps
core = np.zeros((ttTrain.core).size, dtype=np.float64)
ps_my = 0
for i in range(0, ttTrain.d):
    np.savetxt('detLGrid' + str(i) + '.txt', univarGrid[i])
    cri = ttTrain.core[range(ttTrain.ps[i] - 1, ttTrain.ps[i + 1] - 1)]
    np.savetxt('ttdetLCore'+str(i)+'.txt', cri, header = str(ttTrain.r[i])+ ' ,'+ str( ttTrain.n[i])+ ',' +str( ttTrain.r[i + 1]) )
    TTCore[i] = cri.reshape((ttTrain.r[i] , ttTrain.n[i] , ttTrain.r[i + 1]))
    core[range(ps_my, ps_my + ttTrain.r[i] * ttTrain.n[i] * ttTrain.r[i + 1])] = cri
    ps_my = ps_my + ttTrain.r[i] * ttTrain.n[i] * ttTrain.r[i + 1]

