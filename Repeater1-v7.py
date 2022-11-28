# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:24:23 2021

@author: ajebje
"""

import numpy as np
import math
from sympy.utilities.iterables import multiset_permutations
import matplotlib.pyplot as plt
import scipy.special as ss
import scipy.sparse
import random
import scipy.optimize
import QuantumOpticsV7 as QO
from scipy.optimize import brentq


def _buildKet(l):
    # Builds a ket from a kronecker product of a list of 1 and 0
    v0 = np.array([1,0],dtype=np.complex128)
    v1 = np.array([0,1],dtype=np.complex128)
    out = 1
    for j in l:
        if j==0:
            out = np.kron(out,v0)
        elif j==1:
            out = np.kron(out,v1)
    return out.reshape(-1,1)

def _err(n,l,eta):
    # Computes epsilon(n,l) given in eq. 39
    return math.sqrt(ss.binom(n,n-l))*eta**((n-l)/2)*(1-eta)**(l/2)

def _c(n,avg,phi): 
    # Computes the TMSV amplitudes eq. 4
    return (-np.exp(1j*phi))**n*np.sqrt(avg**n/(1+avg)**(n+1))

def _beta(n,N,theta):
    # Computes the sinusoidal dependence eq. 29
    return math.sin(theta)**(N-n)*math.cos(theta)**n

def _Delta(n,N,theta):
    # Computes the optical-to-solid transfer amplitude eq. 33
    return (1/math.sqrt(2))**N*(1/math.sqrt(N))**n*_beta(n,N,theta)*math.sqrt(math.factorial(n))*np.sqrt(scipy.special.binom(N,n))

def _I(N,n):
    # Generates an even and normalized superposition of states with N qubits and N-n ones.
    if n>N:
        print('Error: n>N')
        return
    V = np.zeros(N,dtype='int')
    V[0:N-n] = 1
    out = 0
    if n!=0:
        L = list(multiset_permutations(V))
        for l in L:
            out = out + _buildKet(l)
    else:
        out = _buildKet(V)
    out = out/np.sqrt(np.sum(out))
    return out

def _rho(N,avg,phi,etaR,etaL,thetaR,thetaL):
    # Generates the state given by Eq. 43 but in normalized form. The norm is returned as P
    rho = 0
    for n in range(0,NT+1):
        for m in range(0,NT+1):
            for l in range(0,np.min((n,m))+1):
                for r in range(0,np.min((n,m))+1):
                    if n-l>N or n-r>N or m-l>N or m-r>N:
                        continue
                    A = _c(n,avg,phi)*np.conjugate(_c(m,avg,phi))
                    A = A*_err(n,l,etaL)*np.conjugate(_err(m,l,etaL))*_err(n,r,etaR)*np.conjugate(_err(m,r,etaR))
                    A = A*_Delta(n-l,N,thetaL)*_Delta(n-r,N,thetaR)*np.conjugate(_Delta(m-l,N,thetaL))*np.conjugate(_Delta(m-r,N,thetaR))
                    
                    ketL = _I(N,n-l)
                    braL = _I(N,m-l).reshape(1,-1)
                    stateL = np.kron(ketL,braL)
                    
                    ketR = _I(N,n-r)
                    braR = _I(N,m-r).reshape(1,-1)
                    stateR = np.kron(ketR,braR)
                    
                    rho = rho + A*np.kron(stateL,stateR)
                   
    P = np.trace(rho)
    rho = rho/P
    return rho,P

def _teleStateBell(N,i,j):
    # Creates higher dimensional Bell states based on the |I_(N-n)> state basis
    w = np.exp(1j*2*np.pi/(N+1))
    state = 0
    for n in range(0,N+1):
        R1 = _I(N,n)
        m = (n+i)%(N+1)
        L2 = _I(N,m)
        r12 = np.kron(R1,L2)
        state = state + (w**j)**n*r12
    state = state/np.sqrt(N+1)
    return state

def _swap(N,i,j,m1,m2):
    # Projects onto a particular Bell measurement, returns the normalized final state rho/P and the probability P
    M = np.kron(m1,m2)
    ket = _teleStateBell(N,i,j)
    ket = np.kron(ket,np.identity(2**N))
    ket = np.kron(np.identity(2**N),ket)
    bra = np.conjugate(np.transpose(ket))
    rho = bra.dot(M.dot(ket))
    P = np.real(np.trace(rho))
    return rho/P,P

def _vonNeumannEntropy(M):
    # Computes the von Neumann entropy of the matrix M
    # M: Matrix
    w, v = np.linalg.eig(M)
    rsum = 0
    for a in w:
        if a>0:
            rsum = rsum - a*np.log2(a)
    return np.real(rsum)

def _altBasis(val,basis):
    if basis == 0:
        if val == 0:
            return np.array([[1],[0]])#np.array([[1],[1]])/np.sqrt(2)
        if val == 1:
            return np.array([[0],[1]])#np.array([[1],[-1]])/np.sqrt(2)
    if basis == 1:
        if val == 0:
            return np.array([[1],[1]])/np.sqrt(2)#np.array([[1],[1]])/np.sqrt(2)
        if val == 1:
            return np.array([[1],[-1]])/np.sqrt(2)#np.array([[1],[-1]])/np.sqrt(2)
            

def _secretKey(M):
    # Computes the secret key of a density matrix M composed of 2 subsystems each with N qudits, with each qudit of dimension d
    
    idntty = np.identity(2)
    
    # Compute the mutual information
    PAB = np.zeros((2,2))
    for ii in range(0,2):
        for jj in range(0,2):
            ProjA = np.kron(_altBasis(ii,0),_altBasis(ii,0).reshape(1,-1))#np.kron(_buildKet(x),_buildKet(x).reshape(1,-1))
            ProjB = np.kron(_altBasis(jj,0),_altBasis(jj,0).reshape(1,-1))
            Proj = np.kron(ProjA,ProjB)
            PAB[ii,jj] = np.real(np.trace(Proj.dot(M)))
            
    PA = np.sum(PAB,axis=1)
    PB = np.sum(PAB,axis=0)
    
    MI = 0
    for ii in range(0,2):
        for jj in range(0,2):
            if PAB[ii,jj] != 0:
                MI = MI + PAB[ii,jj]*np.log2(PAB[ii,jj]/PA[ii]/PB[jj])
    
    # Compute the Holevo bound
    HO =  _vonNeumannEntropy(M)   
    for ii in range(0,2):
        ProjA = np.kron(_altBasis(ii,0),_altBasis(ii,0).reshape(1,-1))
        ProjA = np.kron(ProjA,idntty)
        prob = np.trace(ProjA.dot(M))
        rhoB = ProjA.dot(M.dot(np.transpose(ProjA)))
        rhoB = rhoB/np.trace(rhoB)
        HO = HO - prob*_vonNeumannEntropy(rhoB)
    
    
    # Secret key
    SK =  recEff*MI-HO
    return np.real(SK)

    

def _createPair(etaR,etaL,gamma,Pd,CT,avg,thetaR,thetaL):

    state1 = QO.state([], [], focks)
    
    ## Squeezing
    xi = math.acosh(math.sqrt(1+avg))
    QO.TMSV(state1,['1a','2a'],xi,focks)
    
    ## Channel loss
    ## Memory modes
    K = QO.lossyChannel(state1, '2a', 1-etaR, krauss)
    state1.channel(K)
    K = QO.lossyChannel(state1, '1a', 1-etaL, krauss)
    state1.channel(K)
    
    #### LEFT SIDE
    ## Memory prep
    state1.addMode('1c',focks)
    state1.addMode('1n',2)
    
    L1 = np.sin(thetaL)*QO.O('aR', state1, '1c').dot(QO.O('aR', state1, '1n'))
    L2 = np.cos(thetaL)*QO.O('I', state1, '1c').dot(QO.O('I', state1, '1n'))
    L = (L1 + L2)
    state1.operator(L)
    
    ## Coupling noise for NV
    K = QO.lossyChannel(state1, '1c', 1-CT, krauss)
    state1.channel(K)
    
    ## BS between carrier (1a) and NV fiber
    L = QO.BS2(state1, ['1a','1c'],t=np.pi/2)
    state1.operator(L)
    
    # Detector loss
    K = QO.lossyChannel(state1, '1a', 1-gamma, krauss)
    state1.channel(K)
    K = QO.lossyChannel(state1, '1c', 1-gamma, krauss)
    state1.channel(K)
    
    ## Bell measurement with dark counts
    P1 = QO.SPD(state1,'1a',0)
    P2s = [QO.SPD(state1,'1c',1),QO.SPD(state1,'1c',0)]
    state1.operator(P1)
    state1.mixed(P2s,[(1-Pd),Pd])
    state1.partialTrace(['1a'])
    state1.partialTrace(['1c'])

    
    #### RIGHT SIDE
    ## Memory modes
    state1.addMode('2c',focks)
    state1.addMode('2n',2)
    
    ## Memory prep
    L1 = np.sin(thetaR)*QO.O('aR', state1, '2c').dot(QO.O('aR', state1, '2n'))
    L2 = np.cos(thetaR)*QO.O('I', state1, '2c').dot(QO.O('I', state1, '2n'))
    L = L1 + L2
    state1.operator(L)
    
    ## Coupling noise for NV
    K = QO.lossyChannel(state1, '2c', 1-CT, krauss)
    state1.channel(K)
    
    ## BS between carrier (1a) and NV fiber
    L = QO.BS2(state1, ['2a','2c'],t=np.pi/2)
    state1.operator(L)
    
    # Detector loss
    K = QO.lossyChannel(state1, '2a', 1-gamma, krauss)
    state1.channel(K)
    K = QO.lossyChannel(state1, '2c', 1-gamma, krauss)
    state1.channel(K)
    
    ## Bell measurement
    P1 = QO.SPD(state1,'2a',0)
    P2s = [QO.SPD(state1,'2c',1),QO.SPD(state1,'2c',0)]
    state1.operator(P1)
    state1.mixed(P2s,[(1-Pd),Pd])
    state1.partialTrace(['2a'])
    state1.partialTrace(['2c'])
    
    ## Normalize
    
    state1.normalize()
    P = state1.Prob
    rho = state1.matrix
    
    return rho.toarray(),P

def _PDF(N,p,k):
    # Probability density function for  N repeaters succeeding in exactly k attempts, with each repeater having a success probability of p
    return (1-(1-p)**k)**N-(1-(1-p)**(k-1))**N

def _avgK(regs,p,maxAtt):
    # Computes the average number of trials before N repeaters succeed
    rsum = 0
    for k in range(1,maxAtt+1):
        rsum = rsum + _PDF(regs,p,k)*k
    return rsum

def _normalizeKey(darkAtt,p,regs):
    # Normalization factor to obtain the secret key rate giving a secret key
    rsum = 0
    for k in range(1,maxAtt+1):
        rsum = rsum + _PDF(regs,p,k)/k
    return rsum

def _thetaR(eta,eps):
    # Computes the optimal value of thetaR and avg from Eq. 63
    roots = np.roots([-eps+(1+eps)*eta+eta**2,eps*(2*eta-2),eta*(eps-2),2*eps,eps])
    roots = roots[roots>0] # Only certain roots make sense under the assumptions leading to eq. 63
    roots = roots[roots<1]
    root = np.min(roots)
    out = np.arctan(np.sqrt(eta*root**2/(1-root**2)))
    return out,root

def _thetaL(thetaR,avg,eta):
    # Computes the optimal value of thetaL from Eq. 56
    c0 = _c(0,avg,phi)
    c1 = _c(1,avg,phi)
    y = np.abs(c1)**2*eta/(np.abs(c0)**2*np.tan(thetaR)**2)
    return np.arctan(np.sqrt(y))

def _root(p,darkAtt,s,maxAtt):
    # Function used for finding the probability eps at which the entanglement sharing succeeds on average in darkAtt attempts
    # The function returns the average attempts needed (_avgK) minus the target darkAtt
    return _avgK(s,p,maxAtt)-darkAtt

def _bellCorr(rho):
    expVal = lambda rho,m1,m2: np.trace(rho@np.kron(m1,m2))

    
    sz = np.array([
        [0,1],
        [1,0]
        ])
    sx = np.array([
        [1,0],
        [0,-1]
        ])
    ma0 = sx
    ma1 = (sx+sz)/np.sqrt(2)
    ma2 = (sx-sz)/np.sqrt(2)
    
    mb1 = sx
    mb2 = sz
    
    corrArray = np.array([
        [expVal(rho,ma1,mb1),expVal(rho,ma1,mb2)],
        [expVal(rho,ma2,mb1),expVal(rho,ma2,mb2)]
        ])
    
    return corrArray

def _CHSH(corrArray):
    CHSH1 = corrArray[0,0] + corrArray[0,1] + corrArray[1,0] - corrArray[1,1]
    CHSH2 = corrArray[0,1] + corrArray[0,0] + corrArray[1,1] - corrArray[1,0]
    CHSH3 = corrArray[1,0] + corrArray[1,1] + corrArray[0,0] - corrArray[0,1]
    CHSH4 = corrArray[1,1] + corrArray[1,0] + corrArray[0,1] - corrArray[0,0]
    return np.max(np.abs(np.array([CHSH1,CHSH2,CHSH3,CHSH4])))

def _wwIneq(corrArray):
    binList = [(0,0),(0,1),(1,0),(1,1)]
    rsum = 0
    for r in binList:
        ssum = 0
        for s in binList:
            prf = (-1)**(np.array(r)@np.array(s))
            ssum = ssum + prf*corrArray[s]
        rsum = rsum + np.abs(ssum)
    return rsum/(4)

def _getCurve(D,l):
    pairs = int(np.ceil(D/l)) # Number of repeaters (pairs of registers)
    etaR = 10**(-a*l/10) # Channel transmitivity for each repeater
    # Computes relevant repeater values as a function of distance
    Y = np.zeros(pairs) # Empty vector for storing key rates
    X = np.zeros(pairs) # Empty vector for storing distances
    F = np.zeros(pairs)
    thetaLs = np.zeros(pairs)
    thetaRs = np.zeros(pairs)
    wwVals = np.zeros(pairs)
    CHSHVals = np.zeros(pairs)
    RVals = np.zeros(pairs)
    QVals = np.zeros(pairs)
    avgs = np.zeros(pairs) # Empty vector for storing optimum <n> as a function of distance
    for jj in range(0,pairs): # jj is the number of swaps
        print('Number of swaps: ',jj)
    
        # Finds the minimum accepted probability eps if the (j+1) repeaters are to succeed on average in darkAtt attempts
        eps = brentq(_root,0.0001,1,args=(darkAtt,jj+1,maxAtt),xtol=tol) # Finds the probability by root finding
        # Computes thetaR, avg, and thetaL if the channel has transmisitivity eta and the minimum allowed probability of success is eps
        thetaR,avg = _thetaR(etaR,eps)
        thetaL = _thetaL(thetaR,avg,etaR)
        
        # Generates the state with the chosen parameter values for N, avg, phi, eta, thetaR, thetaL
        rho,P = _createPair(etaR,etaL,gamma,Pd,CT,avg,thetaR,thetaL) # Numerical solution, includes all losses
        #rho,P = _rho(N,avg,phi,etaR,etaL,thetaR,thetaL) # Analytic solution, when neglecting some losses
        
        # We sample swaps fairly by letting each swap fill up a chunk of the number line [0,1] corresponding to its probability
        # We then drop a number (rnd) on this number line. The chunk in which the number lands is used for swapping.
        rhoSwap = rho.copy()    # Creates a copy for doing swaps
        measReg = 0 # Registers whether the given sequence of swaps will result in a bit flip between Alice and Bob's measurements in the Z-basis (0,1 basis)
        for kk in range(0,jj):
            rnd = random.uniform(0, 1)
            pSwap_sum = 0
            for xy in [[0,0],[0,1],[1,0],[1,1]]:
                rhoSwap_new, pSwap_new = _swap(N, xy[0], xy[1], rhoSwap, rho) # Creates a swap with the given Bell measurement
                pSwap_sum = pSwap_sum + pSwap_new
                if rnd <= pSwap_sum:
                    break
            rhoSwap = rhoSwap_new.copy()
            measReg = measReg + (1-xy[1]) # keeps track of enacted swaps, and how the density matrix has changed
        
        # Compute relevant characteristics
        corrArray = _bellCorr(rhoSwap)
        wwVals[jj] = _wwIneq(corrArray)
        CHSHVals[jj] = _CHSH(corrArray)
        QVals[jj] = _Q(rhoSwap,measReg%2)
        f = _normalizeKey(darkAtt,4*np.real(P),jj+1)
        RVals[jj] = (1 - _binEnt(QVals[jj]) - _binEnt((1+np.sqrt((CHSHVals[jj]/2)**2-1))/2))*f
        
        print(_avgK(jj+1,4*P,maxAtt))
        F[jj] = f
        Y[jj] = np.real(_secretKey(rhoSwap)*f) # Computes the secret key rate, normalized by darkAtt
        X[jj] = (jj+1)*l # Vector of repeater locations (distance along x)
        avgs[jj] = avg
        thetaLs[jj] = thetaL
        thetaRs[jj] = thetaR
    return X,Y,wwVals,F,avgs,thetaLs,thetaRs,rhoSwap,QVals,CHSHVals,RVals

def _binEnt(p):
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def _Q(M,flip):
    #Calculates the QBER (Quantum Bit Error Rate)
    # If flip=1 then we assume that a bit flip has occured due to swapping
    bss = 1
    # Computes the behavior
    PAB = np.zeros((2,2))
    for ii in range(0,2):
        for jj in range(0,2):
            ProjA = np.kron(_altBasis(ii,bss),_altBasis(ii,bss).reshape(1,-1))
            ProjB = np.kron(_altBasis(jj,bss),_altBasis(jj,bss).reshape(1,-1))
            Proj = np.kron(ProjA,ProjB)
            PAB[ii,jj] = np.real(np.trace(Proj.dot(M)))
    print(PAB)
            
    if flip:
        Q = PAB[0,1] + PAB[1,0]
    else:
        Q = PAB[0,0] + PAB[1,1]
    
    return Q
    

#%%
# Ss = np.linspace(1.5,2*np.sqrt(2),100)
# Ys = np.zeros(len(Ss))

# for kk in range(0,len(Ss)):
#     Ys[kk] = _DIQKD(Ss[kk])
    
# plt.figure()
# plt.plot(Ss,Ys)


# sys.exit()

#%% Computational paramters
phi = 0 # Phase of the TMSV state
a = 0.2 # Absorption of fiber 0.2 db/km
maxAtt = 50000 # Number of terms included in the sum in _avgK and _normalizeKey
tol = 10**(-6) # Tolerance on optimizations
N = 1 # Number of qudits per register
focks = 10 # Number of Fock states used in the numerical simulation
darkAtt = 150# Average number of entanglement sharing attempts allowed by the experimenter
nAvgs = 3 # Curve averages
NT = 10 # Number of terms in the sum described by eq 12
krauss = 10 # Number of krauss operators used in the lossy channel 
recEff = 1 # Reconcilliation efficiency


#%% Robustness check - various error parameters described in text
etaL = 1 # Transmission of the channel ch_L
CT = 1 # Transmission when coupling the emission of the bright state qubit into the fiber
Pd = 0 # Dark counts at the detectors 
gamma = 1 # Transmission of the detectors

#%% Compute the secret key rate at various distances (D) and repeater seperations (l) @ optimal thetaR
l = 10 # Distance between repeaters
D = 300 # Total distance
etaR = 10**(-a*l/10) # Channel transmitivity for each repeater

YList = []
XList = []
wwList = []
QList = []
CHSHList = []
RList = []

varList = [50,100,150,200]#[0,0.5*10**(-4),1*10**(-4),1.5*10**(-4)]
for nn in range(0,len(varList)):
    print(nn)
    darkAtt = varList[nn]
    Ysum = 0
    wwsum = 0
    CHSHsum = 0
    Qsum = 0
    Rsum = 0
    for mm in range(0,nAvgs):
        X,Y,wwVals,F,avgs,thetaLs,thetaRs,rhoSwap,QVals,CHSHVals,RVals = _getCurve(D,l)
        Ysum = Ysum + Y
        wwsum = wwsum + wwVals
        CHSHsum = CHSHsum + CHSHVals
        Qsum = Qsum + QVals
        Rsum = Rsum + RVals
    Yavg = Ysum/nAvgs
    wwAvg = wwsum/nAvgs
    QAvg = Qsum/nAvgs
    CHSHAvg = CHSHsum/nAvgs
    RAvg = Rsum/nAvgs
    
    XList.append(X)
    YList.append(Yavg)
    wwList.append(wwAvg)
    QList.append(QAvg)
    CHSHList.append(CHSHAvg)
    RList.append(RAvg)
    

#%% Plotting for visualization
plt.figure()

plt.plot(XList[0],YList[0],linewidth=2)
plt.plot(XList[1],YList[1],linewidth=2)
plt.plot(XList[2],YList[2],linewidth=2)
plt.plot(XList[3],YList[3],linewidth=2)
plt.plot(XList[0],-np.log2(1-10**(-a*XList[0]/10)),'k--',linewidth=2)
plt.legend(['50','100','150','200','PLOB'])

plt.ylim([1e-5,1e-1])
plt.xlim([60,500])
plt.yscale('log')
plt.xlabel('Distance [km]')
plt.ylabel('Secret Key Rate [bits/attempt]')
plt.grid(1)
plt.savefig('secretKey',dpi=400)
#%%
plt.figure()
plt.plot(XList[0],wwList[0],linewidth=2)
plt.plot(XList[1],wwList[1],linewidth=2)
plt.plot(XList[2],wwList[2],linewidth=2)
plt.plot(XList[3],wwList[3],linewidth=2)
plt.legend(['50','100','150','200'])
plt.ylim([0.2,1.5])
plt.xlim([0,500])
plt.xlabel('Distance [km]')
plt.ylabel(r'W$^3$ZB')
plt.grid(1)
plt.savefig('ww',dpi=400)

#%%
plt.figure()
plt.plot(XList[0],CHSHList[0],linewidth=2)
plt.plot(XList[1],CHSHList[1],linewidth=2)
plt.plot(XList[2],CHSHList[2],linewidth=2)
plt.plot(XList[3],CHSHList[3],linewidth=2)
plt.legend(['50','100','150','200'])
plt.ylim([1,3])
plt.xlim([0,500])
plt.xlabel('Distance [km]')
plt.ylabel(r'CHSH')
plt.grid(1)
plt.savefig('CHSH',dpi=400)

plt.figure()
plt.plot(XList[0],QList[0],linewidth=2)
plt.plot(XList[1],QList[1],linewidth=2)
plt.plot(XList[2],QList[2],linewidth=2)
plt.plot(XList[3],QList[3],linewidth=2)
plt.legend(['50','100','150','200'])
plt.ylim([0,0.15])
plt.xlim([0,500])
plt.xlabel('Distance [km]')
plt.ylabel(r'Quantum Bit Error Rate')
plt.grid(1)
plt.savefig('QBER',dpi=400)

plt.figure()
plt.plot(XList[0],RList[0],linewidth=2)
plt.plot(XList[1],RList[1],linewidth=2)
plt.plot(XList[2],RList[2],linewidth=2)
plt.plot(XList[3],RList[3],linewidth=2)
plt.plot(XList[0],-np.log2(1-10**(-a*XList[0]/10)),'k--',linewidth=2)
plt.legend(['50','100','150','200','PLOB'])
plt.ylim([1e-5,1e-1])
plt.xlim([60,350])
plt.yscale('log')
plt.xlabel('Distance [km]')
plt.ylabel('Device-Independent Secret Key Rate [bits/attempt]')
plt.grid(1)
plt.savefig('DIQKD',dpi=400)

