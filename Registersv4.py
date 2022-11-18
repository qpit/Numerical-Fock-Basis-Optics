# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:34:32 2021

@author: ajebje
"""
import numpy as np
import math
from sympy.utilities.iterables import multiset_permutations
import sys
import matplotlib.pyplot as plt
import scipy.special as ss
from scipy.sparse import csr_matrix
import scipy.sparse
import scipy.optimize
import scipy.linalg as sl

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


def _negativity(M,N,dim):
    PM = _partialTranspose(M,N,dim)
    w,v = sl.eigh(PM)

    mw = w[w<0]  
    return np.abs(np.sum(mw))

def _partialTranspose(M,N,dim):
    out = np.zeros((dim**(2*N),dim**(2*N)),dtype=np.complex128)
    for i in range(0,dim**N):
        for j in range(0,dim**N):
            out[i*dim**N:(i*dim**N+dim**N),j*dim**N:(j*dim**N+dim**N)] = np.transpose(M[i*dim**N:(i*dim**N+dim**N),j*dim**N:(j*dim**N+dim**N)])
    return out


def _error(thetaR,thetaL,N,avg,phi,etaR,etaL,dim):
    rho,P = _rho(N,avg,phi,etaR,etaL,thetaR,thetaL)
    return -_negativity(rho, N, dim)

def Fc(c):
    if 2*c>=1:
        return 0
    elif 2*c<1:
        return -np.log2(2*c)

def _negTMSV(avg,etaL,etaR):
    r = np.arccosh(np.sqrt(1+avg))
    #r = np.arcsinh(np.sqrt(avg))

    v = math.cosh(2*r) # Squeezing parameters
    a = math.sinh(2*r)
    acos = a*np.cos(phi)
    asin = a*np.sin(phi)
    cov0 = (1/2)*np.array([
        [  v,     0,    acos, -asin ],
        [  0,     v,   -asin, -acos ],
        [ acos, -asin,   v,     0   ],
        [-asin, -acos,   0,     v   ]
        ]) # TMSV

    # Gaussian Loss map acting on channel 1,2,3, and 4
    Id = np.eye(4)
    sInf =  (1/2)*Id # Asymptotic covariance matrix. In this case, vacuum.
    G = np.diag([etaL,etaL,etaR,etaR]) # Loss operator
    sqG = np.sqrt(G)
    cov = sqG.dot(cov0).dot(sqG)+(Id-G).dot(sInf) # State covariance

    for kk in range(0,cov.shape[0]):
        for jj in range(0,cov.shape[1]):
            if kk==1 or jj==1:
                cov[kk,jj] = -cov[kk,jj]

    om = np.array([[0,1],[-1,0]])
    omega = sl.block_diag(om,om)
    syCov = 1j*omega@cov

    w,v = np.linalg.eigh(syCov)

    wpos = w[w>0]

    logneg = 0
    for kk in range(0,len(wpos)):
        logneg = logneg + Fc(wpos[kk])
    
    neg = (2**(logneg)-1)/2
        
    return neg 
    
    

#%% Computational
phi = 0
dim = 2
tol = 10**(-6)
NT = 4 # Number of terms in the sum described by eq 11

#%% Range paramters
thetas = np.linspace(0.01,np.pi/2,70)
etaRs = np.linspace(10**(-3),10**(0),15)
ns = np.arange(1,5)
etaL = 1
etaR = 1
avg = 0.5

# %% Plot the negativity against the superposition angle (zero loss)


plt.figure()
optThetaNoLoss = np.zeros(len(ns))
for i in range(0,len(ns)):
    negs = np.zeros(len(thetas))
    N = ns[i]
    for j in range(0,len(thetas)):
        rho,P = _rho(N,avg,phi,etaR,etaL,thetas[j],thetas[j])
        negs[j] = _negativity(rho,N,dim)
    optThetaNoLoss[i] = thetas[np.argmax(negs)]
    plt.plot(thetas/(np.pi/4),negs)
plt.xlabel('Superposition angle $\\theta/\pi/4$')
plt.ylabel('Negativity')
plt.legend(['1','2','3','4'])
plt.grid(1)
plt.savefig('NegvsTheta.png',dpi=400)

plt.figure()
plt.scatter(ns,optThetaNoLoss/(np.pi/4))
plt.grid(1)
plt.xlabel('Number of qubits per register')
plt.ylabel(r'Optimal superposition angle [$\theta/\pi/4$]')
plt.savefig('OptThetaVsN.png',dpi=400)


# %% Optimize thetaR for a given N and eta. We fix thetaL from the optimal choice at zero loss.
optThetaRsLoss = np.zeros((len(etaRs),len(ns)))
for i in range(0,len(etaRs)):
    for k in range(0,len(ns)):
        thetaL = optThetaNoLoss[k]
        x0 = np.pi/4
        res = scipy.optimize.minimize(_error, x0, method='nelder-mead',args = (thetaL,ns[k],avg,phi,etaRs[i],etaL,dim) ,options={'xatol': tol, 'disp': True})
        optThetaRsLoss[i,k] = np.abs(res.x[0])

# %%
# Plot optimal thetaR against eta
plt.figure()
for i in range(0,len(ns)):
    plt.plot(etaRs,optThetaRsLoss[:,i]/(np.pi/4))

plt.xlabel('Transmission $\eta$')
plt.ylabel(r'Optimal [$\theta/\pi/4$]')
plt.legend(['1','2','3','4'])
plt.grid(1)
plt.savefig('ThetavsEta.png',dpi=400)


# %% Plot  Negativity @ Opt thetaR

plt.figure()
for i in range(0,len(ns)):
    N = ns[i]
    thetaL = optThetaNoLoss[i]
    negs = np.zeros(len(etaRs))
    for j in range(0,len(etaRs)):
        rho,P = _rho(N,avg,phi,etaRs[j],etaL,optThetaRsLoss[j,i],thetaL)
        negs[j] = _negativity(rho,N,dim)
    plt.plot(etaRs,negs)
    
negsTMSV = np.zeros(len(etaRs))
for j in range(0,len(etaRs)):
    negsTMSV[j] = _negTMSV(avg,etaL,etaRs[j])
plt.plot(etaRs,negsTMSV,'--')

plt.legend(['1','2','3','4','TMSV'])
plt.xlabel('Transmission $\eta$')
plt.ylabel('Negativity')
plt.yscale('log')
plt.grid(1)
plt.savefig('NegvsEta.png',dpi=400)


# %% Plot  Probability @ Opt thetaR

plt.figure()
for i in range(0,len(ns)):
    N = ns[i]
    thetaL = optThetaNoLoss[i]
    Ps = np.zeros(len(etaRs))
    for j in range(0,len(etaRs)):
        rho,P = _rho(N,avg,phi,etaRs[j],etaL,optThetaRsLoss[j,i],thetaL)
        Ps[j] = np.real(P)*4**N
    plt.plot(etaRs,Ps)
plt.legend(['1','2','3','4'])
plt.yscale('log')
plt.xlabel('Transmission $\eta$')
plt.ylabel('Probability of Success')
plt.grid(1,which="both")
plt.savefig('ProbvsEta.png',dpi=400)


