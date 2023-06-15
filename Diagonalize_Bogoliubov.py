# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:16:21 2023

@author: Jan-Philipp
"""

import os
import numpy as np
import  quadratic_solver_wrapper as qs
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy
import matplotlib 
matplotlib.style.use('JaPh')
plt.rcParams.update({
    "text.usetex": True
})

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

N = 200



PATH = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit Quadratic Hamiltonians\\"


etas = []
epss = []
epss_err = []
Omega = np.block([[np.diag(np.ones(N)),np.zeros((N,N))],[np.zeros((N,N)),-np.diag(np.ones(N))]])
theta = np.block([[np.zeros((N,N)),np.diag(np.ones(N))],[np.diag(np.ones(N)),np.zeros((N,N))]])

def get_eigvals(A,B):
    H = np.block([[A,B],[-np.conj(B),-np.conj(A)]])
    eigvals, eigvecs = np.linalg.eig(H)
    eigvecs = eigvecs.T
    sorted_vals = np.array([val for _, val in sorted(zip(eigvals,eigvals),key = lambda tup: np.real(tup[0]))])#np.sort(np.real(eigvals))
    sorted_vecs = np.array([vec for _, vec in sorted(zip(eigvals,eigvecs),key = lambda tup: np.real(tup[0]))])

    eigvals_ord = np.array([[sorted_vals[i],sorted_vals[len(eigvals)-i-1]] for i in range(len(eigvals)//2)])
    eigvecs_ord = np.array([[sorted_vecs[i],sorted_vecs[len(eigvals)-i-1]] for i in range(len(eigvals)//2)])
    
    return eigvals_ord, eigvecs_ord

def find_right_eigvals(eigvals_ord,eigvecs_ord):

    right_eigvals = []
    conj_eigvals = []
    right_eigvals_err = np.zeros(len(eigvals_ord))
    for i, vecs in enumerate(eigvecs_ord):
        vec0 = vecs[0]
        vec1 = vecs[1] 
        print(eigvals_ord[i])
        matrix_element_0 = np.conj(vec0.T)@Omega@vec0
        matrix_element_1 = np.conj(vec1.T)@Omega@vec1
        if (matrix_element_0>0) and (matrix_element_1<0):# and not np.abs(np.imag(eigvals_ord[i][0]))<1e-10:
            right_eigvals.append(eigvals_ord[i][0])
            conj_eigvals.append(-eigvals_ord[i][1])
        elif (matrix_element_1>0) and (matrix_element_0<0):# and not np.abs(np.imag(eigvals_ord[i][1]))<1e-10:           
            right_eigvals.append(eigvals_ord[i][1])
            conj_eigvals.append(-eigvals_ord[i][0])
        else:
            right_eigvals.append(0)
            print("eta=",eta,"--------",matrix_element_0,matrix_element_1,"\n")
            print(eigvals_ord[i],"\n")
            right_eigvals_err[i]=(max(np.abs(eigvals_ord[i])))
            conj_eigvals.append(0)
            
    return right_eigvals, conj_eigvals, right_eigvals_err

def ground_state_energy(right_eigvals,right_eigvals_err,E_0,A):
    err = 1/2*np.sum(right_eigvals_err)
    return E_0-1/2*np.sum(np.diag(A))+1/2*np.sum(right_eigvals), err #see eq. 33 in Practial Course manual

right_eigvals_list = []

for FILENAME in [file for file in os.listdir(PATH) if not "_t_" in file]:
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    if eta<=-3:
        path_inp = PATH+FILENAME
        om0,V0,W,eps = qs.unpack_arr(np.load(path_inp),200)
        V = V0 + np.diag(om0) # - np.diag(np.diag(V0)) #
        #print("eta=",eta,"eps=",eps)
        A = V
        B = 2*W
        
        eigvals_ord, eigvecs_ord = get_eigvals(A,B)
        right_eigvals, conj_eigvals, eigvals_err = find_right_eigvals(eigvals_ord,eigvecs_ord)
        right_eigvals_list.append(right_eigvals)
        etas.append(eta)
        gs, gs_err = ground_state_energy(conj_eigvals,eigvals_err,eps,A)
        epss.append(gs)
        epss_err.append(gs_err)

#plt.plot(np.linspace(-20,20,1000),-1.05*np.linspace(-20,20,1000),marker='None',linestyle="-")
plt.errorbar(etas,np.real(epss),yerr = np.real(epss_err),color='black',linestyle='None',marker='x',label=r'via Bogoliubov Transformation',capsize = 0, ecolor = 'red')
plt.xlabel(r'$\eta=g_{IB}/g_{BB}$',fontsize=14)
plt.ylabel(r'$E_0$',fontsize=14)
plt.xlim(min(etas)-1,max(etas)+1)
plt.ylim(min(epss)*1.1,max(epss)*1.1)
plt.legend()
plt.tick_params(axis='both', which='minor')
plt.grid(visible=True, which='both', color="grey", linestyle='-', alpha=.2,linewidth=.008)
plt.savefig('E_0-eta_via_Bogliubov_Trafo.pdf',dpi=300)
plt.show()
np.save('eta_E_0_bog.npy',np.array([etas,epss,epss_err]))
