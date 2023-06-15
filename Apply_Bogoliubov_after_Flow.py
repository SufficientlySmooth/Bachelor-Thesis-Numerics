# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 09:02:16 2023

@author: Jan-Philipp
"""

import numpy as np
import  quadratic_solver_wrapper as qs
import matplotlib.pyplot as plt
import os
import scipy

plt.rcParams.update({
    "text.usetex": True
})

N = 200

Omega = np.block([[np.diag(np.ones(N)),np.zeros((N,N))],[np.zeros((N,N)),-np.diag(np.ones(N))]])
theta = np.block([[np.zeros((N,N)),np.diag(np.ones(N))],[np.diag(np.ones(N)),np.zeros((N,N))]])

#print(eigvals)

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
    right_eigvals_err = np.zeros(len(eigvals_ord))
    for i, vecs in enumerate(eigvecs_ord):
        vec0 = vecs[0]
        vec1 = vecs[1] 
        matrix_element_0 = np.conj(vec0.T)@Omega@vec0
        matrix_element_1 = np.conj(vec1.T)@Omega@vec1
        if (matrix_element_0>0) and (matrix_element_1<0):
            right_eigvals.append(eigvals_ord[i][0])
        elif (matrix_element_1>0) and (matrix_element_0<0):           
            right_eigvals.append(eigvals_ord[i][1])
        else:
            right_eigvals.append(0)
            right_eigvals_err[i]=(max(np.abs(eigvals_ord[i])))
            
    return right_eigvals, right_eigvals_err

def ground_state_energy(right_eigvals,right_eigvals_err,E_0,A):
    err = 1/2*np.sum(right_eigvals_err)
    return E_0-1/2*np.sum(np.diag(A))+1/2*np.sum(right_eigvals), err #see eq. 33 in Practial Course manual

lambda_UV = 10
lambda_IR = .1
PATH = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas, full, V_diag is zero\\"


etas = []
epss = []
epss_err = []

for FILENAME in [file for file in os.listdir(PATH) if not "_t" in file][::-1]:
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    t_filename = "sol_full_t"+FILENAME[13:]
    path_inp = PATH+FILENAME
    path_t = PATH+t_filename
    
    inp = np.load(path_inp)
    
    om0,V0,W,eps = qs.unpack_arr(inp.T[-1],200)
    V = V0 + np.diag(om0) # - np.diag(np.diag(V0)) #
    print(min((om0)))
    A = V
    B = 2*W
    
    eigvals_ord, eigvecs_ord = get_eigvals(A,B)
    right_eigvals, right_eigvals_err = find_right_eigvals(eigvals_ord,eigvecs_ord)
    #print(right_eigvals)
    #print("Maximal right eigenvalue: ", max(np.abs(right_eigvals)))
    etas.append(eta)
    gs, gs_err = ground_state_energy(right_eigvals,right_eigvals_err,eps,A)
    epss.append(gs)
    epss_err.append(gs_err)
    
plt.plot(etas,epss,linestyle='None',marker='x')
plt.title('Test Unitarity')
np.save('eta_E_0_flow_and_bog.npy',np.array([etas,epss,epss_err]))
plt.show()