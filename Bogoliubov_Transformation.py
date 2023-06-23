# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:38:20 2023

@author: Jan-Philipp
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib 
matplotlib.style.use('JaPh')

"""
plt.rcParams.update({
    "text.usetex": True
})

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
"""

N = 200



PATH = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas, full, V_diag is zero\\"#N=40,lambda_IR=0.1+phi,lmabda_UV=10+phi\\"
PATH_BOG = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit Quadratic Hamiltonians\\N=200,lambda_IR=0.1,lambda_UV=10.0\\"

etas = []
epss = []
epss_err = []
Omega = np.block([[np.diag(np.ones(N)),np.zeros((N,N))],[np.zeros((N,N)),-np.diag(np.ones(N))]])
theta = np.block([[np.zeros((N,N)),np.diag(np.ones(N))],[np.diag(np.ones(N)),np.zeros((N,N))]])

def unpack_arr(flat,N):
    N = int(N)
    om0 = flat[0:N]
    V0 = flat[N:N+int(N**2)]
    W0 = flat[N+int(N**2):N+int(2*N**2)]
    eps = flat[-1]
    V = V0.reshape((N,N))
    W = W0.reshape((N,N))
    return om0,V,W,eps

def get_eigvals(A,B):
    H = np.block([[A,B],[-np.conj(B),-np.conj(A)]])
    eigvals, eigvecs = np.linalg.eig(H)
    eigvecs = eigvecs.T
    if np.isclose(np.linalg.det(eigvecs),0):
        ValueError("Eigenvectors are not linearly independent")
    else:
        print("Determinant of all eigenvectors is ", np.linalg.det(eigvecs))
    sorted_vals = np.array([val for _, val in sorted(zip(eigvals,eigvals),key = lambda tup: np.real(tup[0]))])#np.sort(np.real(eigvals))
    sorted_vecs = np.array([vec for _, vec in sorted(zip(eigvals,eigvecs),key = lambda tup: np.real(tup[0]))])
        
    eigvals_ord = np.array([[sorted_vals[i],sorted_vals[len(eigvals)-i-1]] for i in range(len(eigvals)//2)])
    eigvecs_ord = np.array([[sorted_vecs[i],sorted_vecs[len(eigvals)-i-1]] for i in range(len(eigvals)//2)])
    
    return eigvals_ord, eigvecs_ord

def find_right_eigvals(eigvals_ord,eigvecs_ord):
    imag_eigvals = np.zeros(len(eigvals_ord))
    right_eigvals = []
    conj_eigvals = []
    for i, vecs in enumerate(eigvecs_ord):
        vec0 = vecs[0]
        vec1 = vecs[1] 
        #print(eigvals_ord[i])
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
            conj_eigvals.append(0)
            if np.isclose(np.real(eigvals_ord[i][0]),0):
                imag_eigvals[i] = np.abs(np.imag(eigvals_ord[i][1]))
            else:
                ValueError("Complex Eigenvalue which is not purely imaginary!")
            
    return imag_eigvals, right_eigvals, conj_eigvals

def ground_state_energy_change(right_eigvals,A):
    return -1/2*np.sum(np.diag(A))+1/2*np.sum(right_eigvals) #see eq. 33 in Practial Course manual

def get_plot_data_bogoliubov(V0, W0, om0):
    V = V0 + np.diag(om0) # - np.diag(np.diag(V0)) #
    W = 1/2 * (W0 + W0.T)
    #print("eta=",eta,"eps=",eps)
    A = V
    B = 2*W
    eigvals_ord, eigvecs_ord = get_eigvals(A,B)
    imag_eigvals, right_eigvals, conj_eigvals = find_right_eigvals(eigvals_ord,eigvecs_ord)
    
    gs_change = ground_state_energy_change(right_eigvals,A)
    imag = max(imag_eigvals)
    first_positve_eigval = min(np.array(right_eigvals)[np.array(right_eigvals)>0])
    smallest_real_eigval = min(np.array(right_eigvals)[np.array(right_eigvals)!=0])
    
    return gs_change, imag, first_positve_eigval, smallest_real_eigval




imag_eigvals_list_bog = []
imag_eigvals_list_flow = []
imag_eigvals_list_flow_and_bog = []

first_positve_eigvals_bog = []
first_positve_eigvals_flow = []
first_positve_eigvals_flow_and_bog = []

smallest_real_eigvals_bog = []
smallest_real_eigvals_flow = []
smallest_real_eigvals_flow_and_bog = []

change_gs_energy_bog = []
change_gs_energy_flow = []
change_gs_energy_flow_and_bog = []

gs_energies_bog = []
gs_energies_flow = []
gs_energies_flow_and_bog = []

for FILENAME in [file for file in os.listdir(PATH) if not "_t" in file]:
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    t_filename = "sol_full_t"+FILENAME[13:]
    path_inp = PATH+FILENAME
    path_t = PATH+t_filename

    inp = np.load(path_inp)
    om0_start,V0_start,W0_start,eps_start = unpack_arr(inp.T[0],N)
    om0_end,V0_end,W0_end,eps_end = unpack_arr(inp.T[-1],N)

    gs_change_bog, imag_bog, first_positve_eigval_bog, smallest_real_eigval_bog = get_plot_data_bogoliubov(V0_start, W0_start, om0_start)
    gs_change_flow_and_bog, imag_flow_and_bog, first_positve_eigval_flow_and_bog, smallest_real_eigval_flow_and_bog = get_plot_data_bogoliubov(V0_end, W0_end, om0_end)
    
    gs_change_flow = eps_end - eps_start
    first_positve_eigval_flow = min(om0_end[om0_end>0])
    smallest_real_eigval_flow = min(om0_end[om0_end!=0])
    imag_flow = 0
    
    gs_energy_bog = eps_start + gs_change_bog
    gs_energy_flow = eps_end
    gs_energy_flow_and_bog = gs_change_flow_and_bog + eps_end
    gs_change_flow_and_bog = gs_energy_flow_and_bog - eps_start
    
    etas.append(eta)
    
    #imag_eigvals_list_bog.append(imag_bog)
    imag_eigvals_list_flow.append(imag_flow)
    imag_eigvals_list_flow_and_bog.append(imag_flow_and_bog)

    #first_positve_eigvals_bog.append(first_positve_eigval_bog)
    first_positve_eigvals_flow.append(first_positve_eigval_flow)
    first_positve_eigvals_flow_and_bog.append(first_positve_eigval_flow_and_bog)

    #smallest_real_eigvals_bog.append(smallest_real_eigval_bog)
    smallest_real_eigvals_flow.append(smallest_real_eigval_flow)
    smallest_real_eigvals_flow_and_bog.append(smallest_real_eigval_flow_and_bog)


    #change_gs_energy_bog.append(gs_change_bog)
    change_gs_energy_flow.append(gs_change_flow)
    change_gs_energy_flow_and_bog.append(gs_change_flow_and_bog)

    #gs_energies_bog.append(gs_energy_bog)
    gs_energies_flow.append(gs_energy_flow)
    gs_energies_flow_and_bog.append(gs_energy_flow_and_bog)

etas_bog = []

for FILENAME in [file for file in os.listdir(PATH_BOG) if not "_t_" in file]:
    eta_bog = float(FILENAME.split(',')[0].split('=')[-1])

    path_inp = PATH_BOG+FILENAME
    inp = np.load(path_inp)
    om0_start,V0_start,W0_start,eps_start = unpack_arr(inp,N)
    gs_change_bog, imag_bog, first_positve_eigval_bog, smallest_real_eigval_bog = get_plot_data_bogoliubov(V0_start, W0_start, om0_start)
       
    gs_energy_bog = eps_start + gs_change_bog
    
    etas_bog.append(eta_bog)
    
    imag_eigvals_list_bog.append(imag_bog)
    
    first_positve_eigvals_bog.append(first_positve_eigval_bog)
    
    smallest_real_eigvals_bog.append(smallest_real_eigval_bog)
    
    change_gs_energy_bog.append(gs_change_bog)

    gs_energies_bog.append(gs_energy_bog)

eta_filter_bog = np.abs(np.array(etas_bog)) < 15
eta_sort_filter_bog = np.argsort(etas_bog)
energy_limit = 1/2 * ((np.array(gs_energies_bog)[eta_sort_filter_bog])[-1] + (np.array(gs_energies_bog)[eta_sort_filter_bog])[0])

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,10/np.sqrt(2))) 

ax1.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)
ax2.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)

ax1.tick_params(axis='both',labelsize=9)
ax2.tick_params(axis='both',labelsize=9)
ax1.set_xlabel(r'$\eta$',fontsize = 14)
ax2.set_xlabel(r'$\eta$',fontsize = 14)

ax1.set_ylabel(r'GS Energy $E_0[c/\xi]$',fontsize = 14)
ax2.set_ylabel(r'Change of GS Energy before vs after diagonalization $\Delta E_0[c/\xi]$',fontsize = 14)

ax1.plot(np.array(etas_bog)[eta_filter_bog],np.array(gs_energies_bog)[eta_filter_bog],marker='x',color='steelblue',linestyle='None',label='direct Bogoliubov',markersize=2)
ax1.plot(etas,gs_energies_flow,marker='x',color='lime',linestyle='None',label='after partially traversed flow',markersize=4)
ax1.plot(etas,gs_energies_flow_and_bog,marker='+',color='firebrick',linestyle='None',label='flow and Bogoliubov',markersize=4)
ax1.axhline(energy_limit,linestyle='dotted',marker='None',label=r'GS in limit $\eta\rightarrow\pm\infty$')

ax2.plot(np.array(etas_bog)[eta_filter_bog],np.array(change_gs_energy_bog)[eta_filter_bog],marker='x',color='steelblue',linestyle='None',label='direct Bogoliubov',markersize=2)
ax2.plot(etas,change_gs_energy_flow,marker='x',color='lime',linestyle='None',label='after partially traversed flow',markersize=4)
ax2.plot(etas,change_gs_energy_flow_and_bog,marker='+',color='firebrick',linestyle='None',label='flow and Bogoliubov',markersize=4)

ax1.legend(loc='best',fontsize=10)
ax2.legend(loc='best',fontsize=10)
plt.tight_layout()
plt.savefig('GS_energies_bog_flow_comp.pdf',dpi=300)
plt.show()

plt.clf()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,10/np.sqrt(2))) 

ax1.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)
ax2.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)
ax1.tick_params(axis='both',labelsize=9)
ax2.tick_params(axis='both',labelsize=9)
ax1.set_xlabel(r'$\eta$',fontsize = 12)
ax2.set_xlabel(r'$\eta$',fontsize = 12)
ax1.set_ylabel(r'Smallest (SR) and smallest positive (SP) eigenvalue $[c/\xi]$',fontsize = 14)
ax2.set_ylabel(r'Absolute value of imaginary eigenvalue $[c/\xi]$',fontsize = 14)
ax2.set_ylim(-0.05,0.8)

ax1.plot(np.array(etas_bog)[eta_filter_bog],np.array(smallest_real_eigvals_bog)[eta_filter_bog],marker='<',color='firebrick',linestyle='None',label='SR via Bogoliubov',markersize=2)
ax1.plot(np.array(etas_bog)[eta_filter_bog],np.array(first_positve_eigvals_bog)[eta_filter_bog],marker='>',color='firebrick',linestyle='None',label='SP via Bogoliubov',markersize=2)

ax1.plot(etas,smallest_real_eigvals_flow,marker='v',color='steelblue',linestyle='None',label='SR via flow',markersize=3)
ax1.plot(etas,first_positve_eigvals_flow,marker='^',color='steelblue',linestyle='None',label='SP via flow',markersize=3)

ax1.plot(etas,smallest_real_eigvals_flow_and_bog,marker='x',color='orange',linestyle='None',label='SR via flow + Bogoliubov',markersize=3)
ax1.plot(etas,first_positve_eigvals_flow_and_bog,marker='+',color='orange',linestyle='None',label='SP via flow + Bogoliubov',markersize=3)


ax2.plot(np.array(etas_bog)[eta_filter_bog],np.array(imag_eigvals_list_bog)[eta_filter_bog],marker='x',color='steelblue',linestyle='None',label='direct Bogoliubov',markersize=2)
ax2.plot(etas,imag_eigvals_list_flow,marker='x',color='lime',linestyle='None',label='after partially traversed flow',markersize=4)
ax2.plot(etas,imag_eigvals_list_flow_and_bog,marker='x',color='firebrick',linestyle='None',label='flow and Bogoliubov',markersize=4)


ax1.legend(loc='best',fontsize=10)
ax2.legend(loc='best',fontsize=10)
plt.tight_layout()
plt.savefig('spectrum_analysis_bog_flow_comp.pdf',dpi=300)
plt.show()
