#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:03:36 2023

@author: santiago
"""

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import aux


plt.close('all')
plt.rcParams['text.usetex'] = True

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'font.size': 15
})


plt.close('all')


#%% Parameters

n_qubits = 4
state = 'GHZ'
# states = ['GHZ', 'W']

n = 1 # number of states to simulate for each combination of eta, etc.
n_rand_states = 1 # number of random states per value of eta. TODO: adapt for this.
n_eta = 3
n_alpha = 2 
n_beta = 2

colours = {'H': 'YlOrRd',
           'L': 'PuBuGn',
           'rho_re': 'inferno',
           'rho_im': 'cividis',
           'C': 'viridis'}

colourbar_limits = {'H': [-1,1],
                    'L': [-2, 2],
                    'rho_re': [0, 0.5],
                    'rho_im': [0, 0.5],
                    'C': [0, 1]}
#%% Flags

PRINT_STATES = 0
PRINT_F = 0
PRINT_H_and_L = 0
PRINT_rho = 0
PRINT_C = 0

PLOT_H_and_L = 0
PLOT_rho = 0
PLOT_C = 1

#%%
def density_matrix(state, n_qubits, n_ones=0):
    if state=='GHZ':
        coefficients = np.zeros(2**n_qubits)
        coefficients[0] = coefficients[-1] = 1
        coefficients = coefficients/np.linalg.norm(coefficients)

    elif state=='W':
        coefficients = np.zeros(2**n_qubits)
        for n in range(n_qubits):
            coefficients[2**n]=1
        coefficients = coefficients/np.linalg.norm(coefficients)
    
    elif state=='noise':
        coefficients = 2*np.random.rand(2**n_qubits)-1 + 2*np.random.rand(2**n_qubits)*1j-1j
        coefficients[0] = coefficients[-1] = 0
        coefficients = coefficients/np.linalg.norm(coefficients)
        
    '''
    elif state=='bitflip':
        # n_ones = 1
        n_zeros = n_qubits - n_ones
        
        bits = set([''.join(x) for x in itertools.permutations(n_ones * '1' + n_zeros * '0', n_qubits)])
        
        coefficients = np.zeros(2**n_qubits)
        options = []
        for b in bits:
            options.append(int(b, base=2))
        i = np.random.choice(options)
        coefficients[i] = 1
    '''
    
    if PRINT_STATES:
        print_state(state, coefficients) # This function does not print imaginary part. Fix.
    
    rho = np.tensordot(np.conj(coefficients), coefficients, axes=0)
    return rho





def plot_matrix(M, title:str='', save:bool=0, save_name:str='M', path:str='M', colour:str='binary', clim:list=[None, None]):
    
    fig, ax = plt.subplots()
    import matplotlib
    im = ax.imshow(M, cmap=colour, norm=matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1]))
    # ax.set_title('Pan on the colorbar to shift the color mapping\n'             'Zoom on the colorbar to scale the color mapping')
    colourbar = fig.colorbar(im, ax=ax, label='')
    # cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cm, norm=mpl.colors.Normalize(vmin=-0.5, vmax=1.5))
    colourbar.ax.set_ylim(clim)
    plt.show()
    
    # plt.imshow(rho,  cmap='Purples')
    # plt.colorbar()
    plt.title(r'%s'%title)
    plt.show()
    if save:
        # aux.check_dir(str(n_qubits)+'_qubits')
        aux.check_dir(path)
        plt.savefig(path + os.sep + save_name + '.pdf')




def print_state(state, coefficients):
    
    basis = list(itertools.product([0, 1], repeat=n_qubits))
    psi = ''
    for c,b in zip(coefficients, basis):
        if c:
            psi = psi + ' + ' + '%.8f'%c + '\t| %s >'%str(b).strip('()') + '\n'
    
    print(state + ':')
    print(psi)



#%%

def H_and_L(n_qubits, print_matrices:bool=PRINT_H_and_L):
    
    sigma_z = np.diag([1, -1])
    id_2 = np.diag([1, 1])
    
    # Operators:
    H = {}
    for i in range(n_qubits):
        H[i] = np.ones((1,1))
        for j in range(n_qubits):
            if i==j:
                H[i] = np.kron(H[i], sigma_z)
                # op[i].append(sigma_z)
            else:
                H[i] = np.kron(H[i], id_2)
    
    
    
    L = {}
    
    for i in range(n_qubits-1):
        L[i] = {}
    
    for i in range(n_qubits):
        for j in range(i, n_qubits):
            if j!=i:
                L[i][j] = H[i] - H[j]

            # Printing:
            if print_matrices:
                print('L[%d][%d] =\n'%(i,j), L[i][j], '\n')
    
    return H, L
    

#%%

data = pd.DataFrame()

# w = np.ones(n_qubits) # 2*np.random.rand(d) - np.ones(d)

# w = w/np.linalg.norm(w) # normalisation
# W = np.outer(w, w)

eta_values = np.linspace(0, 1, num=n_eta)
alpha_values = np.linspace(0, 1, num=n_alpha)
beta_values = np.linspace(0, 1, num=n_beta)

# GHZ_noise is noise everywhere except |00...0> and |11...1>
# single_excitations_noise is one other element of the base (e.g. |1000>)

#%%

H, L = H_and_L(n_qubits, print_matrices=False)

if PLOT_H_and_L:
    for i in range(len(H)):
        plot_matrix(H[i], save=1, save_name='H_%d'%(i), path = str(n_qubits) + '_qubits' + os.sep + 'H', colour=colours['H'], clim=colourbar_limits['H'])

    for i in L.keys():
        for j in L[i].keys():
            plot_matrix(L[i][j], save=1, save_name='L_%d_%d'%(i,j), path=str(n_qubits) + '_qubits' + os.sep + 'L', colour=colours['L'], clim=colourbar_limits['L'])

for eta in eta_values:
    for i in range(n):
        l = len(data)
        rho_state = density_matrix(state, n_qubits)
        rho_noise = density_matrix('noise', n_qubits)
        rho = eta * rho_state + (1-eta) * rho_noise
        data.at[l+1, 'eta'] = eta
        # data.at[l+1, 'alpha'] = alpha
        # data.at[l+1, 'beta'] = beta
        if PLOT_rho:
            plot_matrix(np.real(rho), title='$\eta=%.2f$ - $Re$'%eta, save=1, save_name='rho_%.2f_re'%(eta), path = str(n_qubits) + '_qubits' + os.sep + 'rho', colour=colours['rho_re'], clim=colourbar_limits['rho_re'])
            plot_matrix(np.imag(rho), title='$\eta=%.2f$ - $Im$'%eta, save=1, save_name='rho_%.2f_im'%(eta), path = str(n_qubits) + '_qubits' + os.sep + 'rho', colour=colours['rho_im'], clim=colourbar_limits['rho_im'])
        

        C = {}
        for i in range(n_qubits):
            C[i] = {}
        
        for i in L.keys():
            for j in L[i].keys():
                C[i][j] = np.matmul(L[i][j], rho) - np.matmul(rho, L[i][j])
        
        if PLOT_C:
            for i in C.keys():
                for j in C[i].keys():
                    plot_matrix(np.real(C[i][j]), title='$\eta=%.2f$ - $Re$'%eta, save=1, save_name='C_%d_%d_%.2f_re'%(i,j, eta), path=str(n_qubits) + '_qubits' + os.sep + 'C', colour=colours['C'], clim=colourbar_limits['C'])
                    plot_matrix(np.imag(C[i][j]), title='$\eta=%.2f$ - $Im$'%eta, save=1, save_name='C_%d_%d_%.2f_im'%(i,j, eta), path=str(n_qubits) + '_qubits' + os.sep + 'C', colour=colours['C'], clim=colourbar_limits['C'])
                
                # print(50*'-')
                # print('eta:', eta)
                # print('L[%d][%d]'%(i, j))
                # print(K)



for i in C.keys():
    for j in C[i].keys():
        np.matrix.max(np.real(C[i][j]))


#%%
'''
plt.figure()

for c in data.columns[1:]:
    plt.scatter(data['eta'], data[c], alpha=0.5, label=c)
    
plt.title(r'$\rho = \eta \rho_{%s} + (1-\eta) \rho_{noise}$'%state)
plt.xlabel(r'$\eta$')
plt.ylabel(r'$p(\rho)$')
plt.grid()
plt.legend()
# plt.ylim([0, 1])
plt.show()
plt.savefig(state + os.sep + '%s_%d.png'%(state, n_qubits))
'''

