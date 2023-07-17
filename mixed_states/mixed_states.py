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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'font.size': 15
})


plt.close('all')

n_qubits = 4
state = 'GHZ'
# states = ['GHZ', 'W']

n = 50
n_alpha = 50

PRINT_STATES = 0
PRINT_F = 0

aux.check_dir(state)

#%%
def density_matrix(state, n_qubits):
    if state=='GHZ':
        coefficients = np.zeros(2**n_qubits)
        coefficients[0] = coefficients[-1] = 1
        coefficients = coefficients/np.linalg.norm(coefficients)

    elif state=='W':
        coefficients = np.zeros(2**n_qubits)
        
        for n in range(n_qubits):
            coefficients[2**n]=1
        
        coefficients = coefficients/np.linalg.norm(coefficients)
    
    elif state=='GHZ_noise':
        coefficients = np.random.rand(2**n_qubits)
        coefficients[0] = coefficients[-1] = 0
        coefficients = coefficients/np.linalg.norm(coefficients)
        
    elif state=='single_excitations_noise':
        coefficients = np.zeros(2**n_qubits)
        i = np.random.randint(1, 2**n_qubits-1)
        coefficients[i] = 1
        # coefficients = coefficients/np.linalg.norm(coefficients)
    
    if PRINT_STATES:
        print_state(state, coefficients)
    
    rho = np.tensordot(coefficients, coefficients, axes=0)
    return rho



def print_state(state, coefficients):
    
    basis = list(itertools.product([0, 1], repeat=n_qubits))
    psi = ''
    for c,b in zip(coefficients, basis):
        if c:
            psi = psi + ' + ' + '%.8f'%c + '\t| %s >'%str(b).strip('()') + '\n'
    print(state + ':')
    print(psi)



#%%

def simulate_state(state, alpha, noise_type):
    # states = ['GHZ', 'W', 'noise']
    # alpha between 0 and 1.
    
    rho = {}
    
    # rho['W'] = density_matrix('W', n_qubits)
    # rho['GHZ'] = density_matrix('GHZ', n_qubits)
    
    rho[state] = density_matrix(state, n_qubits)
    rho[noise_type] = density_matrix(noise_type, n_qubits)
    # rho['single_excitations_noise'] = density_matrix('single_excitations_noise', n_qubits)
    
    
    
    rho['MIXED'] = alpha * rho[state] + (1-alpha) * rho[noise_type]
    
    sigma_z = np.diag([1, -1])
    id_2 = np.diag([1, 1])
    
    
    L = {}
    
    for i in range(n_qubits):
        L[i] = {}
    
    for i in range(n_qubits):
        for j in range(n_qubits):
            operators = []
            for k in range(n_qubits):
                operators.append(id_2)
            if not i==j:
                operators[i] = operators [j] = sigma_z # replaces the identitiy matrices with sigma_w in places 'i' and 'j'.
            
            # Printing:
            # print('L[%d][%d]'%(i,j))
            # print(operators)
            # print('\n')
            
            L[i][j] = operators[0]
    
            for k in range(1, n_qubits):
                L[i][j] = np.kron(L[i][j], operators[k])
                # print(L[i][j])
            
    # L = np.kron(np.kron(id_2, id_2), np.kron(sigma_z, sigma_z))
    
    
    F = np.zeros((n_qubits, n_qubits))
    
    for i in range(n_qubits):
        for j in range(n_qubits):
            F[i][j] = np.trace(np.matmul(rho['MIXED'], L[i][j]))
     
    if PRINT_F:
        print('F:')
        print(F)
    return F


#%%

data = pd.DataFrame(columns=['alpha', 'p'])

w = np.ones(n_qubits) # 2*np.random.rand(d) - np.ones(d)

w = w/np.linalg.norm(w) # normalisation
W = np.outer(w, w)

# data.at[i, 'w'] = w

a = np.linspace(0, 1, num=n_alpha)
# alpha = 0.5


# noise_types = ['GHZ_noise', 'single_excitations_noise']
# GHZ_noise is noise everywhere except |00...0> and |11...1>
# single_excitations_noise is one other element of the base (e.g. |1000>)

# noise_types = ['single_excitations_noise']
noise_types = ['GHZ_noise']

for noise_type in noise_types:
    for alpha in a:
        for i in range(n):
            l = len(data)
            F = simulate_state(state, alpha, noise_type)
            data.at[l+1, 'alpha'] = alpha
            p = np.trace(np.dot(W, F))/(np.trace(F)*np.trace(W))
            data.at[l+1, 'p'] = p
            
            # data.append([alpha, p])
#%%

plt.figure()
plt.scatter(data['alpha'], data['p'], alpha=0.2)
plt.title(r'$\rho = \alpha \rho_{%s} + (1-\alpha) \rho_{noise}$'%state)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$p(\rho)$')
plt.grid()
# plt.ylim([0, 1])
plt.show()
plt.savefig(state + os.sep + '%s_%d_%s.png'%(state, n_qubits, noise_type))
