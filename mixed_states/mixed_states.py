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
n_alpha = 10

PRINT_STATES = 0
PRINT_F = 0

aux.check_dir(state)

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
    
    elif state=='GHZ_noise':
        coefficients = np.random.rand(2**n_qubits)
        coefficients[0] = coefficients[-1] = 0
        coefficients = coefficients/np.linalg.norm(coefficients)
    
    elif state=='bitflip':
        # n_ones = 1
        n_zeros = n_qubits - n_ones
        
        bits = set([''.join(x) for x in itertools.permutations(n_ones * '1' + n_zeros * '0', 4)])
        
        coefficients = np.zeros(2**n_qubits)
        options = []
        for b in bits:
            options.append(int(b, base=2))
        i = np.random.choice(options)
        coefficients[i] = 1
    
    elif state=='bitflip_1':
        coefficients = np.zeros(2**n_qubits)
        # i = np.random.randint(1, 2**n_qubits-1)
        i = np.random.choice([1, 2, 4, 8])
        coefficients[i] = 1
        
    elif state=='bitflip_2':
        coefficients = np.zeros(2**n_qubits)
        # i = np.random.randint(1, 2**n_qubits-1)
        i = np.random.choice([3, 5, 6, 10, 12])
        coefficients[i] = 1
        # coefficients = coefficients/np.linalg.norm(coefficients)
    elif state=='bitflip_3':
        coefficients = np.zeros(2**n_qubits)
        # i = np.random.randint(1, 2**n_qubits-1)
        i = np.random.choice([7, 11, 13, 14])
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
    # print(20*'=')
    print(state + ':')
    print(psi)



#%%

def simulate_state(state, alpha, noise_type, n_ones_bitflip=0):
    # states = ['GHZ', 'W', 'noise']
    # alpha between 0 and 1.
    
    rho = {}
    
    # rho['W'] = density_matrix('W', n_qubits)
    # rho['GHZ'] = density_matrix('GHZ', n_qubits)
    
    rho[state] = density_matrix(state, n_qubits)
    rho[noise_type] = density_matrix(noise_type, n_qubits, n_ones=n_ones_bitflip)
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

def privacy(W, F):
    p = np.trace(np.dot(W, F))/(np.trace(W)*np.trace(F))
    return p
#%%

data = pd.DataFrame()

w = np.ones(n_qubits) # 2*np.random.rand(d) - np.ones(d)

w = w/np.linalg.norm(w) # normalisation
W = np.outer(w, w)

# data.at[i, 'w'] = w

a = np.linspace(0, 1, num=n_alpha)
# alpha = 0.5


# GHZ_noise is noise everywhere except |00...0> and |11...1>
# single_excitations_noise is one other element of the base (e.g. |1000>)

noise_types = ['GHZ_noise', 'bitflip']

for noise_type in noise_types:
    if noise_type=='GHZ_noise':
        print(noise_type)
        for alpha in a:
            for i in range(n):
                l = len(data)
                
                F = simulate_state(state, alpha, noise_type)
                data.at[l+1, 'alpha'] = alpha
                
                p = privacy(W, F)
                data.at[l+1, 'p_' + '_' + noise_type] = p
    else:
        for n_ones in range(1, n_qubits):
            print(noise_type, n_ones, 'ones')
            for alpha in a:
                for i in range(n):
                    l = len(data)
                    
                    F = simulate_state(state, alpha, noise_type, n_ones)
                    data.at[l+1, 'alpha'] = alpha
                    
                    p = privacy(W, F)
                    data.at[l+1, 'p_' + '_' + noise_type + '_' + str(n_ones) + '_ones'] = p
            
#%%

plt.figure()

for c in data.columns[1:]:
    plt.scatter(data['alpha'], data[c], alpha=0.2, label=c)
    
plt.title(r'$\rho = \alpha \rho_{%s} + (1-\alpha) \rho_{noise}$'%state)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$p(\rho)$')
plt.grid()
plt.legend()
# plt.ylim([0, 1])
plt.show()
plt.savefig(state + os.sep + '%s_%d.png'%(state, n_qubits))
