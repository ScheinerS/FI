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
        # 'size'   : 22
        }

plt.rc('font', **font)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'font.size': 12
})


plt.close('all')


#%% Parameters

n_qubits = 3
state = 'GHZ'
# states = ['GHZ', 'W']

n = 1 # number of states to simulate for each combination of eta, etc.
n_rand_states = 1 # number of random states per value of eta. TODO: adapt for this.
n_eta = 100

colours = {'H': 'YlOrRd',
           'L': 'PuBuGn',
           'rho_re': 'inferno',
           'rho_im': 'cividis',
           'C': 'viridis'}

colourbar_limits = {'H': [-1, 1],
                    'L': [-2, 2],
                    'rho_re': [None, None],
                    'rho_im': [None, None],
                    'C': [0, 1]}
#%% Flags

flags = {'print_states': 0,
          'plot_states': 0,
          
          'print_H_and_L': 0,
          'plot_H_and_L': 0,
          'save_H_and_L': 0,
          
          'print_rho': 0,
          'plot_rho': 0,
          'save_rho': 0,
          
          'print_C': 0,
          'plot_C': 0,
          'save_C': 0,
          
          'plot_epsilon': 1,
          'save_epsilon': 1,
          }

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
            
    if flags['print_states']:
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
    # plt.show()
    if save:
        # aux.check_dir(str(n_qubits)+'_qubits')
        aux.check_dir(path)
        plt.savefig(path + os.sep + save_name + '.pdf')
        print('Saved:\t', path + os.sep + save_name + '.pdf')



def print_state(state, coefficients):
    
    basis = list(itertools.product([0, 1], repeat=n_qubits))
    psi = ''
    for c,b in zip(coefficients, basis):
        if c:
            psi = psi + ' + ' + '%.8f'%c + '\t| %s >'%str(b).strip('()') + '\n'
    
    print(state + ':')
    print(psi)



#%%

def H_and_L(n_qubits, print_matrices:bool=False):
    
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
def depolarising_Kraus_operators(n_qubits, eta, verify:bool=False):
    
    a = np.sqrt((1+3*eta)/4)
    b = np.sqrt((1-eta)/4)
    
    K_A = a * np.array([[1, 0], [0, 1]])
    K_B = b * np.array([[0, 1], [1, 0]])
    K_C = b * np.array([[0, -1j], [1j, 0]])
    K_D = b * np.array([[1, 0], [0, -1]])
    
    lists = list(itertools.product([K_A, K_B, K_C, K_D], repeat=n_qubits))
    
    K = {}
    for i in range(len(lists)):
        k = np.array([1])
        for j in range(len(lists[i])):
            k = np.kron(k, lists[i][j])
        
        K[i] = k
        
    if verify:
        verify_K(K)
    
    return K



def amplitude_damping_Kraus_operators(n_qubits, eta, verify:bool=False):
    K_A = np.array([[1, 0], [0, np.sqrt(eta)]])
    K_B = np.array([[0, np.sqrt(1-eta)], [0, 0]])
    
    lists = list(itertools.product([K_A, K_B], repeat=n_qubits))
    
    K = {}
    for i in range(len(lists)):
        k = np.array([1])
        for j in range(len(lists[i])):
            k = np.kron(k, lists[i][j])
        
        K[i] = k
    
    if verify:
        verify_K(K)
    
    return K




def verify_K(K, verbose:bool=False):
    s = np.zeros((2**n_qubits, 2**n_qubits))
    print('')
    for i in range(len(K)):
        s = s + np.matmul(np.conjugate(np.transpose(K[i])), K[i])
        #print(i, s) # to print step by step.
    if verbose:
        print(50*'-' + '\n', 'Verification:\n', s, '\n', 50*'-')
        
    if s == np.identity(n_qubits):
        return True
    else:
        return False




def rho_after_noise(rho, K):
    s = np.zeros((2**n_qubits, 2**n_qubits))
    for i in range(len(K)):
        s = s + np.matmul(np.matmul(K[i], rho), np.conjugate(np.transpose(K[i])))
    s = s
    return s

#%% Remove later:
'''
n_qubits = 2
eta = 0.5
print('n_qubits = ', n_qubits)
print('eta = ', eta)

rho = density_matrix(state, n_qubits)
print('rho =\n', rho)

print('\nAmplitude damping:')
K_amplitude = amplitude_damping_Kraus_operators(n_qubits, eta, verify=False)
rho_amplitude = rho_after_noise(rho, K_amplitude)
print('rho_amplitude =\n', rho_amplitude)

print('\nDepolarising noise:')
K_depolarisation = depolarising_Kraus_operators(n_qubits, eta, verify=False)
rho_depolarisation = rho_after_noise(rho, K_depolarisation)
print('rho_depolarisation =\n', rho_depolarisation)
'''
#%%

eta_values = np.linspace(0, 1, num=n_eta)
# eta_values = [0.1]

H, L = H_and_L(n_qubits, print_matrices=False)

if flags['plot_H_and_L']:
    for i in range(len(H)):
        plot_matrix(H[i],
                    save = flags['save_H_and_L'],
                    save_name = 'H_%d'%(i),
                    path = str(n_qubits) + '_qubits' + os.sep + 'H',
                    colour = colours['H'],
                    clim = colourbar_limits['H']
                    )

    for i in L.keys():
        for j in L[i].keys():
            plot_matrix(L[i][j],
                        save = flags['save_H_and_L'],
                        save_name = 'L_%d_%d'%(i,j),
                        path = str(n_qubits) + '_qubits' + os.sep + 'L',
                        colour = colours['L'],
                        clim = colourbar_limits['L'])

data = {}

noise_types = ['Amplitude Damping', 'Depolarising Noise']

for noise in noise_types:
    data[noise] = pd.DataFrame(columns = ['eta', 'epsilon'])
    
for eta in eta_values:
    Kraus_operators = {'Amplitude Damping': amplitude_damping_Kraus_operators(n_qubits, eta),
                       'Depolarising Noise': depolarising_Kraus_operators(n_qubits, eta),
                       }
    
    for key in Kraus_operators.keys():
        
        path = str(n_qubits) + '_qubits' + os.sep + key
        
        K = Kraus_operators[key]
        # l = len(data)
        
        rho = density_matrix(state, n_qubits)
        
        rho_with_noise = rho_after_noise(rho, K)
        
        if flags['plot_rho']:
            plot_matrix(np.real(rho_with_noise),
                        title='$\eta=%.2f$ - $Re$'%eta,
                        save=flags['save_rho'],
                        save_name='rho_%.2f_re'%(eta),
                        path = path + os.sep + 'rho',
                        colour=colours['rho_re'],
                        clim=colourbar_limits['rho_re']
                        )
            plot_matrix(np.imag(rho_with_noise),
                        title='$\eta=%.2f$ - $Im$'%eta,
                        save=flags['save_rho'],
                        save_name='rho_%.2f_im'%(eta),
                        path = path + os.sep + 'rho',
                        colour=colours['rho_im'],
                        clim=colourbar_limits['rho_im']
                        )
        

        C = {}
        for i in range(n_qubits-1):
            C[i] = {}
        
        norms = []
        for i in L.keys():
            for j in L[i].keys():
                C[i][j] = np.matmul(L[i][j], rho_with_noise) - np.matmul(rho_with_noise, L[i][j])
                norms.append(np.linalg.norm(C[i][j], 1))
        
        # print('Norms:', norms) # To print the 1-norm of each all C_ij.
        epsilon = max(norms)
        
        l = len(data[key])
        data[key].at[l, 'eta'] = eta
        data[key].at[l, 'epsilon'] = epsilon
        
        if flags['plot_C']:
            for i in C.keys():
                for j in C[i].keys():
                    plot_matrix(np.real(C[i][j]),
                                title='$\eta=%.2f$ - $Re$'%eta,
                                save=flags['save_C'],
                                save_name='C_%d_%d_%.2f_re'%(i,j, eta),
                                path= path + os.sep + 'C',
                                colour=colours['C'],
                                clim=colourbar_limits['C']
                                )
                    plot_matrix(np.imag(C[i][j]),
                                title='$\eta=%.2f$ - $Im$'%eta,
                                save=flags['save_C'],
                                save_name='C_%d_%d_%.2f_im'%(i,j, eta),
                                path = path + os.sep + 'C',
                                colour=colours['C'],
                                clim=colourbar_limits['C']
                                )
    
    plt.close('all')



for noise in noise_types:
    if flags['plot_epsilon']:
        plt.figure()
        plt.scatter(data[noise]['eta'], data[noise]['epsilon'], alpha = 0.5)
        plt.title(r'%s'%noise)
        plt.xlabel(r'$\eta$')
        plt.ylabel(r'$\epsilon(\eta)$')
        plt.grid()
        plt.show()
        if flags['save_epsilon']:
            path = str(n_qubits) + '_qubits' + os.sep + noise
            aux.check_dir(path)
            plt.savefig('epsilon_%s.pdf'%(noise))