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
import plot_functions as pf
import kraus_operators as ko
import save_results as sr

import plt_parameters



#%%
def density_matrix(state, d, alpha=0):
    if state=='GHZ':
        coefficients = np.zeros(2**d)
        coefficients[0] = coefficients[-1] = 1
        coefficients = coefficients/np.linalg.norm(coefficients)
    
    elif state=='plus':
        coefficients = np.ones(2**d)
        coefficients = coefficients/np.linalg.norm(coefficients)
  
    elif state=='W':
        coefficients = np.zeros(2**d)
        for n in range(d):
            coefficients[2**n] = 1
        coefficients = coefficients/np.linalg.norm(coefficients)
    
    elif state=='random_noise':
        coefficients = 2*np.random.rand(2**d)-1 + 2*np.random.rand(2**d)*1j-1j
        coefficients[0] = coefficients[-1] = 0
        coefficients = coefficients/np.linalg.norm(coefficients)
        
    if flags['print_states']:
        print_state(state,d, coefficients)
    
    rho = np.tensordot(np.conj(coefficients), coefficients, axes=0)
    return rho







def print_state(state, d, coefficients):
    
    basis = list(itertools.product([0, 1], repeat=d))
    psi = ''
    for c,b in zip(coefficients, basis):
        if c:
            psi = psi + ' + ' + '%.3f'%c + '\t| %s >'%str(b).strip('()') + '\n'
    
    print(state + ':')
    print(psi)



#%%

def H_and_M(d, print_matrices:bool=False):
    
    sigma_z = np.diag([1, -1])
    id_2 = np.diag([1, 1])
    
    # Operators:
    H = {}
    for i in range(d):
        H[i] = np.ones((1,1))
        for j in range(d):
            if i==j:
                H[i] = np.kron(H[i], sigma_z/2)
            else:
                H[i] = np.kron(H[i], id_2)
    
    
    M = {}
    
    for i in range(d-1):
        M[i] = {}
    
    for i in range(d):
        for j in range(i, d):
            if j!=i:
                M[i][j] = H[i] - H[j]

            # Printing:
            if print_matrices:
                print('M[%d][%d] =\n'%(i,j), M[i][j], '\n')
    
    return H, M


def trace_norm(m):
    '''
    Parameters
    ----------
    M : Numpy 2d-array

    Returns
    -------
    The "trace norm" of M.

    '''
    
    '''
    # Examples:
    m = np.matrix('1 2 3; 1 1 1; 2 2 2')
    m = np.matrix('1 0 0; 0 1 0; 0 0 1')
    m = np.matrix('0.5 0 0 0.5; 0 0 0 0; 0 0 0 0; 0.5 0 0 0.5')

    m = np.matrix('0.5 0; 0 -0.5')
    m = np.matrix('-1 0 0; 0 -2 0; 0 0 3')
    '''
    s = sum(np.linalg.svd(m)[1])
    return s

def apply_noise(rho, d:int, K:list):
    '''
    Parameters
    ----------
    rho : Numpy 2d-array
        Density matrix of the state to apply .
    d : int
        dimension.
    K : list
        List of Kraus operators.

    Returns
    -------
    s : Numpy 2d-array
        Density matrix after noise.
    '''
    s = np.zeros((2**d, 2**d))
    for i in range(len(K)):
        s = s + np.matmul(np.matmul(K[i], rho), np.conjugate(np.transpose(K[i])))
    s = s
    return s


#%%


def mixed_states(d:int, state:str, eta_values:list, alpha_values:list, plots_path:str, colours:dict, colourbar_limits:dict, flags:dict):
    
    H, M = H_and_M(d, print_matrices=False)
    
    if flags['plot_H_and_M']:
        for i in range(len(H)):
            pf.plot_matrix(H[i],
                           plot='3d',
                        save = flags['save_H_and_M'],
                        save_name = 'H_%d'%(i),
                        path = plots_path + 'H',
                        colour = colours['H'],
                        clim = colourbar_limits['H'],
                        save_as = 'png'
                        )
    
        for i in M.keys():
            for j in M[i].keys():
                pf.plot_matrix(M[i][j],
                            save = flags['save_H_and_M'],
                            save_name = 'M_%d_%d'%(i,j),
                            path = plots_path + 'M',
                            colour = colours['M'],
                            clim = colourbar_limits['M'],
                            save_as = 'png'
                            )
    
    data = {}
    
    data['parameters'] = pd.DataFrame()
    
    data['parameters']['parameter'] = parameters.keys()
    data['parameters']['value'] = parameters.values()
    
    noise_types = ['Amplitude Damping', 'Depolarising Noise']
    
    for noise in noise_types:
        data[noise] = pd.DataFrame(columns = ['alpha', 'eta', 'epsilon'])
    
    
    for alpha in alpha_values:
        for eta in eta_values:
            print('\ralpha:\t%.2f\t\teta:\t%.2f'%(alpha, eta), end='')
            Kraus_operators = {'Amplitude Damping': ko.amplitude_damping_Kraus_operators(d, eta),
                               'Depolarising Noise': ko.depolarising_Kraus_operators(d, eta),
                               }
            
            for noise in Kraus_operators.keys():
                
                path = plots_path + noise + os.sep
                
                K = Kraus_operators[noise]
                
                rho_GHZ = density_matrix('GHZ', d)
                rho_plus = density_matrix('plus', d)
                
                rho_0 = (1-alpha) * rho_GHZ + alpha * rho_plus
    
                # rho_GHZ = density_matrix(state, d, alpha)
                rho_with_noise = apply_noise(rho_0, d , K)
                
                if flags['save_rho_csv']:
                    aux.check_dir(plots_path)
                    sr.save_matrix(m = rho_with_noise,
                                   d = d,
                                   eta = eta,
                                   alpha = alpha,
                                   filename = 'rho_eta=%.2f_alpha=%.2f'%(eta, alpha),
                                   path = path + os.sep + 'rho_txt',
                                   noise = noise)
                
                
                if flags['plot_rho']:
                    pf.plot_matrix(np.real(rho_with_noise),
                                   plot='3d',
                                   title=r'$\alpha=%.2f \quad \eta=%.2f$ - $Re$'%(alpha,eta),
                                   save=flags['save_rho'],
                                   save_name='rho_eta=%.2f_alpha=%.2f_re'%(eta, alpha),
                                   path = path + 'rho',
                                   colour=colours['rho_re'],
                                   clim=colourbar_limits['rho_re'],
                                   save_as = 'png'
                                   )
                    pf.plot_matrix(np.imag(rho_with_noise),
                                   plot='3d',
                                   title=r'$\alpha=%.2f \quad \eta=%.2f$ - $Im$'%(alpha,eta),
                                   save=flags['save_rho'],
                                   save_name='rho_eta=%.2f_alpha=%.2f_im'%(eta, alpha),
                                   path = path + 'rho',
                                   colour=colours['rho_im'],
                                   clim=colourbar_limits['rho_im'],
                                   save_as = 'png'
                                   )
                
        
                C = {}
                for i in range(d-1):
                    C[i] = {}
                
                norms = []
                trace_norms = []
                for i in M.keys():
                    for j in M[i].keys():
                        C[i][j] = np.matmul(M[i][j], rho_with_noise) - np.matmul(rho_with_noise, M[i][j])
                        norms.append(np.linalg.norm(C[i][j], 1))
                        trace_norms.append(trace_norm(C[i][j]))
                
                # print('norms:', norms)
                # epsilon = max(norms)
                epsilon = max(trace_norms)
                
                l = len(data[noise])
                data[noise].at[l, 'alpha'] = alpha
                data[noise].at[l, 'eta'] = eta
                data[noise].at[l, 'epsilon'] = epsilon
                
                
                if flags['plot_C']:
                    for i in C.keys():
                        for j in C[i].keys():
                            pf.plot_matrix(np.real(C[i][j]),
                                           title='$\eta=%.2f$ - $Re$'%eta,
                                           save=flags['save_C'],
                                              save_name='C_%d_%d_eta=%.2f_alpha=%.2f_re'%(i, j, eta, alpha),
                                              path = path + 'C',
                                              colour = colours['C'],
                                              clim = colourbar_limits['C'],
                                              save_as = 'png'
                                              )
                            pf.plot_matrix(np.imag(C[i][j]),
                                           title='$\eta=%.2f$ - $Im$'%eta,
                                           save=flags['save_C'],
                                           save_name='C_%d_%d_eta=%.2f_alpha=%.2f_im'%(i,j, eta, alpha),
                                           path = path + 'C',
                                           colour=colours['C'],
                                           clim=colourbar_limits['C'],
                                           save_as = 'png'
                                           )
            plt.close('all')
    
    if flags['save_results']:
        sr.save_results(data=data, filename='data_' + state + '_d=' + str(d))
                    
    if flags['plot_epsilon']:
        for noise in noise_types:
            pf.plot_epsilon(d,
                            data,
                            noise,
                            plot = '3d',
                            title = '',
                            save = flags['save_epsilon'],
                            save_name = 'data',
                            path = plots_path,
                            state = state
                            )
        
#%%

if __name__=='__main__':
    
    # parameters = aux.read_parameters('parameters.txt')
    # TODO: read parameters from file.

    # Parameters
    parameters = {}
    
    parameters['date'] = aux.get_date()
    
    parameters['d'] = 3   # Number of qubits.
    
    parameters['state'] = 'GHZplus'
    states = ['GHZ', 'plus', 'GHZplus']
    
    parameters['eta_min'] = 0
    parameters['eta_max'] = 1
    parameters['n_eta'] = 101
    parameters['eta_values'] = np.linspace(parameters['eta_min'],
                                           parameters['eta_max'],
                                           num = int(parameters['n_eta'])
                                           )
    
    parameters['alpha_min'] = 0
    parameters['alpha_max'] = 1
    parameters['n_alpha'] = 1
    parameters['alpha_values'] = np.linspace(parameters['alpha_min'],
                                           parameters['alpha_max'],
                                           num = int(parameters['n_alpha'])
                                           )
    
    plots_path = 'plots' + os.sep + parameters['state'] + os.sep + 'd=' + str(parameters['d']) + os.sep  # all plots will be saved in this directory.
    
    parameters['colours'] = {'H': 'coolwarm',
               'M': 'PiYG',
               'rho_re': 'inferno_r',
               'rho_im': 'cividis',
               'C': 'viridis'}
    
    parameters['colourbar_limits'] = {'H': [-1, 1],
                                      'M': [-1, 1],
                                      'rho_re': [-0.55, 0.55],
                                      'rho_im': [-0.55, 0.55],
                                      'C': [-0.55, 0.55]}
    # Flags
    
    flags = {'verify_parameters': 1,
             'save_parameters': 1,
             
             'print_states': 0,
             
             'plot_H_and_M': 0,
             'save_H_and_M': 1,
             
             'plot_rho': 1,
             'save_rho': 1,
             'save_rho_csv': 1, # rho in TXT
             
             'plot_C': 1,
             'save_C': 1,
             
             'plot_epsilon': 1,
             'save_epsilon': 1,
             
             'save_results': 1, # xlsx
             }
    
    if flags['verify_parameters']:
        aux.verify_parameters(parameters)
    
    mixed_states(d = parameters['d'],
                 state = parameters['state'],
                 eta_values = parameters['eta_values'],
                 alpha_values = parameters['alpha_values'],
                 plots_path = plots_path,
                 colours = parameters['colours'],
                 colourbar_limits = parameters['colourbar_limits'],
                 flags = flags,
                 )