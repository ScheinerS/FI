#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:57:34 2023

@author: santiago
"""

import numpy as np
import itertools

def depolarising_Kraus_operators(d, eta, verify:bool=False):
    '''
    Parameters
    ----------
    d : int
        Number of parameters.
    eta : float
        noise parameter:
            eta=1: pure state and no noise,
            eta=0: only noise, the state is lost completely.
    verify : bool, optional
        Verification for the Kraus operators. The default is False.

    Returns
    -------
    K : TYPE
        List of amplitude damping Kraus operators.
    '''
    
    a = np.sqrt(1-3*eta/4)
    b = np.sqrt(eta/4)
    
    K_A = a * np.array([[1, 0], [0, 1]])
    K_B = b * np.array([[0, 1], [1, 0]])
    K_C = b * np.array([[0, -1j], [1j, 0]])
    K_D = b * np.array([[1, 0], [0, -1]])
    
    lists = list(itertools.product([K_A, K_B, K_C, K_D], repeat=d))
    
    K = {}
    for i in range(len(lists)):
        k = np.array([1])
        for j in range(len(lists[i])):
            k = np.kron(k, lists[i][j])
        
        K[i] = k
        
    if verify:
        verify_K(K)
    
    return K



def amplitude_damping_Kraus_operators(d:int, eta:float, verify:bool=False):
    '''
    Parameters
    ----------
    d : int
        Number of parameters.
    eta : float
        noise parameter:
            eta=1: pure state and no noise,
            eta=0: only noise, the state is lost completely.
    verify : bool, optional
        Verification for the Kraus operators. The default is False.

    Returns
    -------
    K : TYPE
        List of amplitude damping Kraus operators.

    '''
    
    K_A = np.array([[1, 0], [0, np.sqrt(1-eta)]])
    K_B = np.array([[0, np.sqrt(eta)], [0, 0]])
    
    lists = list(itertools.product([K_A, K_B], repeat=d))
    
    K = {}
    for i in range(len(lists)):
        k = np.array([1])
        for j in range(len(lists[i])):
            k = np.kron(k, lists[i][j])
        
        K[i] = k
    
    if verify:
        verify_K(K)
    
    return K



def verify_K(K:list, d:int, verbose:bool=True):
    '''
    Parameters
    ----------
    K : list
        List of Kraus operators.
    d : int
        Dimension (i.e. number of parameters).
    verbose : bool, optional
        If True, it prints the result of the matrix multiplication of (K^t)*K,
        which should be the dxd identity matrix.
        The default is False.

    Returns
    -------
    bool
    True, if the Kraus operators are well defined.
    False, otherwise.
    '''
    
    s = np.zeros((2**d, 2**d))
    print('')
    for i in range(len(K)):
        s = s + np.matmul(np.conjugate(np.transpose(K[i])), K[i])
        #print(i, s) # to print step by step.
    if verbose:
        print(50*'-' + '\n', 'Verification:\n', s, '\n', 50*'-')
        
    # if s == np.identity(d):
    #     return True
    # else:
    #     return False


if __name__=='__main__':
    d = 2
    eta = 0.5
    
    print(20*'*'+'\n')
    
    print('Amplitude damping:')
    K_amp_dam = amplitude_damping_Kraus_operators(d, eta)
    verify_K(K_amp_dam, d, verbose=True)

    print('Depolarising noise:')    
    K_dep_noise = depolarising_Kraus_operators(d, eta)
    verify_K(K_dep_noise, d, verbose=True)