#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:03:36 2023

@author: santiago
"""

import numpy as np
import itertools

n_qubits = 4

basis = list(itertools.product([0, 1], repeat=n_qubits))

rho_GHZ = np.zeros((2**n_qubits, 2**n_qubits))
rho_GHZ[0][0] = rho_GHZ[0][-1] = rho_GHZ[-1][0] = rho_GHZ[-1][-1] = 1/2

sigma_z = np.diag([1, -1])
id_2 = np.diag([1, 1])

#%%
L = {}

# ARREGLAR DESDE ACA:

for i in range(n_qubits):
    L[i] = {}

for i in range(n_qubits):
    for j in range(n_qubits):
        operators = []
        for k in range(n_qubits):
            operators.append(id_2)
        operators[i] = operators [j] = sigma_z
        print(operators)
        
        L[i][j] = np.kron(np.kron(operators[0], operators[1]), np.kron(operators[2], operators[3]))
        
# L = np.kron(np.kron(id_2, id_2), np.kron(sigma_z, sigma_z))

F = np.zeros((n_qubits, n_qubits))

for i in range(n_qubits):
    for j in range(n_qubits):
        F[i][j] = np.trace(np.matmul(rho_GHZ, L[i][j]))


# Arreglar. La matriz deberia tener '1' en todos lados. Corroborar que L[i][j] tenga sentido.




# # Checking Kronecker product:
# m1 = np.array([[1, -1],[1, -1]])
# m2 = np.array([[2, -3],[4, 5]])
# m3 = np.kron(m1, m2)

