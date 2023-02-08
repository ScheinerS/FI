#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:15:21 2023

@author: santiago
"""

import numpy as np
import itertools


def compute_Fisher_matrix(basis):
    # 'coeficients' is a list of the coeficients for the state, expressed in the computational basis
    for i in range(len(basis)):
        for j in range(len(basis)):
            F[i,j] = 1
            # TO DO.

def get_distance(F):
    # receives matrix "F" and returns the total sum of squared distances.
    dist = 0

    I, J = F.shape
    for i in range(I):
        for j in range(J):
            for k in range(i, I):
                for l in range(j, J):
                    dist += abs(F[i,j]-F[k,l])**2
    
    # print(dist)
    return dist

def print_basis(basis):
    for state in basis:
        print("|", end='')
        for q in state:
            print(q, end='')
        print(">")
#%%

# Examples for Fisher matrices:
    
# F = np.matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
F = np.ones((3, 3))
F[0, 0] = 1.2
print(F)

#%%
D = get_distance(F)
print("Total sum of squared distances for F:\n", D)

n_cubits = 2
basis = list(itertools.product([0, 1], repeat=n_cubits))

print_basis(basis)
l=len(basis)
initial_coeficients = np.zeros(l)
F = compute_Fisher_matrix()
