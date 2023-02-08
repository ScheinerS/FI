#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:04:59 2023

@author: santiago
"""
import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt

d = 10 # dimension
print_matrices = 0
save_figures = 0

B = -(1/d)*np.ones((d, d)) # whole matrix
np.fill_diagonal(B, (d-1)/d) # replace the diagonal
B[:,0] = np.ones(d) # replace the first column


Bt = np.transpose(B)

M = np.linalg.inv(B)

Mt = np.transpose(M)

if print_matrices:
    print('B = \n', B)
    print('Bt = \n', Bt, '\n')
    print('M = \n', M)
    print('Mt = \n', Mt, '\n')
    
#%%

force_F_prime = 1

if force_F_prime:
    F_prime = np.zeros((d, d))
    k = 2   # place in the diagonal
    F_prime[k, k] = 1
    F = np.matmul(Mt, np.matmul(F_prime, M)) # Mt*F_prime*M 
else:
    F = np.ones((d, d))
    # F = np.array([[1, 2, 3], [4, 5, 6], [-2, 0, 0]])
    print('F = \n', F)

    F_prime = np.matmul(Bt, np.matmul(F, B)) # Bt*F*B

if print_matrices:
    print('F = \n', F, '\n')
    print("F' = \n", F_prime, '\n')

#%%

# import matplotlib as mpl

def plot_matrix(A, name):
    # prints matrix 'A'
    # norm = mpl.colors.Normalize(vmin=A.min, vmax=A.max)
    plt.matshow(A)#, norm=norm)
    plt.title("%s (d=%d)"%(name,d))
    
    plt.colorbar()#norm=norm)
    plt.show()
    if save_figures:
        plt.savefig("img/%s_d=%d.png"%(name, d))

plt.close('all')
plot_matrix(F, 'F')
plot_matrix(F_prime, "F'")
