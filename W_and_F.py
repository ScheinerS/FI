# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import aux

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

states = ['W', 'GHZ']


plt.close('all')

dim = range(2, 6)

F = {}

for state in states:
    aux.check_dir(state)
    F[state] = {}

for d in dim:    
    # W states matrix:
    F['W'][d] = ((d-4)/d)*np.ones((d, d)) # whole matrix
    np.fill_diagonal(F['W'], 1) # replace the diagonal
    
    # GHZ states matrix:
    F['GHZ'][d] = np.ones((d, d)) # whole matrix


#%%

n = 10000

data = pd.DataFrame(columns=['w', 'p'])

for state in states:
    print(20*'-')
    print('State:\t', state)
    for d in dim:
        print('d=\t', d)
        for i in range(n):
            w = 2*np.random.rand(d) - np.ones(d)
            W = np.outer(w, w)
            
            data.at[i, 'w'] = w
           
            data.at[i, 'p'] = np.trace(np.dot(W, F[state]))/(np.trace(F[state])*np.trace(W))
        
        label = '%s states'%state
        title = r'%s - %d random functions $\{w_{i}\}$ - dim=%d'%(label, n, d)
        
        plt.figure()
        plt.scatter(range(len(data)), data['p'], alpha=0.1)
        plt.title(title)
        plt.xlabel(r'i')
        plt.ylabel(r'p')
        plt.ylim([0, 1])
        plt.show()
        plt.savefig(state + os.sep + title + '.png')
        
        plt.figure()
        plt.hist(data['p'], alpha=1.0)
        plt.title(title)
        plt.xlabel(r'p')
        plt.ylabel(r'')
        # plt.ylim([0, 1])
        plt.show()
        plt.savefig(state + os.sep + title + '_hist.png')
        
        
        data['p'] = pd.to_numeric(data['p'])
        
        id_max = data['p'].idxmax()
        
        print('States:\t',state)
        print('F =\t', F[state])
        print('Maximum values:\nw = ', data.at[id_max, 'w'], '\np = ', data.at[id_max, 'p'])
