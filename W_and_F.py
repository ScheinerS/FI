# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "Helvetica"
# })

label = 'Werner states'

plt.close('all')

d = 3

F = -(1/d)*np.ones((d, d)) # whole matrix
np.fill_diagonal(F, 1) # replace the diagonal

n = 10000

data = pd.DataFrame(columns=['w', 'p'])

for i in range(n):
    w = 2*np.random.rand(3) - np.ones(3)
    W = np.outer(w, w)
    
    data.at[i, 'w'] = w
   
    data.at[i, 'p'] = np.trace(np.dot(W, F))/(np.trace(F)*np.trace(W))

title = r'%s - %d random functions $\{w_{i}\}$ - dim=%d'%(label, n, d)

plt.figure()
plt.scatter(range(len(data)), data['p'], alpha=0.1)
plt.title(title)
plt.xlabel(r'i')
plt.ylabel(r'p')
plt.ylim([0, 1])
plt.show()
plt.savefig(title + '.png')

plt.figure()
plt.hist(data['p'], alpha=1.0)
plt.title(title)
plt.xlabel(r'p')
plt.ylabel(r'')
# plt.ylim([0, 1])
plt.show()
plt.savefig(title + '_hist.png')


data['p'] = pd.to_numeric(data['p'])

id_max = data['p'].idxmax()

print('Maximum values:\nw = ', data.at[id_max, 'w'], '\np = ', data.at[id_max, 'p'])
