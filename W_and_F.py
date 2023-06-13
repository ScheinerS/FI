# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Helvetica"
})

# plt.close('all')

d = 3

F = -(1/d)*np.ones((d, d)) # whole matrix
np.fill_diagonal(F, 1) # replace the diagonal

n = 10

p = {}
for a in range(n):
    for b in range(n):
        for c in range(n):
            w = np.array([a, b, c])

            W = np.outer(w, w)

            p[a, b, c] = np.trace(np.dot(W, F))/(np.trace(F)*np.trace(W))
            
plt.plot(p.values())
plt.xlabel(r'w_{i}')
plt.ylabel(r'p')
