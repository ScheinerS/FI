#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:13:48 2023

@author: santiago
"""

import numpy as np
import matplotlib.pyplot as plt

M = np.load('density_matrix.npy')

R = np.real(M)
I = np.real(M)

plt.imshow(R)
plt.title('Re')

plt.figure()
plt.title('Im')
plt.imshow(I)

