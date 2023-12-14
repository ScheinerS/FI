#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:45:54 2023

@author: santiago
"""

import numpy as np
    
sigma_z = np.diag([1, -1])
id_2 = np.diag([1, 1])

a = np.matrix('-3 5 7; 2 6 4; 0 2 8')

np.linalg.eig(a)

np.linalg.norm(a, 1)
np.linalg.norm(a, np.inf)
