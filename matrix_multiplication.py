#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:25:50 2023

@author: santiago
"""


import numpy as np
import mixed_states as ms

d = 3
eta = 0.5

rho = ms.density_matrix('GHZ', d)

id_2 = np.diag([1, 1])
K_0 = np.array([[1, 0], [0, np.sqrt(1-eta)]])
K_1 = np.array([[0, np.sqrt(eta)], [0, 0]])

K = np.kron(K_1, np.kron(K_1, K_0))

K_rho = np.matmul(K, rho)
K_rho_Kt = np.matmul(np.matmul(K, rho), np.conjugate(np.transpose(K)))

K = np.kron(id_2, np.kron(id_2, K_1))

import kraus_operators as ko

KO = ko.amplitude_damping_Kraus_operators(d, eta)
