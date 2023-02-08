#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:34:05 2023

@author: santiago
"""

# from sympy.solvers import solve
# from sympy import Symbol

# x = Symbol('x')
# y = Symbol('y')
# z = Symbol('z')

import sympy as sym
x,y,z = sym.symbols('x,y,z')
eq1 = sym.Eq(x+y-(x+y)**2, z+y-(z+y)**2)
eq2 = sym.Eq(x,(x+y)*(x-z))
result = sym.solve([eq1,eq2],(x,y,z))

for i in range(len(result)):
    print(result[i])

#%%

import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(0, 1, 100)

solution_1 = [z - np.sqrt(8*z + 1)/2 + 1/2, -z + np.sqrt(8*z + 1)/4 + 1/4, z]
solution_2 = [z + np.sqrt(8*z + 1)/2 + 1/2, -z - np.sqrt(8*z + 1)/4 + 1/4, z]

plt.close('all')

plt.figure()
plt.title('Solution 1')
plt.plot(z,solution_1[0], label='x')
plt.plot(z,solution_1[1], label='y')
plt.plot(z,solution_1[2], label='z')
plt.legend()
plt.grid()
plt.savefig('solution_1.png')

plt.figure()
plt.title('Solution 2')
plt.plot(z,solution_2[0], label='x')
plt.plot(z,solution_2[1], label='y')
plt.plot(z,solution_2[2], label='z')
plt.legend()
plt.grid()
plt.savefig('solution_2.png')

