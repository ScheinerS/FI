#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:34:23 2023

@author: santiago
"""

#%%

# 2023-02-20

import sympy as sym

x0,y0,x1,y1,x2,y2,x3,y3 = sym.symbols('x0,y0,x1,y1,x2,y2,x3,y3')

eq1 = sym.Eq(x0*x1 + y0*y1 + x2*x3 + y2*y3, 0)
eq2 = sym.Eq(x1*x3 + y1*y3 + x0*x2 + y0*y2, 0)
eq3 = sym.Eq(x1*x2 + y1*y2, 0)

eq4 = sym.Eq(x0*y1 - x1*y0 + x2*y3 - x3*y2, 0)
eq5 = sym.Eq(x1*y3 - x3*y1 + x0*y2 - x2*y0, 0)
eq6 = sym.Eq(x2*y1 - x1*y2, 0)

result = sym.solve([eq1,eq2,eq3,eq4,eq5,eq6],(x0,y0,x1,y1,x2,y2,x3,y3))

for i in range(len(result)):
    print(result[i])

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

y1 = np.linspace(0, 1, 100)
y2 = np.linspace(0, 1, 100)

# y1 = 0.5
# y2 = 0.5

solution_1 = [-y1/2 - y2/2 - np.sqrt(y1**2 + 6*y1*y2 + y2**2)/2, y1, -y1/2 - y2/2 + np.sqrt(y1**2 + 6*y1*y2 + y2**2)/2, y2]
solution_2 = [-y1/2 - y2/2 + np.sqrt(y1**2 + 6*y1*y2 + y2**2)/2, y1, -y1/2 - y2/2 - np.sqrt(y1**2 + 6*y1*y2 + y2**2)/2, y2]

#%%
plt.close('all')

plt.figure()
plt.title('Solution 1')
plt.plot(solution_1[0], solution_1[1], label='alpha_1')
plt.plot(solution_1[2], solution_1[3], label='alpha_2')
# plt.plot(y1,solution_1[1], label='y1')
# plt.plot(y1,solution_1[2], label='x2')
# plt.plot(y1,solution_1[3], label='y2')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid()
plt.show()
plt.savefig('solution_1.png')

# plt.figure()
# plt.title('Solution 2')
# plt.plot(z,solution_2[0], label='x')
# plt.plot(z,solution_2[1], label='y')
# plt.plot(z,solution_2[2], label='z')
# plt.legend()
# plt.grid()
# plt.savefig('solution_2.png')